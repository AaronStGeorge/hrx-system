# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transitive archives retaining C++ link and ThinLTO producer metadata.

Derived from rules_cc cc_static_library. Compiler actions, toolchains, and
provider identities remain owned by the selected rules_cc module.
"""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

# These constructors retain producer-owned ThinLTO backends. Recreating a
# LibraryToLink through cc_common would register a second set of actions.
# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/compile:lto_compilation_context.bzl", "merge_lto_compilation_contexts")

# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/link:create_library_to_link.bzl", "make_library_to_link")

def _declare_static_library(*, name, actions, feature_configuration):
    basename = paths.basename(name)

    # Public artifact spelling belongs to this rule. Toolchain-private naming
    # APIs are not available to module-owned rules; the archiver accepts the
    # explicit output path through its normal link variables.
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "targets_windows"):
        new_basename = basename + ".lib"
    else:
        new_basename = "lib" + basename + ".a"
    return actions.declare_file(name.removesuffix(basename) + new_basename)

def _collect_linker_inputs(deps):
    transitive_linker_inputs = [dep[CcInfo].linking_context.linker_inputs for dep in deps]
    return depset(transitive = transitive_linker_inputs, order = "topological")

def _collect_archive_inputs(linker_inputs):
    # Preserve the metadata for exactly the object variant placed in the archive.
    # ThinLTO consumers need the original bitcode, compile flags, and reusable
    # non-indexed backends; importing the archive alone loses these identities.
    objects = []
    lto_contexts = []
    all_pic = True
    must_keep_debug = False
    shared_non_lto_backends = {}
    libraries = []
    user_link_flags = []
    additional_inputs = []
    linkstamps = []
    for linker_input in linker_inputs.to_list():
        user_link_flags.extend(linker_input.user_link_flags)
        additional_inputs.extend(linker_input.additional_inputs)
        linkstamps.extend(linker_input.linkstamps)
        for lib in linker_input.libraries:
            if not lib._contains_objects:
                libraries.append(lib)
                continue
            must_keep_debug = must_keep_debug or lib._must_keep_debug
            if lib.pic_objects:
                objects.append(depset(lib.pic_objects))
                lto_context = lib._pic_lto_compilation_context
                shared_non_lto_backends.update(lib._pic_shared_non_lto_backends or {})
            else:
                all_pic = all_pic and not lib.objects
                objects.append(depset(lib.objects))
                lto_context = lib._lto_compilation_context
                shared_non_lto_backends.update(lib._shared_non_lto_backends or {})
            if lto_context:
                lto_contexts.append(lto_context)
    return struct(
        # Exact archived object files in link order.
        objects = depset(transitive = objects, order = "topological"),
        # Whether every archived object came from a PIC variant.
        all_pic = all_pic,
        # Whether an archived component requires debug information at final link.
        must_keep_debug = must_keep_debug,
        # Original minimized bitcode and backend flags for the selected objects.
        lto_context = merge_lto_compilation_contexts(lto_compilation_contexts = lto_contexts),
        # Producer-owned backend artifacts for links without per-binary indexing.
        shared_non_lto_backends = shared_non_lto_backends,
        # Native libraries whose contents were not folded into the archive.
        libraries = libraries,
        # Ordered flags from both archived and residual linker inputs.
        user_link_flags = user_link_flags,
        # Non-code inputs required by the final link.
        additional_inputs = depset(additional_inputs),
        # Link-time compilations retained from the component linking contexts.
        linkstamps = depset(linkstamps),
    )

def _archive_objects(*, name, actions, cc_toolchain, feature_configuration, objects):
    static_library = _declare_static_library(
        name = name,
        actions = actions,
        feature_configuration = feature_configuration,
    )

    archiver_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
    )
    archiver_variables = cc_common.create_link_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        output_file = static_library.path,
        is_using_linker = False,
    )
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
        variables = archiver_variables,
    )
    args = actions.args()
    args.add_all(command_line)
    args.add_all(objects)

    if cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "archive_param_file",
    ):
        # archive_param_file uses the compiler driver's response-file syntax.
        args.use_param_file("@%s", use_always = True)

    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
        variables = archiver_variables,
    )
    execution_requirements_keys = cc_common.get_execution_requirements(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
    )

    actions.run(
        executable = archiver_path,
        arguments = [args],
        env = env,
        execution_requirements = {k: "" for k in execution_requirements_keys},
        inputs = depset(transitive = [cc_toolchain.all_files, objects]),
        outputs = [static_library],
        use_default_shell_env = True,
        mnemonic = "CppTransitiveArchive",
        progress_message = "Creating static library %{output}",
    )

    return static_library

def _validate_static_library(*, name, actions, cc_toolchain, feature_configuration, static_library):
    if not cc_common.action_is_enabled(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.validate_static_library,
    ):
        return None

    validation_output = actions.declare_file(name + "_validation_output.txt")

    validator_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.validate_static_library,
    )
    args = actions.args()
    args.add(static_library)
    args.add(validation_output)

    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.validate_static_library,
        variables = cc_common.empty_variables(),
    )

    execution_requirements_keys = cc_common.get_execution_requirements(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.validate_static_library,
    )

    actions.run(
        executable = validator_path,
        arguments = [args],
        env = env,
        execution_requirements = {k: "" for k in execution_requirements_keys},
        inputs = depset(
            direct = [static_library],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [validation_output],
        use_default_shell_env = True,
        mnemonic = "ValidateStaticLibrary",
        progress_message = "Validating static library %{label}",
    )

    return validation_output

_MAIN_REPOSITORY_LABEL_PREFIXES = ("@@//", "@//")  # buildifier: disable=canonical-repository

def _pretty_label(label):
    s = str(label)

    # Emit main repo labels (both with and without --enable_bzlmod) without a
    # repo prefix.
    if s.startswith(_MAIN_REPOSITORY_LABEL_PREFIXES):
        return s.lstrip("@")
    return s

def _linkdeps_map_each(linker_input):
    has_library = False
    for lib in linker_input.libraries:
        if lib._contains_objects:
            # Has been added to the archive.
            continue
        if lib.pic_static_library != None or lib.static_library != None or lib.dynamic_library != None or lib.interface_library != None:
            has_library = True
    if not has_library:
        # Does not provide any linkable artifact. May still contribute to linkopts.
        return None

    return _pretty_label(linker_input.owner)

def _linkopts_map_each(linker_input):
    return linker_input.user_link_flags

def _format_linker_inputs(*, actions, name, linker_inputs, map_each):
    file = actions.declare_file(name)
    args = actions.args().add_all(linker_inputs, map_each = map_each)
    actions.write(output = file, content = args)
    return file

def _cc_static_library_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features + ["symbol_check"],
        unsupported_features = ctx.disabled_features,
    )

    linker_inputs = _collect_linker_inputs(ctx.attr.deps)
    archive_inputs = _collect_archive_inputs(linker_inputs)

    static_library = _archive_objects(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        objects = archive_inputs.objects,
    )

    linkdeps_file = _format_linker_inputs(
        actions = ctx.actions,
        name = ctx.label.name + "_linkdeps.txt",
        linker_inputs = linker_inputs,
        map_each = _linkdeps_map_each,
    )

    linkopts_file = _format_linker_inputs(
        actions = ctx.actions,
        name = ctx.label.name + "_linkopts.txt",
        linker_inputs = linker_inputs,
        map_each = _linkopts_map_each,
    )

    validation_output = _validate_static_library(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        static_library = static_library,
    )

    output_groups = {
        "linkdeps": depset([linkdeps_file]),
        "linkopts": depset([linkopts_file]),
    }
    if validation_output:
        output_groups["_validation"] = depset([validation_output])

    runfiles = ctx.runfiles().merge_all([
        dep[DefaultInfo].default_runfiles
        for dep in ctx.attr.deps
    ])

    # A single archive may contain ordinary objects when a dependency has no
    # PIC variant. Only advertise it as PIC when every archived object is PIC.
    if archive_inputs.all_pic:
        library_fields = dict(
            pic_static_library = static_library,
            pic_objects = archive_inputs.objects.to_list(),
            _pic_lto_compilation_context = archive_inputs.lto_context,
            _pic_shared_non_lto_backends = archive_inputs.shared_non_lto_backends,
        )
    else:
        library_fields = dict(
            static_library = static_library,
            objects = archive_inputs.objects.to_list(),
            _lto_compilation_context = archive_inputs.lto_context,
            _shared_non_lto_backends = archive_inputs.shared_non_lto_backends,
        )
    library = make_library_to_link(
        _library_identifier = paths.replace_extension(static_library.short_path, ""),
        _contains_objects = True,
        _must_keep_debug = archive_inputs.must_keep_debug,
        **library_fields
    )

    # Exporting this archive also exports its residual native requirements.
    # Keep them attached to its ownership rather than separate component labels.
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([cc_common.create_linker_input(
            owner = ctx.label,
            libraries = depset([library] + archive_inputs.libraries),
            user_link_flags = archive_inputs.user_link_flags,
            additional_inputs = archive_inputs.additional_inputs,
            linkstamps = archive_inputs.linkstamps,
        )]),
    )

    return [
        CcInfo(linking_context = linking_context),
        DefaultInfo(
            files = depset([static_library]),
            runfiles = runfiles,
        ),
        OutputGroupInfo(**output_groups),
    ]

iree_cc_static_library = rule(
    implementation = _cc_static_library_impl,
    doc = """Combines transitive implementation objects into one linkable archive.

PIC objects take precedence over ordinary objects. The CcInfo retains their
ThinLTO metadata and residual native link requirements, but not compilation
headers. Output groups expose linkdeps, linkopts, and toolchain validation.
""",
    attrs = {
        "deps": attr.label_list(
            providers = [CcInfo],
            doc = """
The list of targets to combine into a static library, including all their transitive
dependencies.

<p>Dependencies that do not provide any object files are not included in the static
library, but their labels are collected in the file provided by the
<code>linkdeps</code> output group.</p>
""",
        ),
    },
    provides = [CcInfo],
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
)
