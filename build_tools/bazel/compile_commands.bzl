# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository-owned C/C++ compile command extraction.

This file owns build-system introspection shared by clang tooling, IDE support,
and source-analysis tools. It is intentionally independent of clang-tidy.
"""

load("@rules_cc//cc:action_names.bzl", "CPP_COMPILE_ACTION_NAME", "C_COMPILE_ACTION_NAME")
load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_ATTRS", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

IreeCompileCommandsInfo = provider(
    doc = "C/C++ compile command metadata collected from configured targets.",
    fields = {
        "entries": "depset of JSON-encoded compile command entries.",
        "fragments": "depset of JSON fragment files containing compile command entries.",
    },
)

_C_SOURCE_EXTENSIONS = (".c",)
_CXX_SOURCE_EXTENSIONS = (".cc", ".cpp", ".cxx", ".c++", ".C")

def _sanitize_label(label):
    result = str(label)
    for old, new in [
        ("@", "external_"),
        ("//", ""),
        (":", "__"),
        ("/", "_"),
        ("+", "_"),
        ("-", "_"),
        (".", "_"),
    ]:
        result = result.replace(old, new)
    return result

def _source_language(source):
    if source.extension:
        extension = "." + source.extension
    else:
        extension = ""
    if extension in _C_SOURCE_EXTENSIONS:
        return "c"
    if extension in _CXX_SOURCE_EXTENSIONS:
        return "c++"
    return None

def _is_main_workspace_source(source):
    return source.is_source and source.owner.workspace_name == ""

def _source_files(ctx):
    if not hasattr(ctx.rule.attr, "srcs"):
        return []
    result = []
    for src_target in ctx.rule.attr.srcs:
        for src in src_target.files.to_list():
            if _is_main_workspace_source(src) and _source_language(src) != None:
                result.append(src)
    return result

def _string_list_attr(ctx, name):
    if not hasattr(ctx.rule.attr, name):
        return []
    value = getattr(ctx.rule.attr, name)
    if value == None:
        return []
    return value

def _user_compile_flags(ctx, language):
    flags = []
    flags.extend(_string_list_attr(ctx, "copts"))
    if language == "c":
        flags.extend(_string_list_attr(ctx, "conlyopts"))
    elif language == "c++":
        flags.extend(_string_list_attr(ctx, "cxxopts"))
    return flags

def _compile_action_name(language):
    if language == "c":
        return C_COMPILE_ACTION_NAME
    if language == "c++":
        return CPP_COMPILE_ACTION_NAME
    fail("unsupported source language: %s" % language)

def _compile_variables_extension(feature_configuration, compilation_context):
    variables = {}

    # Bazel/rules_cc currently exposes module-map data through private
    # compilation-context fields while public cc_common.create_compile_variables
    # still needs these variables when the toolchain enables module maps.
    module_maps_enabled = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "module_maps",
    )
    layering_check_enabled = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "layering_check",
    )
    if module_maps_enabled and hasattr(compilation_context, "_module_map"):
        module_map = compilation_context._module_map
        if module_map:
            variables["module_name"] = module_map.name
            variables["module_map_file"] = module_map.file.path

    if layering_check_enabled and hasattr(compilation_context, "_direct_module_maps"):
        variables["dependent_module_map_files"] = [
            module_map.path
            for module_map in compilation_context._direct_module_maps.to_list()
        ]

    return variables

def _compile_command_entry(ctx, target, cc_toolchain, feature_configuration, source):
    language = _source_language(source)
    action_name = _compile_action_name(language)
    compilation_context = target[CcInfo].compilation_context
    variables = cc_common.create_compile_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        source_file = source.path,
        user_compile_flags = _user_compile_flags(ctx, language),
        include_directories = compilation_context.includes,
        quote_include_directories = compilation_context.quote_includes,
        system_include_directories = compilation_context.system_includes,
        framework_include_directories = compilation_context.framework_includes,
        preprocessor_defines = depset(transitive = [
            compilation_context.defines,
            compilation_context.local_defines,
        ]),
        variables_extension = _compile_variables_extension(
            feature_configuration,
            compilation_context,
        ),
    )
    compiler = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = action_name,
    )
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = action_name,
        variables = variables,
    )
    return json.encode({
        "arguments": [compiler] + command_line,
        "directory": ".",
        "file": source.path,
    })

def _collect_compile_commands_aspect_impl(target, ctx):
    transitive_entries = []
    transitive_fragments = []
    for attr_name in [
        "actual",
        "deps",
        "implementation_deps",
    ]:
        if not hasattr(ctx.rule.attr, attr_name):
            continue
        attr_value = getattr(ctx.rule.attr, attr_name)
        if type(attr_value) == type([]):
            deps = attr_value
        elif attr_value:
            deps = [attr_value]
        else:
            deps = []
        for dep in deps:
            if IreeCompileCommandsInfo in dep:
                transitive_entries.append(dep[IreeCompileCommandsInfo].entries)
                transitive_fragments.append(dep[IreeCompileCommandsInfo].fragments)

    local_entries = []
    if CcInfo in target:
        cc_toolchain = find_cc_toolchain(ctx)
        feature_configuration = cc_common.configure_features(
            ctx = ctx,
            cc_toolchain = cc_toolchain,
            requested_features = ctx.features,
            unsupported_features = ctx.disabled_features,
        )
        for source in _source_files(ctx):
            local_entries.append(_compile_command_entry(
                ctx,
                target,
                cc_toolchain,
                feature_configuration,
                source,
            ))

    local_fragments = []
    if local_entries:
        fragment = ctx.actions.declare_file(
            "%s.compile_commands.json" % _sanitize_label(target.label),
        )
        ctx.actions.write(
            output = fragment,
            content = "[\n%s\n]\n" % ",\n".join(local_entries),
        )
        local_fragments.append(fragment)

    compile_commands = IreeCompileCommandsInfo(
        entries = depset(local_entries, transitive = transitive_entries),
        fragments = depset(local_fragments, transitive = transitive_fragments),
    )
    output_groups = OutputGroupInfo(
        iree_compile_commands_fragments = depset(
            local_fragments,
            transitive = transitive_fragments,
        ),
    )
    return [compile_commands, output_groups]

collect_compile_commands_aspect = aspect(
    implementation = _collect_compile_commands_aspect_impl,
    attr_aspects = [
        "actual",
        "deps",
        "implementation_deps",
    ],
    attrs = CC_TOOLCHAIN_ATTRS,
    fragments = ["cpp"],
    required_providers = [CcInfo],
    toolchains = use_cc_toolchain(),
)

def _iree_compile_commands_impl(ctx):
    transitive_entries = []
    transitive_fragments = []
    for target in ctx.attr.targets:
        if IreeCompileCommandsInfo not in target:
            continue
        compile_commands = target[IreeCompileCommandsInfo]
        transitive_entries.append(compile_commands.entries)
        transitive_fragments.append(compile_commands.fragments)

    fragments = depset(transitive = transitive_fragments)
    args = ctx.actions.args()
    args.add(ctx.outputs.out)
    args.add_all(fragments)
    ctx.actions.run(
        executable = ctx.executable._merge_tool,
        arguments = [args],
        inputs = fragments,
        outputs = [ctx.outputs.out],
        mnemonic = "IreeCompileCommands",
        progress_message = "Merging C/C++ compile commands %{output}",
    )

    return [
        DefaultInfo(files = depset([ctx.outputs.out])),
        IreeCompileCommandsInfo(
            entries = depset(transitive = transitive_entries),
            fragments = fragments,
        ),
    ]

iree_compile_commands = rule(
    implementation = _iree_compile_commands_impl,
    attrs = {
        "out": attr.output(
            mandatory = True,
            doc = "Output compile_commands.json file.",
        ),
        "targets": attr.label_list(
            aspects = [collect_compile_commands_aspect],
            doc = "C/C++ targets whose configured compile commands should be emitted.",
            mandatory = True,
            providers = [CcInfo],
        ),
        "_merge_tool": attr.label(
            cfg = "exec",
            default = Label("//build_tools/bazel:compile_commands_merge"),
            executable = True,
        ),
    },
    doc = "Emits a compile_commands.json file for configured C/C++ targets.",
)
