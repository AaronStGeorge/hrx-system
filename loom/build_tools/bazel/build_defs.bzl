# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom-specific Bazel build macros.

The rules in this file are intentionally scoped to Loom instead of the shared
IREE build layer. They describe target-low descriptor generation: compact C
tables derived from checked-in Loom Python descriptions and, for some targets,
fetched vendor machine-readable data.
"""

load(
    "//loom/build_tools/bazel:build_defs.bzl",
    "iree_genrule",
    "loom_cc_library",
)

def _loom_canonical_query_labels(labels):
    canonical_labels = []
    package_name = native.package_name()
    for label in labels:
        if label.startswith(":"):
            canonical_labels.append("//%s%s" % (package_name, label))
        else:
            canonical_labels.append(label)
    return canonical_labels

def _loom_query_set(labels):
    if len(labels) == 0:
        fail("query label set must not be empty")
    return "set(%s)" % " ".join(labels)

def loom_assert_no_dependencies(
        name,
        targets,
        dependencies,
        tags = [],
        **kwargs):
    """Asserts that target boundary labels do not depend on forbidden labels.

    This is the Loom package-boundary form of iree_assert_no_dependency. Use it
    for architectural isolation checks over public/interface targets, not for
    duplicating every private rule edge in a package. The generated query is a
    single transitive check from the target set to the forbidden dependency set.

    Bazel genquery does not accept wildcard package patterns such as :all or
    ... in a portable way, so callers should pass explicit package boundary
    labels: public interface libraries, provider entry points, or other small
    roots that represent the component contract.

    Args:
      name: Test target name.
      targets: Boundary target labels that must remain independent.
      dependencies: Forbidden dependency labels.
      tags: Additional tags for the test target.
      **kwargs: Additional arguments forwarded to genquery and sh_test.
    """

    canonical_targets = _loom_canonical_query_labels(targets)
    canonical_dependencies = _loom_canonical_query_labels(dependencies)
    query_name = name + "_query"
    native.genquery(
        name = query_name,
        expression = "deps(%s) intersect %s" % (
            _loom_query_set(canonical_targets),
            _loom_query_set(canonical_dependencies),
        ),
        scope = canonical_targets + canonical_dependencies,
        tags = ["manual"],
        **kwargs
    )
    native.sh_test(
        name = name,
        srcs = ["//build_tools/bazel:assert_empty_query.sh"],
        args = ["$(location :%s)" % query_name],
        data = [":%s" % query_name],
        size = "small",
        tags = tags,
        **kwargs
    )

def loom_low_descriptor_data_archive(
        name,
        repo_name,
        urls,
        sha256,
        **kwargs):
    """Declares a fetched data archive used by low descriptor generators.

    Bazel repository fetching is still owned by MODULE.bazel/module extensions.
    This declaration exists so bazel_to_cmake has the same target-local fetch
    metadata without each backend embedding CMake ExternalProject snippets.

    Args:
      name: Package-local archive name used for generated CMake targets.
      repo_name: Bazel external repository name without a leading '@'.
      urls: Archive URLs.
      sha256: Archive SHA-256 digest.
      **kwargs: Reserved for CMake-converted metadata. Ignored by Bazel.
    """

    # The Bazel repository rule is declared from MODULE.bazel; this macro is a
    # target-local CMake conversion contract.
    _ignore = (name, repo_name, urls, sha256, kwargs)

def loom_target_table_cc_library(
        name,
        generator,
        source = None,
        header = None,
        args = [],
        inputs = [],
        deps = [],
        exclude_from_cmake_all = False,
        tags = [],
        testonly = False,
        visibility = None,
        **kwargs):
    """Generates a target-owned C/H table and wraps it in a runtime library.

    This is the common build-system contract for compact target-owned generated
    data. It deliberately abstracts only generation and library wiring. Target
    packages still own vendor schema parsing, overlay selection, semantic tests,
    registry composition, and which shards are selected by default.

    Args:
      name: Runtime C library target name.
      generator: Python executable label that writes C/H outputs. Bazel uses it
        as the genrule tool; bazel_to_cmake resolves the Python target's main
        script and transitive sources for generated CMake dependencies.
      source: Generated C source filename. Defaults to <name>.c.
      header: Generated C header filename. Defaults to <name>.h.
      args: Generator arguments before the output flags. Arguments may use
        $(rootpath <label>) for declared inputs; the Bazel action rewrites
        those to $(execpath <label>) while CMake conversion maps them to source
        paths.
      inputs: Source/vendor data labels consumed by the generator.
      deps: Runtime C library dependencies.
      exclude_from_cmake_all: Excludes the generated C library from CMake all.
      tags: Additional Bazel tags for the internal generator action.
      testonly: Passed through to the runtime C library.
      visibility: Passed through to the generator action and runtime library.
      **kwargs: Additional arguments passed through to loom_cc_library.
    """
    genrule_kwargs = {}
    if visibility != None:
        genrule_kwargs["visibility"] = visibility

    source = source or (name + ".c")
    header = header or (name + ".h")

    # BUILD files use rootpath so the CMake converter can map external data
    # labels to their fetched source directories. Bazel genrules execute in the
    # action sandbox and need execpath to read declared srcs.
    bazel_args = [arg.replace("$(rootpath ", "$(execpath ") for arg in args]

    iree_genrule(
        name = name + "_gen",
        srcs = inputs,
        outs = [
            source,
            header,
        ],
        cmd = " ".join(
            ["$(location %s)" % generator] +
            bazel_args + [
                "--source=$(execpath %s)" % source,
                "--header=$(execpath %s)" % header,
            ],
        ),
        tags = tags + ["skip-bazel_to_cmake"],
        tools = [generator],
        **genrule_kwargs
    )

    package_name = native.package_name()
    generated_source = "//%s:%s" % (package_name, source)
    generated_header = "//%s:%s" % (package_name, header)
    loom_cc_library(
        name = name,
        srcs = [generated_source],
        hdrs = [generated_header],
        deps = deps,
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )
    _ignore = (exclude_from_cmake_all,)

def loom_low_descriptor_cc_library(
        name,
        generator,
        source = None,
        header = None,
        args = [],
        inputs = [],
        deps = [],
        exclude_from_cmake_all = False,
        tags = [],
        testonly = False,
        visibility = None,
        **kwargs):
    """Generates a low descriptor C/H shard and wraps it in a runtime library."""
    source = source or (name + ".c")
    header = header or (name + ".h")
    loom_target_table_cc_library(
        name = name,
        generator = generator,
        source = source,
        header = header,
        args = args,
        inputs = inputs,
        deps = deps,
        exclude_from_cmake_all = exclude_from_cmake_all,
        tags = tags,
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

    package_name = native.package_name()
    generated_header = "//%s:%s" % (package_name, header)
    loom_cc_library(
        name = name + "_ids",
        hdrs = [generated_header],
        deps = deps,
        testonly = testonly,
        visibility = visibility,
    )

def loom_low_descriptor_exclude_from_cmake_all(
        cc_libraries = [],
        targets = []):
    """Excludes descriptor-dependent CMake targets from the default all target.

    Descriptor registries and backend-specific tests often depend on large
    target tables. Keeping those targets declared but outside CMake all lets
    explicit builds and tests work without making every default build fetch and
    compile every backend descriptor package.

    Args:
      cc_libraries: iree_cc_library names in the current package.
      targets: Other CMake target names in the current package, usually tests.
    """
    _ignore = (cc_libraries, targets)
