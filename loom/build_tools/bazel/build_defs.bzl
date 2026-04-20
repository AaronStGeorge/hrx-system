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
        cmake_generator,
        source,
        header,
        args = [],
        inputs = [],
        cmake_generator_deps = [],
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
      generator: Bazel executable label that writes C/H outputs.
      cmake_generator: Source-file label for the same generator script in CMake.
      source: Generated C source filename.
      header: Generated C header filename.
      args: Generator arguments before the output flags. Arguments may use
        $(rootpath <label>) for declared inputs.
      inputs: Source/vendor data labels consumed by the generator.
      cmake_generator_deps: Source-file labels that should retrigger CMake
        generation when the generator implementation changes.
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

    iree_genrule(
        name = name + "_gen",
        srcs = inputs,
        outs = [
            source,
            header,
        ],
        cmd = " ".join(
            ["$(location %s)" % generator] +
            args + [
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

    _ignore = (cmake_generator, cmake_generator_deps, exclude_from_cmake_all)

def loom_low_descriptor_cc_library(
        name,
        generator,
        cmake_generator,
        source,
        header,
        args = [],
        inputs = [],
        cmake_generator_deps = [],
        deps = [],
        exclude_from_cmake_all = False,
        tags = [],
        testonly = False,
        visibility = None,
        **kwargs):
    """Generates a low descriptor C/H shard and wraps it in a runtime library."""
    loom_target_table_cc_library(
        name = name,
        generator = generator,
        cmake_generator = cmake_generator,
        source = source,
        header = header,
        args = args,
        inputs = inputs,
        cmake_generator_deps = cmake_generator_deps,
        deps = deps,
        exclude_from_cmake_all = exclude_from_cmake_all,
        tags = tags,
        testonly = testonly,
        visibility = visibility,
        **kwargs
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
