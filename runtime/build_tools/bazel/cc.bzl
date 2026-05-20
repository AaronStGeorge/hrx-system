# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ Bazel macros.

These macros encode policy for targets under `//runtime`. Shared mechanics live
in `//build_tools/bazel:cc.bzl`; this file adds only runtime-specific
dependencies and test scheduling metadata.
"""

load(
    "//build_tools/bazel:cc.bzl",
    "iree_cc_binary",
    "iree_cc_library",
    "iree_cc_test",
)

_RUNTIME_DEFINES_DEP = Label("//runtime/src:defines")

def _with_runtime_deps(deps):
    if deps == None:
        deps = []
    return deps + [_RUNTIME_DEFINES_DEP]

def _with_resource_group_tags(tags, resource_group):
    if tags == None:
        tags = []
    if resource_group == None or resource_group == "":
        return tags
    return tags + [
        "exclusive-if-local",
        "resource_group:" + resource_group,
    ]

def _iree_runtime_cc_library_impl(name, visibility, deps, **kwargs):
    iree_cc_library(
        name = name,
        visibility = visibility,
        deps = _with_runtime_deps(deps),
        **kwargs
    )

iree_runtime_cc_library = macro(
    doc = """Defines a runtime C/C++ library target.

    Runtime libraries receive the public runtime include root through
    `//runtime/src:defines`. Callers should depend on this wrapper rather than
    the shared `iree_cc_library` macro unless they are defining the runtime
    root itself.
    """,
    implementation = _iree_runtime_cc_library_impl,
    inherit_attrs = iree_cc_library,
    attrs = {},
)

def _iree_runtime_cc_binary_impl(name, visibility, deps, **kwargs):
    iree_cc_binary(
        name = name,
        visibility = visibility,
        deps = _with_runtime_deps(deps),
        **kwargs
    )

iree_runtime_cc_binary = macro(
    doc = """Defines a runtime C/C++ binary target.""",
    implementation = _iree_runtime_cc_binary_impl,
    inherit_attrs = iree_cc_binary,
    attrs = {},
)

def _iree_runtime_cc_test_impl(
        name,
        visibility,
        deps,
        tags,
        resource_group,
        **kwargs):
    iree_cc_test(
        name = name,
        visibility = visibility,
        deps = _with_runtime_deps(deps),
        tags = _with_resource_group_tags(tags, resource_group),
        **kwargs
    )

iree_runtime_cc_test = macro(
    doc = """Defines a runtime C/C++ test target.

    `resource_group` serializes tests that compete for a named local resource.
    Bazel receives the conservative `exclusive-if-local` tag plus a structured
    `resource_group:<name>` tag that CI and other generators can inspect.
    """,
    implementation = _iree_runtime_cc_test_impl,
    inherit_attrs = iree_cc_test,
    attrs = {
        "resource_group": attr.string(
            configurable = False,
            doc = "Local resource name used to serialize tests competing for the same host resource.",
        ),
    },
)
