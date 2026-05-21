# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ Bazel test macros."""

load("//build_tools/bazel:cc_test.bzl", "iree_cc_test")
load(":cc_attrs.bzl", "runtime_cc_attrs")

def _iree_runtime_cc_test_impl(
        name,
        visibility,
        deps,
        **kwargs):
    iree_cc_test(
        name = name,
        visibility = visibility,
        deps = runtime_cc_attrs.with_runtime_deps(deps),
        **kwargs
    )

iree_runtime_cc_test = macro(
    doc = """Defines a runtime C/C++ test target.""",
    implementation = _iree_runtime_cc_test_impl,
    inherit_attrs = iree_cc_test,
    attrs = {},
)
