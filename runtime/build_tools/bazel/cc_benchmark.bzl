# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ Bazel benchmark macros."""

load("//build_tools/bazel:cc_benchmark.bzl", "iree_cc_benchmark")
load(":cc_attrs.bzl", "runtime_cc_attrs")

def _iree_runtime_cc_benchmark_impl(name, visibility, deps, **kwargs):
    iree_cc_benchmark(
        name = name,
        visibility = visibility,
        deps = runtime_cc_attrs.with_runtime_deps(deps),
        **kwargs
    )

iree_runtime_cc_benchmark = macro(
    doc = """Defines a runtime C/C++ benchmark binary and smoke test.""",
    implementation = _iree_runtime_cc_benchmark_impl,
    inherit_attrs = iree_cc_benchmark,
    attrs = {},
)
