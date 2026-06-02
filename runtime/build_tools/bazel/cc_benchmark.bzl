# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ Bazel benchmark macros."""

load("//build_tools/bazel:cc_benchmark.bzl", "iree_cc_benchmark")
load(":cc_attrs.bzl", "runtime_cc_attrs")

_GOOGLE_BENCHMARK_DEP = Label("//third_party:google_benchmark")

def _iree_runtime_cc_benchmark_impl(
        name,
        visibility,
        copts,
        conlyopts,
        cxxopts,
        deps,
        **kwargs):
    if deps == None:
        deps = []
    compiler_options = runtime_cc_attrs.with_runtime_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
    )
    iree_cc_benchmark(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        deps = runtime_cc_attrs.with_runtime_deps(deps + [_GOOGLE_BENCHMARK_DEP]),
        **kwargs
    )

iree_runtime_cc_benchmark = macro(
    doc = """Defines a runtime C/C++ benchmark binary and smoke test.

    Runtime benchmarks use Google Benchmark through the shared
    `//third_party:google_benchmark` dependency facade.
    """,
    implementation = _iree_runtime_cc_benchmark_impl,
    inherit_attrs = iree_cc_benchmark,
    attrs = {},
)
