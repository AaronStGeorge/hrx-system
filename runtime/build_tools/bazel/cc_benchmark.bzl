# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ Bazel benchmark macros."""

load("//build_tools/bazel:cc_benchmark.bzl", "iree_cc_benchmark")
load(
    "//runtime/requirements:package_policy.bzl",
    "apply_runtime_test_policy",
)
load(":cc.bzl", "iree_runtime_cc_library")
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
    kwargs = apply_runtime_test_policy(kwargs)
    if deps == None:
        deps = []
    compiler_options = runtime_cc_attrs.with_runtime_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        features = kwargs.pop("features", None),
    )
    iree_cc_benchmark(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        features = compiler_options.features,
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

# The disabled implementation is consumed only by the native build emitter.
# buildifier: disable=unused-variable
def iree_runtime_benchmark_library(name, srcs, deps, disabled_srcs, disabled_deps, **kwargs):
    """Declares the runtime benchmark adapter and its uninstrumented implementation.

    Bazel builds requested benchmark targets explicitly and uses the full adapter.
    Native builds may disable benchmark support for the entire runtime, including
    threadless deployments. Both implementations have their source/dependency
    declarations here; the native configuration chooses the implementation.

    Args:
      name: Public adapter library.
      srcs: Instrumented implementation sources.
      deps: Instrumented implementation dependencies.
      disabled_srcs: Implementation when native benchmarking is disabled.
      disabled_deps: Dependencies when native benchmarking is disabled.
      **kwargs: Shared runtime library attributes.
    """
    iree_runtime_cc_library(name = name, srcs = srcs, deps = deps, **kwargs)
