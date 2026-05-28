# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HRX-specific C/C++ Bazel macros."""

load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_binary", "iree_runtime_cc_library")
load("//runtime/build_tools/bazel:cc_benchmark.bzl", "iree_runtime_cc_benchmark")
load("//runtime/build_tools/bazel:cc_test.bzl", "iree_runtime_cc_test")

_LIBHRX_DEPS = [
    "//libhrx:defines",
]

def _with_libhrx_deps(deps):
    if deps == None:
        deps = []
    return deps + _LIBHRX_DEPS

def hrx_cc_library(name, deps = None, **kwargs):
    iree_runtime_cc_library(
        name = name,
        deps = _with_libhrx_deps(deps),
        **kwargs
    )

def hrx_cc_binary(name, deps = None, **kwargs):
    iree_runtime_cc_binary(
        name = name,
        deps = _with_libhrx_deps(deps),
        **kwargs
    )

def hrx_cc_test(name, deps = None, **kwargs):
    iree_runtime_cc_test(
        name = name,
        deps = _with_libhrx_deps(deps),
        **kwargs
    )

def hrx_cc_benchmark(name, deps = None, **kwargs):
    iree_runtime_cc_benchmark(
        name = name,
        deps = _with_libhrx_deps(deps),
        **kwargs
    )

def hrx_cc_shared_library(
        name,
        srcs = None,
        hdrs = None,
        textual_hdrs = None,
        deps = None,
        copts = None,
        **kwargs):
    """Builds an HRX shared-library artifact under Bazel.

    bazel_to_cmake has a matching handler that emits an iree_cc_library(SHARED)
    target from the same BUILD call.
    """
    if srcs == None:
        srcs = []
    if hdrs == None:
        hdrs = []
    if textual_hdrs == None:
        textual_hdrs = []
    if copts == None:
        copts = []
    iree_runtime_cc_binary(
        name = name,
        srcs = srcs + hdrs + textual_hdrs,
        deps = _with_libhrx_deps(deps),
        copts = copts,
        linkshared = True,
        **kwargs
    )

