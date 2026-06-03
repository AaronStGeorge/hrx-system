# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ Bazel benchmark macros for IREE repositories."""

load(":cc.bzl", "iree_cc_binary")
load(":cc_attrs.bzl", "cc_attrs")
load(":executable.bzl", "iree_executable_test")

_SMOKE_TEST_ARGS = ["--benchmark_min_time=0s"]

def _iree_cc_benchmark_impl(
        name,
        visibility,
        srcs,
        deps,
        data,
        copts,
        conlyopts,
        cxxopts,
        defines,
        local_defines,
        includes,
        system_includes,
        linkopts,
        linkstatic,
        args,
        env,
        resource_group,
        size,
        tags,
        testonly,
        timeout,
        **kwargs):
    iree_cc_binary(
        name = name,
        visibility = visibility,
        srcs = srcs,
        deps = deps,
        data = data,
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        system_includes = system_includes,
        linkopts = linkopts,
        linkstatic = linkstatic,
        tags = tags,
        testonly = testonly,
        **kwargs
    )

    smoke_test_attrs = cc_attrs.merge_dicts(kwargs, {
        "args": _SMOKE_TEST_ARGS + args,
        "data": data,
        "env": env,
        "size": size,
        "src": name,
        "tags": cc_attrs.with_resource_group_tags(tags, resource_group),
        "visibility": visibility,
    })
    if timeout:
        smoke_test_attrs["timeout"] = timeout
    iree_executable_test(
        name = name + "_test",
        **smoke_test_attrs
    )

iree_cc_benchmark = macro(
    doc = """Defines a C/C++ benchmark binary and smoke test.

    The benchmark binary is emitted as `<name>`. A companion `<name>_test`
    target runs the benchmark with `--benchmark_min_time=0s` so ordinary test
    runs verify that the benchmark starts without collecting timing data.

    This shared macro does not select a benchmark framework. Project-specific
    wrappers add the framework dependency that matches their benchmark harness.

    `resource_group` serializes the generated smoke test when the benchmark
    touches a scarce local resource such as a GPU.
    """,
    implementation = _iree_cc_benchmark_impl,
    inherit_attrs = "common",
    attrs = cc_attrs.merge_dicts(
        cc_attrs.compilation,
        cc_attrs.dependency,
        cc_attrs.binary_source,
        cc_attrs.link,
        {
            "args": attr.string_list(
                doc = "Command-line arguments passed to the generated benchmark smoke test.",
            ),
            "env": attr.string_dict(
                doc = "Environment variables passed to the generated benchmark smoke test.",
            ),
            "resource_group": attr.string(
                configurable = False,
                doc = "Local resource name used to serialize the generated smoke test.",
            ),
            "size": attr.string(
                configurable = False,
                default = "small",
                doc = "Bazel test size for the generated benchmark smoke test.",
            ),
            "testonly": attr.bool(
                configurable = False,
                default = True,
                doc = "Whether the benchmark binary is only available to test targets.",
            ),
            "timeout": attr.string(
                configurable = False,
                doc = "Bazel timeout for the generated benchmark smoke test.",
            ),
        },
    ),
)
