# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared C/C++ Bazel macros."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:cc.bzl", "iree_cc_binary", "iree_cc_library")
load("//build_tools/bazel:cc_test.bzl", "iree_cc_test")

def _all_compilation_paths(compilation_context):
    return [
        str(path)
        for path in (
            compilation_context.includes.to_list() +
            compilation_context.quote_includes.to_list() +
            compilation_context.system_includes.to_list()
        )
    ]

def _expect_path_suffix(env, paths, suffix):
    for path in paths:
        if path.endswith(suffix):
            return
    env.fail("expected one of %s to end with %r" % (paths, suffix))

def _test_cc_library_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_library,
        name = name + "_subject",
        hdrs = [name + "_subject.h"],
        includes = ["public_include"],
        system_includes = ["system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_library_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_library_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "public_include")
    _expect_path_suffix(env, paths, "system_include")

def _test_cc_binary_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        includes = ["binary_include"],
        system_includes = ["binary_system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_binary_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "binary_include")
    _expect_path_suffix(env, paths, "binary_system_include")

def _test_cc_test_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        includes = ["test_include"],
        system_includes = ["test_system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_test_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "test_include")
    _expect_path_suffix(env, paths, "test_system_include")

def _test_cc_test_applies_resource_group_tags(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        resource_group = "gpu",
        tags = [
            "driver=vulkan",
            "manual",
        ],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_test_applies_resource_group_tags_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_applies_resource_group_tags_impl(env, target):
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "driver=vulkan",
        "exclusive-if-local",
        "resource_group:gpu",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def cc_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_cc_library_preserves_system_include_inputs,
            _test_cc_binary_preserves_system_include_inputs,
            _test_cc_test_preserves_system_include_inputs,
            _test_cc_test_applies_resource_group_tags,
        ],
    )
