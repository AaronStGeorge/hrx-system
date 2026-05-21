# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for configured dependency checks."""

load("//build_tools/bazel:cc.bzl", "iree_cc_library")
load("//build_tools/bazel:query.bzl", "iree_assert_no_dependency")

def query_rules_test_suite(name):
    leaf_name = name + "_leaf"
    root_name = name + "_root"
    unrelated_name = name + "_unrelated"
    absent_test_name = name + "_allows_absent_dependency"

    iree_cc_library(
        name = leaf_name,
        tags = ["manual"],
    )
    iree_cc_library(
        name = unrelated_name,
        tags = ["manual"],
    )
    iree_cc_library(
        name = root_name,
        deps = [":" + leaf_name],
        tags = ["manual"],
    )

    iree_assert_no_dependency(
        name = absent_test_name,
        dependency = ":" + unrelated_name,
        target = ":" + root_name,
    )

    native.test_suite(
        name = name,
        tests = [":" + absent_test_name],
    )
