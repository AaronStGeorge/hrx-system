# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for checked-glob Bazel helpers."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")

def _expect_basenames(env, files, expected):
    actual = sorted([
        file.basename
        for file in files
    ])
    if actual != expected:
        env.fail("expected basenames %r, got %r" % (expected, actual))

def _test_checked_glob_returns_explicit_files(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_checked_glob_returns_explicit_files_impl,
        target = ":checked_glob_fixture",
        **kwargs
    )

def _test_checked_glob_returns_explicit_files_impl(env, target):
    _expect_basenames(
        env,
        target[DefaultInfo].files.to_list(),
        [
            "checked_glob_alpha.txt",
            "checked_glob_beta.txt",
        ],
    )

def glob_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_checked_glob_returns_explicit_files,
        ],
    )
