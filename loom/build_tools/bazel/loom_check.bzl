# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that run .loom-test files through loom-check."""

load("//build_tools/bazel:executable.bzl", "iree_executable_test")

_LOOM_CHECK_EXTENSION = ".loom-test"

def _loom_check_test_base_name(src):
    if not src.endswith(_LOOM_CHECK_EXTENSION):
        fail("loom_check_test source must use the .loom-test extension: %s" %
             src)
    return src[:-len(_LOOM_CHECK_EXTENSION)]

def loom_check_test(name, src, size = "small", tags = [], **kwargs):
    """Creates a test that runs a single .loom-test file through loom-check.

    Args:
      name: Name of the generated test.
      src: Source .loom-test file containing the test cases.
      size: Test size (default: "small").
      tags: Additional tags to apply to the test.
      **kwargs: Additional attributes passed to native_test.
    """
    _loom_check_test_base_name(src)
    iree_executable_test(
        name = name,
        src = "//loom/src/loom/tools/loom-check",
        data = [src],
        args = ["$(rootpath %s)" % src],
        size = size,
        tags = tags + ["loom-check"],
        **kwargs
    )

def loom_check_test_suite(name, srcs, **kwargs):
    """Creates one test per .loom-test file, bundled into a test suite.

    Each .loom-test file becomes an independent test target. The test name
    is derived from the file path by replacing "/" with "_" and
    stripping the .loom-test extension.

    Args:
      name: Name of the generated test suite.
      srcs: List of .loom-test files to test.
      **kwargs: Additional attributes passed to each loom_check_test
          and the test suite.
    """
    tests = []
    for src in srcs:
        test_name = _loom_check_test_base_name(src).replace("/", "_")
        loom_check_test(name = test_name, src = src, **kwargs)
        tests.append(test_name)
    native.test_suite(name = name, tests = tests, **kwargs)
