# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that run .loom-test files through loom-check."""

load(
    "//loom/build_tools/bazel:build_defs.bzl",
    "loom_cc_binary",
)
load("//build_tools/bazel:executable.bzl", "iree_executable_test")

_LOOM_CHECK_EXTENSION = ".loom-test"

def _loom_check_test_base_name(src):
    if not src.endswith(_LOOM_CHECK_EXTENSION):
        fail("loom_check_test source must use the .loom-test extension: %s" %
             src)
    return src[:-len(_LOOM_CHECK_EXTENSION)]

def loom_check_runner_binary(name, src, deps = [], **kwargs):
    """Creates a loom-check compatible runner binary.

    Args:
      name: Name of the generated runner binary.
      src: C source file that calls loom_check_production_main().
      deps: Runner-specific descriptor and lowering policy dependencies.
      **kwargs: Additional attributes passed to loom_cc_binary.
    """
    loom_cc_binary(
        name = name,
        srcs = [src],
        deps = deps + ["//loom/src/loom/tools/loom-check:main"],
        **kwargs
    )

def loom_check_test(
        name,
        src,
        size = "small",
        tags = [],
        data = [],
        env = {},
        runner = "//loom/src/loom/tools/loom-check",
        **kwargs):
    """Creates a test that runs a single .loom-test file through loom-check.

    Args:
      name: Name of the generated test.
      src: Source .loom-test file containing the test cases.
      size: Test size (default: "small").
      tags: Additional tags to apply to the test.
      data: Additional runfiles made available to loom-check.
      env: Additional test environment variables.
      runner: loom-check compatible runner binary.
      **kwargs: Additional attributes passed to native_test.
    """
    _loom_check_test_base_name(src)
    iree_executable_test(
        name = name,
        src = runner,
        data = [src] + data,
        args = ["$(rootpath %s)" % src],
        env = env,
        size = size,
        tags = tags + ["loom-check"],
        **kwargs
    )

def loom_check_test_suite(
        name,
        srcs,
        runner = "//loom/src/loom/tools/loom-check",
        **kwargs):
    """Creates one test per .loom-test file, bundled into a test suite.

    Each .loom-test file becomes an independent test target. The test name
    is derived from the file path by replacing "/" with "_" and
    stripping the .loom-test extension.

    Args:
      name: Name of the generated test suite.
      srcs: List of .loom-test files to test.
      runner: loom-check compatible runner binary.
      **kwargs: Additional attributes passed to each loom_check_test
          and the test suite.
    """
    tests = []
    for src in srcs:
        test_name = _loom_check_test_base_name(src).replace("/", "_")
        loom_check_test(name = test_name, src = src, runner = runner, **kwargs)
        tests.append(test_name)
    native.test_suite(name = name, tests = tests, **kwargs)
