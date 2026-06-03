# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel rules for JSON execution tests."""

load("@rules_python//python:defs.bzl", "py_test")

def iree_execution_test_suite(
        name,
        manifests,
        tools,
        data = None,
        args = None,
        **kwargs):
    """Declares a JSON execution test suite.

    Args:
      name: Bazel target name.
      manifests: JSON manifest files to run.
      tools: Dictionary mapping manifest tool names to executable labels.
      data: Additional runtime data files made available to manifests.
      args: Additional runner arguments.
      **kwargs: Extra py_test attributes.
    """
    data = list(data or [])
    args = list(args or [])
    tool_labels = []
    runner_args = []
    for manifest in manifests:
        runner_args.append("--manifest=$(rootpath %s)" % manifest)
    for tool_name, tool_label in sorted(tools.items()):
        tool_labels.append(tool_label)
        runner_args.append("--tool=%s=$(rootpath %s)" % (tool_name, tool_label))
    py_test(
        name = name,
        srcs = [
            "//build_tools/testing:execution.py",
            "//build_tools/testing:execution_main.py",
        ],
        args = runner_args + args,
        data = list(manifests) + data + tool_labels,
        main = "//build_tools/testing:execution_main.py",
        **kwargs
    )
