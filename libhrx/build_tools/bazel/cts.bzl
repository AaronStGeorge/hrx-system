# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HRX CTS Bazel macros."""

load("//libhrx/build_tools/bazel:cc.bzl", "hrx_cc_test")

def hrx_cts_test(name, srcs, deps = None, data = None, args = None, **kwargs):
    if deps == None:
        deps = []
    if data == None:
        data = []
    if args == None:
        args = []
    hrx_cc_test(
        name = "hrx_cts_" + name,
        srcs = srcs,
        deps = [
            ":core",
            "@catch2//:catch2",
        ] + deps,
        data = [
            "//libhrx/src/libhrx:hrx",
        ] + data,
        args = [
            "--hrx-library",
            "$(location //libhrx/src/libhrx:hrx)",
        ] + args,
        **kwargs
    )

