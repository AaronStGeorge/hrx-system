# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authored source inventory for sanitizer semantic conformance."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

SANITIZER_CORPUS = loom_corpus_manifest(
    name = "sanitizer",
    package = "//loom/src/loom/test/corpus/sanitizer",
    srcs = [
        "access_reporting.loom",
        "assertion_reporting.loom",
        "race_capacity.loom",
        "race_subgroup.loom",
        "race_workgroup.loom",
    ],
)
