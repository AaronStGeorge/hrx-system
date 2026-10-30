# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical LoomC public header inventories."""

LOOMC_BASE_HDRS = [
    "include/loomc/base.h",
    "include/loomc/byte_sequence.h",
    "include/loomc/iree.h",
    "include/loomc/status.h",
]

LOOMC_PUBLIC_HDRS = LOOMC_BASE_HDRS + [
    "include/loomc/artifact.h",
    "include/loomc/artifact_manifest.h",
    "include/loomc/compile.h",
    "include/loomc/compile_report.h",
    "include/loomc/config.h",
    "include/loomc/context.h",
    "include/loomc/diagnostic.h",
    "include/loomc/emit.h",
    "include/loomc/launch_config.h",
    "include/loomc/link.h",
    "include/loomc/link_dependency.h",
    "include/loomc/link_index.h",
    "include/loomc/loomc.h",
    "include/loomc/module.h",
    "include/loomc/module_config.h",
    "include/loomc/pass.h",
    "include/loomc/pass_trace.h",
    "include/loomc/product.h",
    "include/loomc/result.h",
    "include/loomc/sanitizer.h",
    "include/loomc/source.h",
    "include/loomc/task.h",
    "include/loomc/target.h",
    "include/loomc/workspace.h",
]
