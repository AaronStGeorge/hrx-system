# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import configure_project

_PROJECT = """\
cmake_minimum_required(VERSION 3.21)
project(loom_target_presets NONE)
set(LOOM_BUILD ON)
include("${IREE_REPO_ROOT}/loom/build_tools/cmake/loom_target_presets.cmake")
loom_resolve_target_presets()
file(WRITE "${CMAKE_BINARY_DIR}/resolved.txt"
  "${LOOM_RESOLVED_ARCHITECTURES}\n"
  "${LOOM_RESOLVED_ARTIFACT_FORMATS}\n"
  "${LOOM_RESOLVED_EXECUTION_SUBSTRATES}\n")
"""


class LoomTargetPresetsTest(unittest.TestCase):
    def test_reconfigure_replaces_every_derived_capability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            build = root / "build"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(_PROJECT, encoding="utf-8")

            steps = (
                (
                    (),
                    "aie2p;amdgpu;spirv;vm;x86\nelf;spirv;vm\niree_hal;iree_vm\n",
                ),
                (
                    ("-DLOOM_TARGET_ALL_CPU=ON",),
                    "vm;x86\nelf;vm\niree_hal;iree_vm\n",
                ),
                (
                    (
                        "-DLOOM_TARGET_ALL_CPU=OFF",
                        "-DLOOM_TARGET_DEFAULT=OFF",
                        "-DLOOM_ARCH_AIE2P=ON",
                        "-DLOOM_EMIT_ELF=ON",
                        "-DLOOM_EXECUTE_IREE_HAL=ON",
                    ),
                    "aie2p\nelf\niree_hal\n",
                ),
                (
                    (
                        "-DLOOM_ARCH_AIE2P=OFF",
                        "-DLOOM_EMIT_ELF=OFF",
                        "-DLOOM_EXECUTE_IREE_HAL=OFF",
                        "-DLOOM_TARGET_ALL_CPU=ON",
                        "-DLOOM_TARGET_ALL_AMD_NPU=ON",
                    ),
                    "aie2p;vm;x86\nelf;vm\niree_hal;iree_vm\n",
                ),
            )
            for arguments, expected in steps:
                configure_project(source, build, *arguments)
                self.assertEqual(
                    (build / "resolved.txt").read_text(encoding="utf-8"),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
