# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import (
    CONFIGURATION,
    REPO_ROOT,
    build_project,
    configure_project,
    run_command,
)
from build_tools.devtools import cmake_targets

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/target_map"


class CMakeTargetMapTest(unittest.TestCase):
    def test_native_target_paths_and_late_targets(self):
        with tempfile.TemporaryDirectory(prefix="cmake target map ") as temporary:
            build = Path(temporary) / "build with spaces"
            configure_project(
                FIXTURE, build, f"-DCMAKE_CONFIGURATION_TYPES={CONFIGURATION};Debug"
            )
            self.assertEqual(
                cmake_targets.configured_target_names(build),
                {"fixture_program", "late_target", "generated_headers"},
            )
            build_project(build, "fixture_program")
            for name in ("fixture_program", "fixture::tool", "named tool"):
                target = cmake_targets.resolve_executable(build, name)
                self.assertTrue(target.path.is_file())
                self.assertEqual(
                    run_command(str(target.path)), "target map executable\n"
                )
            output = run_command(
                sys.executable,
                str(REPO_ROOT / "dev.py"),
                "--system",
                "--cmake-build-dir",
                str(build),
                "cmake",
                "run",
                "fixture::tool",
            )
            self.assertEqual(output, "target map executable\n")

            # Reconfiguration changes both the artifact name and the set of
            # configurations. Old generated files must not affect resolution.
            configure_project(
                FIXTURE,
                build,
                "-DFIXTURE_OUTPUT_NAME=renamed tool",
                "-DCMAKE_BUILD_TYPE=",
                f"-DCMAKE_CONFIGURATION_TYPES={CONFIGURATION}",
            )
            targets = cmake_targets.executable_targets(build)
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].path.stem, "renamed tool")
            build_project(build, "fixture_program")
            self.assertEqual(
                run_command(str(targets[0].path)), "target map executable\n"
            )


if __name__ == "__main__":
    unittest.main()
