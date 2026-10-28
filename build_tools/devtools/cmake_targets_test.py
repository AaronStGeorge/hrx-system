# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import cmake_targets


class CMakeTargetsTest(unittest.TestCase):
    def test_resolve_executable_by_target_alias_and_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            executable = build / "output with spaces/loom-compile.exe"
            self.write_map(build, {"package_compiler": executable, "library": None})
            cmake_targets.target_aliases_path(build).write_text(
                json.dumps({"loom::compiler": "package_compiler"}), encoding="utf-8"
            )
            for name in (
                "package_compiler",
                "loom::compiler",
                "loom-compile",
                "loom-compile.exe",
            ):
                with self.subTest(name=name):
                    target = cmake_targets.resolve_executable(build, name)
                    self.assertEqual(target.name, "package_compiler")
                    self.assertEqual(target.path, executable)

            with mock.patch.object(
                cmake_targets,
                "executable_targets",
                wraps=cmake_targets.executable_targets,
            ) as executable_targets:
                self.assertEqual(
                    cmake_targets.resolve_target_names(
                        build, ["loom-compile", "loom-compile"]
                    ),
                    ["package_compiler", "package_compiler"],
                )
                executable_targets.assert_called_once_with(build)

    def test_resolve_target_rejects_ambiguous_artifact_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            self.write_map(
                build,
                {
                    "first_tool": build / "first/tool.exe",
                    "second_tool": build / "second/tool.exe",
                },
            )
            with self.assertRaisesRegex(
                cmake_targets.CMakeMetadataError,
                "matching targets: first_tool, second_tool",
            ):
                cmake_targets.resolve_target_name(build, "tool")

    def test_exact_non_executable_target_wins_over_artifact_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            self.write_map(build, {"tool": None, "package_tool": build / "tool.exe"})
            self.assertEqual(cmake_targets.resolve_target_name(build, "tool"), "tool")

    def test_only_active_configurations_contribute_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            self.write_map(build, {"obsolete": build / "Debug/tool.exe"}, "Debug")
            self.write_map(build, {"tool": build / "Release/tool.exe"}, "Release")
            self.assertEqual(cmake_targets.configured_target_names(build), {"tool"})
            self.assertEqual(
                cmake_targets.resolve_executable(build, "tool").path,
                build / "Release/tool.exe",
            )
            cmake_targets.target_configurations_path(build).write_text(
                "targets-Debug.txt\ntargets-Release.txt\n", encoding="utf-8"
            )
            self.assertEqual(
                cmake_targets.configured_target_names(build), {"obsolete", "tool"}
            )

    def test_unconfigured_build_preserves_native_names_but_cannot_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            self.assertEqual(cmake_targets.resolve_target_name(build, "tool"), "tool")
            with self.assertRaisesRegex(
                cmake_targets.CMakeMetadataError, "configure first"
            ):
                cmake_targets.resolve_executable(build, "tool")

    def write_map(
        self, build: Path, targets: dict[str, Path | None], configuration="Release"
    ):
        index = cmake_targets.target_configurations_path(build)
        index.parent.mkdir(parents=True, exist_ok=True)
        filename = f"targets-{configuration}.txt"
        index.write_text(filename + "\n", encoding="utf-8")
        (index.parent / filename).write_text(
            "".join(
                f"{name}\t{artifact}\n" if artifact else f"{name}\n"
                for name, artifact in targets.items()
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
