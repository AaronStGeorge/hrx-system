# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import shutil
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

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/generated_packages"


class CMakeGeneratedPackagesTest(unittest.TestCase):
    def test_embedded_packages_regenerate_without_source_outputs(self):
        with tempfile.TemporaryDirectory(
            prefix="cmake generated packages "
        ) as temporary:
            root = Path(temporary)
            source = root / "source"
            # Deep build roots and qualified target names exercise Windows
            # object-path limits encountered by embedded builds in CI.
            build = root / "nested build directory matching an embedded CI workspace"
            shutil.copytree(FIXTURE, source)
            for template in source.rglob("BUILD.bazel.in"):
                template.rename(template.with_suffix(""))
            # Exercise add_subdirectory embedding with independent source and
            # binary roots, including a nested authored project and local import.
            host = source / "host"
            host.mkdir()
            (host / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.26)\n"
                "project(embedding C CXX)\n"
                'add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." owned)\n'
                'if(NOT PROJECT_NAME STREQUAL "embedding")\n'
                '  message(FATAL_ERROR "Embedded project scope leaked")\n'
                "endif()\n"
            )
            configure_project(host, build)
            owned = build / "owned"
            generated = owned / ".iree/cmake/package/CMakeLists.txt"
            manifest = owned / ".iree/cmake/inputs.cmake"
            programs = [
                Path(path)
                for path in (owned / f"paths-{CONFIGURATION}.txt")
                .read_text()
                .splitlines()
            ]

            def check(value):
                build_project(build)
                self.assertEqual(run_command(str(programs[0])), f"{value}\n")
                self.assertEqual(run_command(str(programs[1])), "23\n")
                self.assertFalse((source / "package/CMakeLists.txt").exists())

            check(7)
            paths = [generated, manifest, *programs]
            timestamps = [path.stat().st_mtime_ns for path in paths]
            check(7)
            configure_project(host, build)
            check(7)
            self.assertEqual([path.stat().st_mtime_ns for path in paths], timestamps)

            settings = source / "package/settings.bzl"
            settings.write_text(
                settings.read_text().replace("BASE_VALUE=4", "BASE_VALUE=6")
            )
            check(9)
            declaration = source / "package/BUILD.bazel"
            declaration.write_text(
                declaration.read_text().replace("ADJUSTMENT=0", "ADJUSTMENT=5")
            )
            check(14)
            header = source / "package/generated.h.in"
            header.write_text(
                header.read_text().replace("HEADER_VALUE 3", "HEADER_VALUE 4")
            )
            check(15)

            extra = source / "package/value_extra.c"
            extra.write_text("int fixture_extra(void) { return 42; }\n")
            check(15)
            commands = json.loads((build / "compile_commands.json").read_text())
            self.assertIn(
                extra.as_posix(), {Path(row["file"]).as_posix() for row in commands}
            )

            added = source / "package/added"
            added.mkdir()
            (added / "BUILD.bazel").write_text(
                'cc_binary(name="probe", srcs=["main.c"])\n'
            )
            (added / "main.c").write_text("int main(void) { return 0; }\n")
            check(15)
            added_output = owned / ".iree/cmake/package/added/CMakeLists.txt"
            self.assertTrue(added_output.is_file())
            (added / "BUILD.bazel").unlink()
            check(15)
            self.assertFalse(added_output.exists())

            # Native regeneration repairs missing generated lists; it must not
            # require a separate dev.py or source updater invocation.
            generated.unlink()
            check(15)
            self.assertTrue(generated.is_file())
            manifest.unlink()
            check(15)
            self.assertTrue(manifest.is_file())
            self.assertEqual(
                {path.relative_to(source) for path in source.rglob("CMakeLists.txt")},
                {
                    Path("CMakeLists.txt"),
                    Path("host/CMakeLists.txt"),
                    Path("package/authored/CMakeLists.txt"),
                },
            )


if __name__ == "__main__":
    unittest.main()
