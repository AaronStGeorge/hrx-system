#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for deps.py."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deps


def _write_module_files(root: Path, fragment: str) -> None:
    (root / "MODULE.bazel").write_text(
        textwrap.dedent(
            """\
            module(name = "sample")

            include("//build_tools/third_party:deps.MODULE.bazel")
            """
        ),
        encoding="utf-8",
    )
    fragment_path = root / "build_tools" / "third_party" / "deps.MODULE.bazel"
    fragment_path.parent.mkdir(parents=True)
    fragment_path.write_text(textwrap.dedent(fragment), encoding="utf-8")


def _write_lock(root: Path, locked_deps: list[deps.Dependency]) -> None:
    (root / "MODULE.cmake.lock").write_text(
        deps.render_cmake_lock(locked_deps),
        encoding="utf-8",
    )


class DepsTest(unittest.TestCase):
    def _run_deps(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(deps.__file__).resolve()),
                *args,
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_check_mode_uses_locked_bcr_metadata_without_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_module_files(
                root,
                """\
                bazel_dep(
                    name = "demo",
                    version = "1.2.3",
                    dev_dependency = True,
                    repo_name = "com_demo",
                )
                """,
            )
            _write_lock(
                root,
                [
                    deps.Dependency(
                        name="demo",
                        kind="bazel_dep",
                        owner="shared",
                        module_name="demo",
                        repo_name="com_demo",
                        version="1.2.3",
                        dev_dependency=True,
                        source_url="https://bcr.bazel.build/modules/demo/1.2.3/source.json",
                        urls=("https://example.com/demo-1.2.3.tar.gz",),
                        sha256="0" * 64,
                        strip_prefix="demo-1.2.3",
                    ),
                ],
            )

            result = self._run_deps(
                root,
                "--check",
                "--registry_url",
                "http://127.0.0.1:1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)

    def test_check_mode_fails_when_bcr_version_drifts_from_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_module_files(
                root,
                """\
                bazel_dep(name = "demo", version = "1.2.4")
                """,
            )
            _write_lock(
                root,
                [
                    deps.Dependency(
                        name="demo",
                        kind="bazel_dep",
                        owner="shared",
                        module_name="demo",
                        repo_name="demo",
                        version="1.2.3",
                        dev_dependency=False,
                        source_url="https://bcr.bazel.build/modules/demo/1.2.3/source.json",
                        urls=("https://example.com/demo-1.2.3.tar.gz",),
                        sha256="0" * 64,
                        strip_prefix="demo-1.2.3",
                    ),
                ],
            )

            result = self._run_deps(
                root,
                "--check",
                "--registry_url",
                "http://127.0.0.1:1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("VERSION", result.stderr)

    def test_check_mode_fails_when_http_archive_lock_is_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_module_files(
                root,
                """\
                http_archive = use_repo_rule(
                    "@bazel_tools//tools/build_defs/repo:http.bzl",
                    "http_archive",
                )

                http_archive(
                    name = "archive_dep",
                    sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
                    strip_prefix = "archive_dep-old",
                    url = "https://example.com/archive_dep-old.tar.gz",
                )
                """,
            )
            _write_lock(
                root,
                [
                    deps.Dependency(
                        name="archive_dep",
                        kind="http_archive",
                        owner="shared",
                        module_name="archive_dep",
                        repo_name="archive_dep",
                        version="",
                        dev_dependency=False,
                        urls=("https://example.com/archive_dep.tar.gz",),
                        sha256="1" * 64,
                        strip_prefix="archive_dep",
                    ),
                ],
            )

            result = self._run_deps(root, "--check")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MODULE.cmake.lock is stale", result.stderr)


if __name__ == "__main__":
    unittest.main()
