# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import unittest

from build_tools.devtools import ci


class CiTest(unittest.TestCase):
    def test_cpu_dry_run_exposes_copyable_commands(self):
        args = ci.parse_arguments(
            [
                "iree-cpu",
                "--dry-run",
                "--target",
                "//runtime/...",
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                ci.run_steps(
                    ci.steps_from_args(args),
                    dry_run=True,
                    keep_going=False,
                    verbose=False,
                ),
                0,
            )

        text = output.getvalue()
        self.assertIn("dev.py bazel configure", text)
        self.assertIn("dev.py bazel build //runtime/...", text)
        self.assertIn("dev.py bazel test //runtime/...", text)

    def test_amdgpu_dry_run_does_not_embed_machine_paths(self):
        args = ci.parse_arguments(
            [
                "iree-amdgpu",
                "--dry-run",
                "--target",
                "//runtime/...",
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                ci.run_steps(
                    ci.steps_from_args(args),
                    dry_run=True,
                    keep_going=False,
                    verbose=False,
                ),
                0,
            )

        text = output.getvalue()
        self.assertIn("dev.py bazel configure", text)
        self.assertNotIn("IREE_ROCM_PATH", text)
        self.assertNotIn("/opt/rocm", text)

    def test_sanitizer_command_runs_tests_and_msan_build(self):
        args = ci.parse_arguments(
            [
                "iree-cpu-sanitizers",
                "--target",
                "//runtime/...",
            ]
        )

        command_lines = [step.command_line() for step in ci.steps_from_args(args)]

        self.assertTrue(
            any(
                "bazel test //runtime/... --config=asan" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel test //runtime/... --config=ubsan" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel test //runtime/... --config=tsan" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel build //runtime/... --config=msan" in line
                for line in command_lines
            )
        )


if __name__ == "__main__":
    unittest.main()
