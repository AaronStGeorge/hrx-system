# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import unittest

from build_tools.devtools import ci, ci_config


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

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                "bazel test --config=asan -- //runtime/..." in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel test --config=ubsan -- //runtime/..." in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel test --config=tsan -- //runtime/..." in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel build //runtime/... --config=msan" in line
                for line in command_lines
            )
        )
        sanitizer_test_steps = [
            step for step in steps if step.name.startswith("Test IREE")
        ]
        for xfail_target in ci_config.CPU_SANITIZERS_XFAIL_TARGETS:
            self.assertTrue(
                any(xfail_target in step.argv for step in sanitizer_test_steps)
            )

    def test_amdgpu_command_scopes_tests_to_amdgpu(self):
        args = ci.parse_arguments(
            [
                "iree-amdgpu",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        for target in ci_config.AMDGPU_DRIVER_TARGETS:
            self.assertTrue(any(target in step.argv for step in build_steps))
        self.assertTrue(
            any(
                "--test_tag_filters=" + ci_config.AMDGPU_RESOURCE_TAG in line
                for line in command_lines
            )
        )
        resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU resources"
        )
        self.assertIn("//runtime/...", resource_test.argv)
        for target in ci_config.AMDGPU_XFAIL_TARGETS:
            self.assertIn(target, resource_test.argv)

    def test_amdgpu_sanitizer_command_uses_amdgpu_sanitizer_xfails(self):
        args = ci.parse_arguments(
            [
                "iree-amdgpu-sanitizers",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        sanitizer_test_steps = [
            step for step in steps if step.name.startswith("Test IREE")
        ]

        for xfail_target in ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS:
            self.assertTrue(
                any(xfail_target in step.argv for step in sanitizer_test_steps)
            )


if __name__ == "__main__":
    unittest.main()
