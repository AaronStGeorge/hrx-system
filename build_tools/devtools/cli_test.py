# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from build_tools.devtools import cli
from build_tools.devtools.command_plan import WriteFileStep


class CliTest(unittest.TestCase):
    def parse_help(self, arguments: list[str]) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            cli.parse_arguments(arguments)

        self.assertEqual(cm.exception.code, 0)
        return output.getvalue()

    def parse_agent_md(self, arguments: list[str]) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            cli.parse_arguments(arguments)

        self.assertEqual(cm.exception.code, 0)
        return output.getvalue()

    def test_cmake_configure_forwards_options_without_separator(self):
        args = cli.parse_arguments(["cmake", "configure", "--fresh"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--fresh", description)
        self.assertNotIn("-- --fresh", description)

    def test_bazel_configure_accepts_portable_project_options(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "configure",
                "-DIREE_HAL_DRIVER_AMDGPU=ON",
                "-DIREE_ROCM_PATH=/opt/rocm",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("-DIREE_HAL_DRIVER_AMDGPU=ON", description)
        self.assertIn("-DIREE_ROCM_PATH=/opt/rocm", description)

    def test_bazel_configure_accepts_native_driver_options(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "configure",
                "--//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null",
                "--repo_env=IREE_ROCM_PATH=/opt/rocm",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--//runtime/config/hal:drivers=amdgpu", description)
        self.assertIn("--repo_env=IREE_ROCM_PATH=/opt/rocm", description)

    def test_bazel_build_defaults_to_project_targets(self):
        args = cli.parse_arguments(["bazel", "build"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("//runtime/...", description)
        self.assertIn("//libhrx/...", description)

    def test_root_verbose_survives_nested_command_parser(self):
        args = cli.parse_arguments(["--verbose", "bazel", "build"])

        self.assertTrue(args.verbose)

    def test_root_dry_run_survives_nested_command_parser(self):
        args = cli.parse_arguments(["--dry-run", "bazel", "build"])

        self.assertTrue(args.dry_run)

    def test_tool_environment_options_can_precede_passthrough_command(self):
        args = cli.parse_arguments(["--system", "bazel", "build"])

        self.assertTrue(args.system)

        args = cli.parse_arguments(["bazel", "--system", "build"])

        self.assertTrue(args.system)

    def test_forwarded_verbose_is_not_wrapper_verbose(self):
        args = cli.parse_arguments(["bazel", "build", "--verbose"])

        self.assertFalse(args.verbose)
        plan = args.handler(args)
        self.assertIn("bazel build --verbose", plan.describe())

    def test_forwarded_dry_run_is_not_wrapper_dry_run(self):
        args = cli.parse_arguments(["bazel", "build", "--dry-run"])

        self.assertFalse(args.dry_run)
        plan = args.handler(args)
        self.assertIn("bazel build --dry-run", plan.describe())

    def test_hyphenated_flags_accept_underscore_aliases(self):
        args = cli.parse_arguments(["--dry_run", "bazel", "build"])

        self.assertTrue(args.dry_run)

        args = cli.parse_arguments(
            ["bazel", "setup", "--tool_root", ".tools", "--alias_dir", "aliases"]
        )

        self.assertEqual(args.tool_root, Path(".tools"))
        self.assertEqual(args.alias_dir, Path("aliases"))

    def test_root_agents_md_includes_bazel_and_cmake_sections(self):
        output = self.parse_agent_md(["--agents_md"])

        self.assertIn("### Bazel", output)
        self.assertIn("### CMake", output)

    def test_bazel_agents_md_includes_only_bazel_section(self):
        output = self.parse_agent_md(["bazel", "--agents_md"])

        self.assertIn("### Bazel", output)
        self.assertNotIn("### CMake", output)

    def test_cmake_agents_md_includes_only_cmake_section(self):
        output = self.parse_agent_md(["cmake", "--agent_md"])

        self.assertIn("### CMake", output)
        self.assertNotIn("### Bazel", output)

    def test_cmake_build_target_shorthand(self):
        args = cli.parse_arguments(["cmake", "build", "hrx"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--target hrx", description)

    def test_cmake_build_forwards_raw_backend_args_without_separator(self):
        args = cli.parse_arguments(["cmake", "build", "--parallel", "8"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--parallel 8", description)
        self.assertNotIn("--target --parallel", description)

    def test_cmake_build_accepts_targets_before_raw_backend_args(self):
        args = cli.parse_arguments(["cmake", "build", "hrx", "--parallel", "8"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--target hrx", description)
        self.assertIn("--parallel 8", description)

    def test_cmake_test_forwards_options_without_separator(self):
        args = cli.parse_arguments(["cmake", "test", "-R", "hrx"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("-R hrx", description)
        self.assertNotIn("-- -R", description)

    def test_bazel_precommit_defaults_to_changed_paranoid_profile(self):
        args = cli.parse_arguments(["bazel", "precommit"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--changed", description)
        self.assertIn("--lane bazel", description)
        self.assertIn("--profile paranoid", description)

    def test_precommit_profile_can_be_selected(self):
        args = cli.parse_arguments(["bazel", "precommit", "--profile", "default"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--lane bazel", description)
        self.assertIn("--profile default", description)
        self.assertNotIn("--profile paranoid", description)

        args = cli.parse_arguments(["cmake", "precommit", "--profile", "paranoid"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--lane cmake", description)
        self.assertIn("--profile paranoid", description)
        self.assertNotIn("--hygiene", description)

    def test_bazel_precommit_can_use_base_ref(self):
        args = cli.parse_arguments(["bazel", "precommit", "--base", "origin/main"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--base origin/main", description)
        self.assertNotIn("--changed", description)

    def test_bazel_precommit_can_use_staged_files(self):
        args = cli.parse_arguments(["bazel", "precommit", "--staged"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--staged", description)
        self.assertNotIn("--changed", description)

    def test_bazel_precommit_can_use_explicit_paths(self):
        args = cli.parse_arguments(["bazel", "precommit", "README.md", "dev.py"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("README.md dev.py", description)
        self.assertNotIn("--changed", description)
        self.assertNotIn("--staged", description)

    def test_precommit_rejects_explicit_paths_with_input_mode(self):
        with self.assertRaises(SystemExit):
            cli.parse_arguments(
                ["bazel", "precommit", "--base", "origin/main", "README.md"]
            )

    def test_cmake_precommit_uses_selected_lane(self):
        args = cli.parse_arguments(["cmake", "precommit"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--changed", description)
        self.assertIn("--lane cmake", description)
        self.assertNotIn("--hygiene", description)
        self.assertIn("--profile default", description)

    def test_cmake_presubmit_profile_can_be_selected(self):
        args = cli.parse_arguments(["cmake", "presubmit", "--profile", "paranoid"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--lane cmake", description)
        self.assertIn("--profile paranoid", description)
        self.assertIn("--all", description)
        self.assertNotIn("--hygiene", description)

    def test_bazel_presubmit_uses_full_tree_input(self):
        args = cli.parse_arguments(["bazel", "presubmit", "--profile", "default"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--lane bazel", description)
        self.assertIn("--profile default", description)
        self.assertIn("--all", description)

    def test_bazel_presubmit_can_skip_project_tests(self):
        args = cli.parse_arguments(["bazel", "presubmit", "--no_project_tests"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--no-project-tests", description)

    def test_hook_writes_selected_profile(self):
        args = cli.parse_arguments(["bazel", "hook", "--profile", "ci"])

        plan = args.handler(args)
        step = plan.steps[0]

        self.assertIsInstance(step, WriteFileStep)
        self.assertIn("python dev.py bazel hook --profile ci", step.content)
        self.assertIn(
            "run: python dev.py bazel precommit --profile ci {staged_files}",
            step.content,
        )

    def test_cmake_hook_defaults_to_default_profile(self):
        args = cli.parse_arguments(["cmake", "hook"])

        plan = args.handler(args)
        step = plan.steps[0]

        self.assertIsInstance(step, WriteFileStep)
        self.assertIn(
            "run: python dev.py cmake precommit --profile default {staged_files}",
            step.content,
        )

    def test_cmake_configure_help_explains_user_visible_build_tree(self):
        output = self.parse_help(["cmake", "configure", "--help"])

        self.assertIn("Configure the CMake package/install-test build tree", output)
        self.assertIn("../builds/<checkout-name>/", output)
        self.assertNotIn("backend configure tool", output)

    def test_bazel_build_help_explains_default_targets(self):
        output = self.parse_help(["bazel", "build", "--help"])

        self.assertIn("Build Bazel targets", output)
        self.assertIn("//runtime/...", output)
        self.assertIn("//libhrx/...", output)
        self.assertNotIn("backend", output.lower())

    def test_precommit_help_explains_changed_file_set(self):
        output = self.parse_help(["bazel", "precommit", "--help"])

        self.assertIn("staged, unstaged, and untracked files", output)
        self.assertIn("--base", output)
        self.assertIn("--staged", output)
        self.assertIn("Explicit paths", output)

    def test_presubmit_help_calls_out_full_tree_default(self):
        output = self.parse_help(["bazel", "presubmit", "--help"])

        self.assertIn("The default profile is ci", output)
        self.assertIn("full-tree", output)
        self.assertIn("precommit", output)


if __name__ == "__main__":
    unittest.main()
