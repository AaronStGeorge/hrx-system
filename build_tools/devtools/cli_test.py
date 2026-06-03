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

    def test_backend_separator_is_not_forwarded(self):
        args = cli.parse_arguments(["cmake", "configure", "--", "--fresh"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--fresh", description)
        self.assertNotIn("-- --fresh", description)

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

    def test_forwarded_verbose_is_not_wrapper_verbose(self):
        args = cli.parse_arguments(["bazel", "build", "--", "--verbose"])

        self.assertFalse(args.verbose)
        self.assertEqual(args.args, ["--", "--verbose"])

    def test_forwarded_dry_run_is_not_wrapper_dry_run(self):
        args = cli.parse_arguments(["bazel", "build", "--", "--dry-run"])

        self.assertFalse(args.dry_run)
        self.assertEqual(args.args, ["--", "--dry-run"])

    def test_hyphenated_flags_accept_underscore_aliases(self):
        args = cli.parse_arguments(["bazel", "build", "--dry_run"])

        self.assertTrue(args.dry_run)

        args = cli.parse_arguments(
            ["bazel", "setup", "--tool_root", ".tools", "--alias_dir", "aliases"]
        )

        self.assertEqual(args.tool_root, Path(".tools"))
        self.assertEqual(args.alias_dir, Path("aliases"))

    def test_root_agents_md_includes_bazel_and_cmake_lanes(self):
        output = self.parse_agent_md(["--agents_md"])

        self.assertIn("### Bazel Lane", output)
        self.assertIn("### CMake Lane", output)
        self.assertIn("python dev.py bazel precommit", output)
        self.assertIn("python dev.py cmake build hrx", output)

    def test_bazel_agents_md_includes_only_bazel_lane(self):
        output = self.parse_agent_md(["bazel", "--agents_md"])

        self.assertIn("### Bazel Lane", output)
        self.assertNotIn("### CMake Lane", output)
        self.assertIn("python dev.py bazel test", output)

    def test_cmake_agents_md_includes_only_cmake_lane(self):
        output = self.parse_agent_md(["cmake", "--agent_md"])

        self.assertIn("### CMake Lane", output)
        self.assertNotIn("### Bazel Lane", output)
        self.assertIn("../builds/<checkout-name>/", output)

    def test_command_agents_md_uses_command_lane(self):
        output = self.parse_agent_md(["bazel", "build", "--agents-md"])

        self.assertIn("### Bazel Lane", output)
        self.assertNotIn("### CMake Lane", output)

    def test_cmake_build_target_shorthand(self):
        args = cli.parse_arguments(["cmake", "build", "hrx"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--target hrx", description)

    def test_cmake_build_separator_for_raw_backend_args(self):
        args = cli.parse_arguments(["cmake", "build", "--", "--parallel", "8"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--parallel 8", description)
        self.assertNotIn("--target --parallel", description)

    def test_bazel_precommit_defaults_to_changed_paranoid_profile(self):
        args = cli.parse_arguments(["bazel", "precommit"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--changed", description)
        self.assertIn("--profile paranoid", description)

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

    def test_cmake_precommit_is_hygiene_only(self):
        args = cli.parse_arguments(["cmake", "precommit"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--changed", description)
        self.assertIn("--hygiene", description)
        self.assertIn("--profile default", description)

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
