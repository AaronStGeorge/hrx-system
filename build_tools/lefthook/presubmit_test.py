# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PRESUBMIT_PATH = Path(__file__).with_name("presubmit.py")
PRESUBMIT_SPEC = importlib.util.spec_from_file_location("presubmit", PRESUBMIT_PATH)
if PRESUBMIT_SPEC is None or PRESUBMIT_SPEC.loader is None:
    raise RuntimeError(f"unable to load {PRESUBMIT_PATH}")
presubmit = importlib.util.module_from_spec(PRESUBMIT_SPEC)
sys.modules[PRESUBMIT_SPEC.name] = presubmit
PRESUBMIT_SPEC.loader.exec_module(presubmit)


class PresubmitTest(unittest.TestCase):
    def test_semgrep_candidates_require_configured_prefix_and_extension(self):
        with (
            mock.patch.object(presubmit, "SEMGREP_PATH_PREFIXES", ("project/src/",)),
            mock.patch.object(presubmit, "SEMGREP_EXTENSIONS", {".c"}),
        ):
            self.assertTrue(presubmit.is_semgrep_candidate_file("project/src/file.c"))
            self.assertFalse(presubmit.is_semgrep_candidate_file("project/src/file.h"))
            self.assertFalse(presubmit.is_semgrep_candidate_file("other/src/file.c"))

    def test_semgrep_scan_command_uses_local_error_rules(self):
        with mock.patch.dict(os.environ, {"IREE_SEMGREP_JOBS": "7"}):
            command = presubmit.semgrep_scan_command(["runtime/src/iree/base/status.c"])

        self.assertEqual(command[0:2], ["semgrep", "scan"])
        self.assertIn("--metrics=off", command)
        self.assertIn("--disable-version-check", command)
        self.assertIn("--strict", command)
        self.assertIn("--error", command)
        self.assertIn("ERROR", command)
        self.assertIn(presubmit.SEMGREP_CONFIG, command)
        self.assertIn("7", command)
        self.assertEqual(command[-1], "runtime/src/iree/base/status.c")

    def test_semgrep_default_jobs_are_capped_on_large_machines(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(presubmit.os, "cpu_count", return_value=192),
        ):
            self.assertEqual(presubmit.semgrep_jobs(), 14)

    def test_clang_tidy_candidates_require_configured_prefix_and_extension(self):
        with (
            mock.patch.object(presubmit, "CLANG_TIDY_PATH_PREFIXES", ("project/src/",)),
            mock.patch.object(presubmit, "CLANG_TIDY_EXTENSIONS", {".c", ".h"}),
        ):
            self.assertTrue(
                presubmit.is_clang_tidy_candidate_file("project/src/file.c")
            )
            self.assertTrue(
                presubmit.is_clang_tidy_candidate_file("project/src/file.h")
            )
            self.assertFalse(
                presubmit.is_clang_tidy_candidate_file("project/src/file.py")
            )
            self.assertFalse(presubmit.is_clang_tidy_candidate_file("other/src/file.c"))

    def test_clang_tidy_bazel_command_uses_aspect_output_group(self):
        command = presubmit.clang_tidy_bazel_command(["//runtime/src/iree/base:all"])

        self.assertEqual(command[0:2], ["bazel", "build"])
        self.assertNotIn("--keep_going", command)
        self.assertIn(presubmit.CLANG_TIDY_REPO_ENV, command)
        self.assertIn(f"--aspects={presubmit.CLANG_TIDY_ASPECT}", command)
        self.assertIn(f"--output_groups={presubmit.CLANG_TIDY_OUTPUT_GROUP}", command)
        self.assertEqual(command[-1], "//runtime/src/iree/base:all")

    def test_clang_tidy_bazel_command_can_keep_going(self):
        command = presubmit.clang_tidy_bazel_command(
            ["//runtime/src/iree/base:all"],
            keep_going=True,
        )

        self.assertEqual(command[0:3], ["bazel", "build", "--keep_going"])

    def test_cmake_clang_tidy_candidates_are_translation_units(self):
        with mock.patch.object(
            presubmit, "CLANG_TIDY_PATH_PREFIXES", ("runtime/src/iree/",)
        ):
            self.assertEqual(
                presubmit.cmake_clang_tidy_candidate_files(
                    [
                        "runtime/src/iree/base/status.c",
                        "runtime/src/iree/base/status.h",
                        "runtime/src/iree/base/status.py",
                    ]
                ),
                ["runtime/src/iree/base/status.c"],
            )

    def test_cmake_clang_tidy_command_uses_compile_database(self):
        command = presubmit.cmake_clang_tidy_command(
            clang_tidy="clang-tidy",
            plugin=Path(".tmp/plugin/libIREEClangTidyPlugin.so"),
            compile_commands_dir=Path("build/cmake-debug"),
            files=["runtime/src/iree/base/status.c"],
        )

        self.assertEqual(command[0], "clang-tidy")
        self.assertIn("--load=.tmp/plugin/libIREEClangTidyPlugin.so", command)
        self.assertIn(f"--checks={presubmit.CLANG_TIDY_CHECKS}", command)
        self.assertIn("-p=build/cmake-debug", command)
        self.assertEqual(command[-1], "runtime/src/iree/base/status.c")

    def test_cmake_build_dir_uses_recorded_devtools_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_file = Path(temporary_dir) / "iree" / "cmake_build_dir"
            state_file.parent.mkdir()
            state_file.write_text("/tmp/iree-cmake-configured\n", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {presubmit.DEVTOOLS_TMP_ENV: temporary_dir},
                clear=True,
            ):
                self.assertEqual(
                    presubmit.cmake_build_dir_from_env(),
                    Path("/tmp/iree-cmake-configured"),
                )

    def test_bazel_package_target_for_path_finds_nearest_package(self):
        self.assertEqual(
            presubmit.bazel_package_target_for_path(
                "build_tools/bazel/test/cc_benchmark_smoke_test_fixture.c"
            ),
            "//build_tools/bazel/test:all",
        )

    def test_clang_tidy_skips_locally_when_llvm_is_missing(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("runtime/src/iree/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=False
            ),
        ):
            ok = presubmit.run_clang_tidy(
                ["runtime/src/iree/base/status.c"],
                profile="paranoid",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        self.assertIn("[skip] clang-tidy", output.getvalue())

    def test_clang_tidy_required_ci_fails_when_llvm_is_missing(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.dict(os.environ, {"IREE_CLANG_TIDY_REQUIRED": "1"}),
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("runtime/src/iree/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=False
            ),
        ):
            ok = presubmit.run_clang_tidy(
                ["runtime/src/iree/base/status.c"],
                profile="ci",
                lane="bazel",
                verbose=False,
            )

        self.assertFalse(ok)
        self.assertIn("[fail] clang-tidy", output.getvalue())

    def test_clang_tidy_runs_bazel_package_for_candidate_file(self):
        with (
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("build_tools/bazel/test/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=True
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            ok = presubmit.run_clang_tidy(
                ["build_tools/bazel/test/cc_benchmark_smoke_test_fixture.c"],
                profile="paranoid",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        command = run_command.call_args.args[0]
        self.assertNotIn("--keep_going", command)
        self.assertIn("//build_tools/bazel/test:all", command)

    def test_clang_tidy_ci_runs_bazel_packages_keep_going(self):
        with (
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("build_tools/bazel/test/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=True
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            ok = presubmit.run_clang_tidy(
                ["build_tools/bazel/test/cc_benchmark_smoke_test_fixture.c"],
                profile="ci",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        command = run_command.call_args.args[0]
        self.assertIn("--keep_going", command)
        self.assertIn("//build_tools/bazel/test:all", command)

    def test_clang_tidy_infra_runs_plugin_tests(self):
        with (
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=True
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            ok = presubmit.run_clang_tidy(
                ["build_tools/clang_tidy/iree/IreeTidyModule.cc"],
                profile="paranoid",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        plugin_test_command = run_command.call_args_list[0].args[0]
        self.assertIn("//build_tools/clang_tidy:plugin_smoke_test", plugin_test_command)
        self.assertIn(
            "//build_tools/clang_tidy:status_checks_test", plugin_test_command
        )

    def test_default_profile_has_no_static_analysis_provider(self):
        ok = presubmit.run_static_analysis(
            ["runtime/src/iree/base/status.c"],
            profile="default",
            lane="bazel",
            verbose=False,
        )

        self.assertTrue(ok)

    def test_missing_static_tool_is_only_fatal_in_ci(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(presubmit.shutil, "which", return_value=None),
        ):
            self.assertTrue(
                presubmit.require_static_tool(
                    "missing-tool", "Missing tool", "paranoid"
                )
            )
            self.assertFalse(
                presubmit.require_static_tool("missing-tool", "Missing tool", "ci")
            )

        self.assertIn("[skip]", output.getvalue())
        self.assertIn("[fail]", output.getvalue())

    def test_requirements_files_trigger_root_devtools_tests(self):
        self.assertTrue(presubmit.is_root_devtools_trigger("requirements-analysis.in"))
        self.assertTrue(
            presubmit.is_root_devtools_trigger("requirements-analysis.lock.txt")
        )
        self.assertFalse(presubmit.is_root_devtools_trigger("runtime/requirements.txt"))

    def test_existing_project_scripts_include_loom(self):
        self.assertIn(
            "loom",
            {project.name for project in presubmit.existing_project_scripts()},
        )


if __name__ == "__main__":
    unittest.main()
