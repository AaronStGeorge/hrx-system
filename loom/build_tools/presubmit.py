#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom project presubmit entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "loom/py"))

from loom.gen import checked_in_artifacts
from loom.tools.source_inventory import is_format_source_path, is_lint_source_path

from build_tools.devtools import bazel_test_selection, project_presubmit
from build_tools.devtools.command_line import batch_path_commands
from build_tools.devtools.source_lock import NonEmptyTrackedFileSnapshot
from loom.build_tools import target_presets

PROJECT_NAME = "loom"
PROJECT_ROOT = "loom/"
CMAKE_TEST_REGEX = "^loom/"
GLOBAL_TEST_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
)
CTEST_RESOURCE_LABEL_EXCLUDE_REGEX = "runtime-resource="
CMAKE_SOURCE_FORMAT_CAPABILITY_SETS = (
    (
        "architecture",
        "LOOM_RESOLVED_ARCHITECTURES",
        target_presets.ARCHITECTURES,
    ),
    (
        "artifact",
        "LOOM_RESOLVED_ARTIFACT_FORMATS",
        target_presets.ARTIFACT_FORMATS,
    ),
    (
        "execution",
        "LOOM_RESOLVED_EXECUTION_SUBSTRATES",
        target_presets.EXECUTION_SUBSTRATES,
    ),
)
# Formatting, hygiene, dependency selection, and tests share one build
# configuration so successive phases retain Bazel's configured analysis graph.
BAZEL_CONFIGURATION_ARGS = (
    "--config=loom-target-all",
    "--//loom/config/import:enable=cxx",
)
BAZEL_SOURCE_TOOL_ARGS = (
    "--config=locked",
    *BAZEL_CONFIGURATION_ARGS,
)
BAZEL_TEST_CONFIGURATION_ARGS = (
    "--config=presubmit",
    *BAZEL_CONFIGURATION_ARGS,
)
BAZEL_TESTS = bazel_test_selection.TestSuite(
    name=PROJECT_NAME,
    targets=("//loom/...",),
    configuration_args=BAZEL_TEST_CONFIGURATION_ARGS,
)
LOOM_FORMAT_BAZEL_TARGET = "//loom/src/loom/tools/loom-format:loom-format"
LOOM_FORMAT_CMAKE_TARGET = "loom::tools::loom-format"
LOOM_LINT_PYTHON_SOURCE = "loom/py/loom/tools/source_lint.py"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Loom project presubmit.")
    project_presubmit.add_common_arguments(parser, project_name=PROJECT_NAME)
    return parser.parse_args()


def run_command(
    command: list[str],
    description: str,
    *,
    success_exit_codes: tuple[int, ...] = (0,),
) -> bool:
    return project_presubmit.run_command(
        PROJECT_NAME,
        command,
        description,
        cwd=REPO_ROOT,
        success_exit_codes=success_exit_codes,
    )


def run_batched_path_command(
    command_prefix: list[str], paths: list[str], description: str
) -> bool:
    try:
        commands = batch_path_commands(command_prefix, paths)
    except ValueError as exc:
        print(f"loom presubmit: {exc}", file=sys.stderr)
        return False

    ok = True
    for index, command in enumerate(commands, start=1):
        batch_description = (
            description
            if len(commands) == 1
            else f"{description} ({index}/{len(commands)})"
        )
        ok = run_command(command, batch_description) and ok
    return ok


def is_global_trigger(path: str) -> bool:
    if "build_tools" in Path(path).parts:
        return True
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(
        path == trigger or path.startswith(trigger) for trigger in GLOBAL_TEST_TRIGGERS
    )


def selected_files(files_from: str | None) -> list[str]:
    if not files_from:
        return []
    with open(files_from, encoding="utf-8") as file_list:
        return [line.strip() for line in file_list if line.strip()]


def should_run_presubmit(files_from: str | None) -> bool:
    paths = selected_files(files_from)
    if not paths:
        return files_from is None
    return any(
        path.startswith(PROJECT_ROOT) or is_global_trigger(path) for path in paths
    )


def run_generated_artifact_maintenance(
    fix: bool, files_from: str | None = None
) -> bool:
    print("loom presubmit: Checked-in generated artifacts")
    if fix and files_from is not None:
        result = checked_in_artifacts.maintain_checked_in_artifacts(
            "update", selected_paths=selected_files(files_from)
        )
    else:
        result = checked_in_artifacts.maintain_checked_in_artifacts(
            "update" if fix else "check"
        )
    if not result.ok:
        return False
    if not fix:
        return True
    return project_presubmit.stage_changed_paths(
        PROJECT_NAME,
        REPO_ROOT,
        result.changed_paths,
    )


def existing_format_source_paths(paths: list[str]) -> list[str]:
    return sorted(
        {
            path
            for path in paths
            if is_format_source_path(path) and (REPO_ROOT / path).is_file()
        }
    )


def _tracked_project_paths() -> list[str] | None:
    result = subprocess.run(
        ["git", "--literal-pathspecs", "ls-files", "-z", "--", PROJECT_ROOT],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(
            "loom presubmit: failed to query tracked Loom project files",
            file=sys.stderr,
        )
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        return None
    return result.stdout.split("\0")


def tracked_format_source_paths() -> list[str] | None:
    paths = _tracked_project_paths()
    return None if paths is None else existing_format_source_paths(paths)


def existing_lint_source_paths(paths: list[str]) -> list[str]:
    return sorted(
        {
            path
            for path in paths
            if is_lint_source_path(path) and (REPO_ROOT / path).is_file()
        }
    )


def tracked_lint_source_paths() -> list[str] | None:
    paths = _tracked_project_paths()
    return None if paths is None else existing_lint_source_paths(paths)


def validate_cmake_source_format_configuration() -> bool:
    build_dir = project_presubmit.cmake_build_dir(REPO_ROOT)
    if not project_presubmit.validate_cmake_build_tree(PROJECT_NAME, build_dir):
        return False

    missing_capabilities = []
    for (
        capability_kind,
        cache_key,
        required_values,
    ) in CMAKE_SOURCE_FORMAT_CAPABILITY_SETS:
        resolved_values = set(
            filter(
                None,
                (project_presubmit.cmake_cache_value(build_dir, cache_key) or "").split(
                    ";"
                ),
            )
        )
        missing_capabilities.extend(
            f"{capability_kind} {value}"
            for value in required_values
            if value not in resolved_values
        )
    if not missing_capabilities:
        return True

    missing_names = ", ".join(missing_capabilities)
    print(
        "loom presubmit: canonical source formatting requires the CMake "
        f"build tree to enable every Loom capability; missing: {missing_names}"
    )
    print(
        "loom presubmit: reconfigure that tree with "
        "`-DLOOM_TARGET_ALL=ON` or select a full-target tree with "
        f"{project_presubmit.CMAKE_BUILD_DIR_ENV}"
    )
    return False


def run_source_format_maintenance(
    *, lane: str, files_from: str | None, fix: bool
) -> bool:
    # Bazel owns the complete read-only sweep. Only explicit source mutation
    # requires a local formatter, and only when selected Loom text exists.
    if lane == "bazel" and not fix:
        return True
    tracked_paths = (
        []
        if lane == "bazel" and files_from is not None
        else tracked_format_source_paths()
    )
    if tracked_paths is None:
        return False
    if files_from is None:
        selected_paths = tracked_paths
    else:
        selected_paths = existing_format_source_paths(selected_files(files_from))
    check_paths = sorted(set(tracked_paths).union(selected_paths))
    if not check_paths:
        return True
    if lane == "cmake" and not validate_cmake_source_format_configuration():
        return False

    formatter_path = project_presubmit.build_and_resolve_executable(
        PROJECT_NAME,
        REPO_ROOT,
        lane=lane,
        bazel_target=LOOM_FORMAT_BAZEL_TARGET,
        cmake_target=LOOM_FORMAT_CMAKE_TARGET,
        bazel_args=BAZEL_SOURCE_TOOL_ARGS,
    )
    if formatter_path is None:
        return False

    if fix and selected_paths:
        if not run_batched_path_command(
            [str(formatter_path), "--in-place"],
            selected_paths,
            "Canonicalize selected Loom source",
        ):
            return False
        if not project_presubmit.stage_changed_paths(
            PROJECT_NAME,
            REPO_ROOT,
            selected_paths,
        ):
            return False

    if lane == "bazel":
        return True
    return run_batched_path_command(
        [str(formatter_path), "--check"],
        check_paths,
        "Canonical Loom source",
    )


def run_bazel_hygiene() -> bool:
    return run_command(
        [
            "bazel",
            "build",
            "--keep_going",
            *BAZEL_SOURCE_TOOL_ARGS,
            *project_presubmit.bazel_config_args(),
            "//loom/build_tools/hygiene:checks",
        ],
        "Loom source hygiene",
    )


def run_cmake_source_lint(*, files_from: str | None) -> bool:
    tracked_paths = tracked_lint_source_paths()
    if tracked_paths is None:
        return False
    selected_paths = (
        []
        if files_from is None
        else existing_lint_source_paths(selected_files(files_from))
    )
    check_paths = sorted(set(tracked_paths).union(selected_paths))

    public_lint_ok = True
    if check_paths:
        # The CMake source-bearing Python target has no executable artifact.
        public_lint_ok = run_batched_path_command(
            [sys.executable, LOOM_LINT_PYTHON_SOURCE],
            check_paths,
            "Loom authoring policy",
        )

    repository_lint_ok = run_command(
        [
            sys.executable,
            "loom/build_tools/linters/loom_source_lint.py",
        ],
        "Loom repository invariants",
    )
    return public_lint_ok and repository_lint_ok


def run_bazel_tests(files_from: str | None = None) -> bool:
    return BAZEL_TESTS.run(
        REPO_ROOT,
        paths=selected_files(files_from) if files_from is not None else None,
    )


def run_cmake_tests() -> bool:
    build_dir = project_presubmit.cmake_build_dir(REPO_ROOT)
    if not project_presubmit.validate_cmake_build_tree(PROJECT_NAME, build_dir):
        return False
    if not run_command(
        ["cmake", "--build", str(build_dir), "--parallel"],
        "CMake build",
    ):
        return False
    return run_command(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--output-on-failure",
            "-R",
            CMAKE_TEST_REGEX,
            "-LE",
            CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
        ],
        "CTest tests",
    )


def run_presubmit(args: argparse.Namespace) -> int:
    if not args.hygiene and not args.tests:
        return 0
    project_inputs = should_run_presubmit(args.files_from)
    ok = True
    if args.hygiene and project_inputs:
        ok = run_generated_artifact_maintenance(args.fix, args.files_from) and ok
        ok = (
            run_source_format_maintenance(
                lane=args.lane,
                files_from=args.files_from,
                fix=args.fix,
            )
            and ok
        )
        if args.lane == "bazel":
            ok = run_bazel_hygiene() and ok
        elif args.lane == "cmake":
            ok = run_cmake_source_lint(files_from=args.files_from) and ok
        else:
            raise ValueError(f"unknown lane: {args.lane}")
    if args.tests:
        if args.lane == "bazel":
            ok = run_bazel_tests(args.files_from) and ok
        elif args.lane == "cmake":
            ok = (not project_inputs or run_cmake_tests()) and ok
        else:
            raise ValueError(f"unknown lane: {args.lane}")
    return 0 if ok else 1


def main() -> int:
    args = parse_arguments()
    snapshot = NonEmptyTrackedFileSnapshot.capture_tracked_package_initializers(
        REPO_ROOT
    )
    result = run_presubmit(args)
    if not snapshot.verify(REPO_ROOT):
        result = 1
    return result


if __name__ == "__main__":
    sys.exit(main())
