#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Root presubmit dispatcher.

The root owns repository-wide hygiene and maps changed files to project
presubmit entry points. Project-specific test policy stays in each project.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

BUILDIFIER_EXTENSIONS = {".bzl", ".BUILD"}
BUILDIFIER_NAMES = {
    "BUILD",
    "BUILD.bazel",
    "MODULE.bazel",
    "WORKSPACE",
    "WORKSPACE.bazel",
    "WORKSPACE.bzlmod",
}
C_FORMAT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".mm",
}
WATCHWORD_PATTERNS = (
    re.compile("DO " + "NOT SUBMIT"),
    re.compile("DO_" + "NOT_SUBMIT"),
    re.compile("DO" + "NOTSUBMIT"),
    re.compile("TODO before " + "submit", re.IGNORECASE),
)
CONFLICT_MARKER_PATTERN = re.compile(r"^(<<<<<<< .+|>>>>>>> .+)")


@dataclass(frozen=True)
class Project:
    name: str
    root: str
    script: str


PROJECTS = (
    Project(
        name="runtime",
        root="runtime/",
        script="runtime/build_tools/presubmit.py",
    ),
    Project(
        name="libhrx",
        root="libhrx/",
        script="libhrx/build_tools/presubmit.py",
    ),
    Project(
        name="loom",
        root="loom/",
        script="loom/build_tools/presubmit.py",
    ),
)

GLOBAL_PROJECT_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
    "build_tools/bazel/",
    "build_tools/bazel_to_cmake/",
    "build_tools/testing/",
    "build_tools/third_party/",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository presubmit checks.")
    parser.add_argument(
        "--profile",
        choices=("default", "paranoid", "ci"),
        default=os.environ.get("IREE_PRESUBMIT_PROFILE", "default"),
        help="Presubmit profile. Defaults to IREE_PRESUBMIT_PROFILE or default.",
    )

    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--fix", action="store_true", help="Apply local fixups.")
    mutation.add_argument(
        "--check",
        action="store_true",
        help="Check without writing files. This is the CI-safe mode.",
    )

    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--staged", action="store_true", help="Use files staged for commit."
    )
    inputs.add_argument("--all", action="store_true", help="Use all tracked files.")
    inputs.add_argument(
        "--since",
        metavar="GIT_REF",
        help="Use files changed since the given Git ref.",
    )
    inputs.add_argument(
        "--files-from",
        metavar="PATH",
        help="Use newline-separated repo-relative files from PATH.",
    )

    parser.add_argument(
        "--hygiene",
        action="store_true",
        help="Run formatting, generated-file, and cheap invariant checks.",
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Run affected project Bazel presubmit tests.",
    )
    parser.add_argument(
        "--static-analysis",
        action="store_true",
        help="Run configured static-analysis providers.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the selected plan before running it.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Explicit repo-relative paths. Used by lefthook file templates.",
    )

    args = parser.parse_args()
    if args.paths and (args.staged or args.all or args.since or args.files_from):
        parser.error("explicit paths cannot be combined with another input mode")
    apply_profile_defaults(args)
    return args


def apply_profile_defaults(args: argparse.Namespace) -> None:
    if not args.fix and not args.check:
        args.check = args.profile == "ci"
        args.fix = not args.check

    if (
        not args.paths
        and not args.staged
        and not args.all
        and not args.since
        and not args.files_from
    ):
        args.all = args.profile == "ci"
        args.staged = not args.all

    if not args.hygiene and not args.tests and not args.static_analysis:
        args.hygiene = True
        if args.profile in ("paranoid", "ci"):
            args.tests = True
            args.static_analysis = True


def run_command(command: list[str], description: str) -> bool:
    print(f"presubmit: {description}")
    print("  " + " ".join(command))
    sys.stdout.flush()
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode == 0:
        return True
    print(f"presubmit: {description} failed with exit code {result.returncode}")
    return False


def require_tool(tool: str) -> bool:
    if shutil.which(tool):
        return True
    print(f"presubmit: required tool '{tool}' was not found on PATH.", file=sys.stderr)
    return False


def git_list(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git"] + args,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path
    ]


def selected_files(args: argparse.Namespace) -> list[str]:
    if args.paths:
        return args.paths
    if args.staged:
        return git_list(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"])
    if args.all:
        return git_list(["ls-files", "-z"])
    if args.files_from:
        with open(args.files_from, "r", encoding="utf-8") as file_list:
            return [line.strip() for line in file_list if line.strip()]
    return git_list(
        ["diff", "--name-only", "-z", "--diff-filter=ACMR", args.since, "--"]
    )


def existing_files(paths: list[str]) -> list[str]:
    return [path for path in paths if (REPO_ROOT / path).is_file()]


def stage_files(paths: list[str]) -> bool:
    if not paths:
        return True
    return run_command(["git", "add", "--"] + paths, "stage local fixups")


def is_buildifier_file(path: str) -> bool:
    file_path = Path(path)
    return file_path.name in BUILDIFIER_NAMES or file_path.suffix in BUILDIFIER_EXTENSIONS


def is_c_format_file(path: str) -> bool:
    return Path(path).suffix in C_FORMAT_EXTENSIONS


def run_buildifier(paths: list[str], fix: bool) -> bool:
    files = existing_files([path for path in paths if is_buildifier_file(path)])
    if not files:
        return True
    if not require_tool("buildifier"):
        return False
    command = ["buildifier", "-lint=off"]
    if not fix:
        command.append("-mode=check")
    command += files
    ok = run_command(command, "buildifier")
    if fix and ok:
        ok = stage_files(files)
    return ok


def run_clang_format(paths: list[str], fix: bool) -> bool:
    files = existing_files([path for path in paths if is_c_format_file(path)])
    if not files:
        return True
    if not require_tool("clang-format"):
        return False
    command = ["clang-format"]
    if fix:
        command.append("-i")
    else:
        command += ["--dry-run", "--Werror"]
    command += files
    ok = run_command(command, "clang-format")
    if fix and ok:
        ok = stage_files(files)
    return ok


def text_for_path(path: str) -> str | None:
    try:
        data = (REPO_ROOT / path).read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def fix_text_hygiene(path: str, text: str) -> bool:
    lines = text.splitlines(keepends=True)
    fixed_lines = []
    for line in lines:
        if line.endswith("\r\n"):
            fixed_lines.append(line[:-2].rstrip(" \t") + "\n")
        elif line.endswith("\n"):
            fixed_lines.append(line[:-1].rstrip(" \t") + "\n")
        else:
            fixed_lines.append(line.rstrip(" \t"))
    fixed_text = "".join(fixed_lines)
    if fixed_text and not fixed_text.endswith("\n"):
        fixed_text += "\n"
    if fixed_text == text:
        return False
    (REPO_ROOT / path).write_text(fixed_text, encoding="utf-8")
    return True


def run_text_hygiene(paths: list[str], fix: bool) -> bool:
    ok = True
    changed = []
    for path in existing_files(paths):
        text = text_for_path(path)
        if text is None:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            if CONFLICT_MARKER_PATTERN.search(line):
                print(f"{path}:{index}: merge conflict marker")
                ok = False
        if fix:
            if fix_text_hygiene(path, text):
                changed.append(path)
        else:
            lines = text.splitlines(keepends=True)
            for index, line in enumerate(lines, start=1):
                content = line[:-1] if line.endswith("\n") else line
                if content.endswith((" ", "\t")):
                    print(f"{path}:{index}: trailing whitespace")
                    ok = False
            if text and not text.endswith("\n"):
                print(f"{path}: missing final newline")
                ok = False
    if fix and changed:
        ok = stage_files(changed) and ok
    return ok


def run_watchwords(paths: list[str]) -> bool:
    ok = True
    for path in existing_files(paths):
        text = text_for_path(path)
        if text is None:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            for pattern in WATCHWORD_PATTERNS:
                match = pattern.search(line)
                if match:
                    print(f'{path}:{index}: forbidden watchword "{match.group(0)}"')
                    ok = False
    return ok


def run_build_filename_check(paths: list[str]) -> bool:
    ok = True
    for path in paths:
        if Path(path).name == "BUILD":
            print(f"{path}: use BUILD.bazel instead of BUILD")
            ok = False
    return ok


def run_bazel_to_cmake(fix: bool) -> bool:
    command = ["python", "build_tools/bazel_to_cmake/bazel_to_cmake.py"]
    command.append("--stage-updates" if fix else "--check")
    return run_command(command, "bazel-to-cmake")


def run_amdgpu_target_map(paths: list[str], fix: bool) -> bool:
    relevant_prefixes = (
        "build_tools/scripts/amdgpu_target_map.py",
        "runtime/src/iree/hal/drivers/amdgpu/device/binaries/target_map.",
        "runtime/src/iree/hal/drivers/amdgpu/util/target_id_map.inl",
    )
    if not any(path.startswith(relevant_prefixes) for path in paths):
        return True
    command = ["python", "build_tools/scripts/amdgpu_target_map.py"]
    if not fix:
        command.append("--check")
    ok = run_command(command, "AMDGPU target map")
    if fix and ok:
        ok = stage_files(
            [
                "runtime/src/iree/hal/drivers/amdgpu/device/binaries/target_map.bzl",
                "runtime/src/iree/hal/drivers/amdgpu/device/binaries/target_map.cmake",
                "runtime/src/iree/hal/drivers/amdgpu/util/target_id_map.inl",
            ]
        )
    return ok


def run_hygiene(paths: list[str], fix: bool) -> bool:
    ok = True
    ok = run_build_filename_check(paths) and ok
    ok = run_text_hygiene(paths, fix=fix) and ok
    ok = run_watchwords(paths) and ok
    ok = run_buildifier(paths, fix=fix) and ok
    ok = run_clang_format(paths, fix=fix) and ok
    ok = run_bazel_to_cmake(fix=fix) and ok
    ok = run_amdgpu_target_map(paths, fix=fix) and ok
    return ok


def existing_project_scripts() -> list[Project]:
    return [project for project in PROJECTS if (REPO_ROOT / project.script).is_file()]


def projects_for_paths(paths: list[str]) -> list[Project]:
    projects = existing_project_scripts()
    if any(is_global_trigger(path) for path in paths):
        return projects
    selected = []
    for project in projects:
        if any(path.startswith(project.root) for path in paths):
            selected.append(project)
    return selected


def is_global_trigger(path: str) -> bool:
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(path == trigger or path.startswith(trigger) for trigger in GLOBAL_PROJECT_TRIGGERS)


def run_project_tests(projects: list[Project], paths: list[str], fix: bool) -> bool:
    if not projects:
        print("presubmit: no affected project test entry points")
        return True
    ok = True
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, dir=REPO_ROOT / ".git"
    ) as file_list:
        for path in paths:
            file_list.write(path + "\n")
        file_list_path = file_list.name
    try:
        for project in projects:
            command = [
                "python",
                project.script,
                "--files-from",
                file_list_path,
                "--tests",
            ]
            command.append("--fix" if fix else "--check")
            ok = run_command(command, f"{project.name} presubmit") and ok
    finally:
        Path(file_list_path).unlink(missing_ok=True)
    return ok


def run_static_analysis(paths: list[str]) -> bool:
    if not paths:
        return True
    print("presubmit: static-analysis lane has no configured providers yet")
    return True


def print_plan(
    args: argparse.Namespace, paths: list[str], projects: list[Project]
) -> None:
    mutation = "fix" if args.fix else "check"
    if args.paths:
        input_mode = "explicit paths"
    elif args.all:
        input_mode = "all"
    elif args.since:
        input_mode = f"since {args.since}"
    elif args.files_from:
        input_mode = f"files from {args.files_from}"
    else:
        input_mode = "staged"
    scopes = []
    if args.hygiene:
        scopes.append("hygiene")
    if args.tests:
        scopes.append("tests")
    if args.static_analysis:
        scopes.append("static-analysis")
    print("presubmit plan:")
    print(f"  profile: {args.profile}")
    print(f"  mode: {mutation}")
    print(f"  input: {input_mode} ({len(paths)} file(s))")
    print("  scopes: " + ", ".join(scopes))
    if projects:
        print("  projects: " + ", ".join(project.name for project in projects))
    else:
        print("  projects: none")
    sys.stdout.flush()


def main() -> int:
    args = parse_arguments()
    paths = selected_files(args)
    projects = projects_for_paths(paths)
    if args.print_plan:
        print_plan(args, paths, projects)

    ok = True
    if args.hygiene:
        ok = run_hygiene(paths, fix=args.fix) and ok
    if args.tests:
        ok = run_project_tests(projects, paths, fix=args.fix) and ok
    if args.static_analysis:
        ok = run_static_analysis(paths) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
