# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Presubmit and fix command planning."""

from __future__ import annotations

from build_tools.devtools.command_plan import CommandPlan, CommandStep
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment

PROFILES = ("default", "paranoid", "ci")
PRESUBMIT_DEFAULT_PROFILE = "ci"
PRECOMMIT_DEFAULT_PROFILES = {
    "bazel": "paranoid",
    "cmake": "default",
}


def precommit_default_profile(lane: str) -> str:
    try:
        return PRECOMMIT_DEFAULT_PROFILES[lane]
    except KeyError:
        raise ValueError(f"unknown lane: {lane}") from None


def presubmit_plan(
    lane: str,
    tool_env: ToolEnvironment,
    profile: str,
    verbose: bool = False,
) -> CommandPlan:
    env = tool_env.path_env()
    plan = CommandPlan()
    if lane == "bazel":
        command = [
            tool_env.python,
            str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
            "--profile",
            profile,
            "--check",
            "--all",
        ]
        if verbose:
            command.append("--verbose")
        plan.add(
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label="run Bazel-lane presubmit",
            )
        )
    elif lane == "cmake":
        command = [
            tool_env.python,
            str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
            "--profile",
            profile,
            "--check",
            "--all",
            "--hygiene",
        ]
        if verbose:
            command.append("--verbose")
        plan.add(
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label="run CMake-lane shared hygiene",
            )
        )
    else:
        raise ValueError(f"unknown lane: {lane}")
    return plan


def precommit_plan(
    lane: str,
    tool_env: ToolEnvironment,
    profile: str,
    base: str | None = None,
    staged: bool = False,
    paths: list[str] | None = None,
    verbose: bool = False,
) -> CommandPlan:
    env = tool_env.path_env()
    command = [
        tool_env.python,
        str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
        "--check",
    ]
    if paths:
        pass
    elif base is not None:
        command += ["--base", base]
    elif staged:
        command.append("--staged")
    else:
        command.append("--changed")

    if lane == "bazel":
        command += ["--profile", profile]
    elif lane == "cmake":
        command += ["--profile", profile, "--hygiene"]
    else:
        raise ValueError(f"unknown lane: {lane}")

    if verbose:
        command.append("--verbose")
    if paths:
        command += paths
    return CommandPlan(
        [
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label=f"run {lane}-lane precommit",
            )
        ]
    )


def fix_plan(tool_env: ToolEnvironment, verbose: bool = False) -> CommandPlan:
    env = tool_env.path_env()
    command = [
        tool_env.python,
        str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
        "--fix",
        "--staged",
        "--hygiene",
    ]
    if verbose:
        command.append("--verbose")
    return CommandPlan(
        [
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label="apply staged mechanical fixups",
            )
        ]
    )
