# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Git hook command planning."""

from __future__ import annotations

from build_tools.devtools.command_plan import CommandPlan, CommandStep, CopyFileStep
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment

LANE_TEMPLATE_PATHS = {
    "bazel": REPO_ROOT / "build_tools/lefthook/lefthook-local.bazel.yml",
    "cmake": REPO_ROOT / "build_tools/lefthook/lefthook-local.cmake.yml",
}


def hook_plan(lane: str, tool_env: ToolEnvironment, verify: bool) -> CommandPlan:
    plan = CommandPlan()
    template_path = LANE_TEMPLATE_PATHS[lane]
    plan.add(
        CopyFileStep(
            source=template_path,
            destination=REPO_ROOT / "lefthook-local.yml",
            label=f"select {lane} hook policy",
        )
    )
    env = tool_env.path_env()
    plan.add(
        CommandStep(
            [tool_env.tool("lefthook"), "install"],
            cwd=REPO_ROOT,
            env=env,
            label="install Lefthook Git hooks",
        )
    )
    if verify:
        plan.add(
            CommandStep(
                [tool_env.tool("lefthook"), "run", "pre-commit", "--files", "dev.py"],
                cwd=REPO_ROOT,
                env=env,
                label="verify selected hook policy",
            )
        )
    return plan
