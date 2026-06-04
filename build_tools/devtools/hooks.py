# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Git hook command planning."""

from __future__ import annotations

from build_tools.devtools.command_plan import CommandPlan, CommandStep, WriteFileStep
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment


def hook_content(lane: str, profile: str) -> str:
    if lane not in ("bazel", "cmake"):
        raise ValueError(f"unknown lane: {lane}")
    lane_name = {
        "bazel": "Bazel",
        "cmake": "CMake",
    }[lane]
    return f"""# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Local {lane_name}-lane hook policy.
# Installed by `python dev.py {lane} hook --profile {profile}`.
# Test-bearing staged precommit profiles apply fixups before validation.

pre-commit:
  commands:
    precommit:
      run: python dev.py {lane} precommit --profile {profile} {{staged_files}}
"""


def hook_plan(
    lane: str, tool_env: ToolEnvironment, verify: bool, profile: str
) -> CommandPlan:
    plan = CommandPlan()
    plan.add(
        WriteFileStep(
            path=REPO_ROOT / "lefthook-local.yml",
            content=hook_content(lane, profile),
            label=f"select {lane} hook policy with {profile} profile",
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
