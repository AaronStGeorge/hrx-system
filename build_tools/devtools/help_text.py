# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command help for the developer command router."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandHelp:
    description: str
    arguments: str | None = None
    epilog: str | None = None


def root_command_help(command: str) -> CommandHelp:
    if command == "setup":
        return CommandHelp(
            description="Set up shared developer tools.",
            epilog="""Examples:
  python dev.py setup
  python dev.py setup --system
  python dev.py setup --tool-root ../.tools/iree-x

The default setup creates or updates the repo-local .venv. Use --system when
tools are already available on PATH and this checkout should not get a managed
tool environment.""",
        )
    if command == "doctor":
        return CommandHelp(
            description="Check shared developer tools and show their versions.",
            epilog="""Examples:
  python dev.py doctor
  python dev.py doctor --system
  python dev.py doctor --tool-root ../.tools/iree-x""",
        )
    return CommandHelp(description=f"Run {command}.")


def lane_command_help(lane: str, command: str) -> CommandHelp:
    if command == "setup":
        return CommandHelp(
            description=f"Set up developer tools for the {lane} lane.",
            epilog=f"""Examples:
  python dev.py {lane} setup
  python dev.py {lane} setup --system
  python dev.py {lane} setup --tool-root ../.tools/iree-x

The default setup creates or updates .venv. Use --system when tools are already
installed and this checkout should not get a local tool environment.""",
        )
    if command == "hook":
        default_profile = {
            "bazel": "paranoid",
            "cmake": "default",
        }[lane]
        return CommandHelp(
            description=f"Install Git hooks for the {lane} lane.",
            epilog=f"""Examples:
  python dev.py {lane} hook
  python dev.py {lane} hook --profile {default_profile}
  python dev.py {lane} hook --verify

This writes ignored lefthook-local.yml with the selected lane/profile and then
runs lefthook install. Re-run this command with a different --profile to change
the default profile used by Git commits.""",
        )
    if command == "configure":
        if lane == "bazel":
            return CommandHelp(
                description="Configure the Bazel source graph for this checkout.",
                arguments="Portable -D project options or documented native Bazel rc options.",
                epilog="""Examples:
  python dev.py bazel configure
  python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python dev.py bazel configure --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm

This writes .bazelrc.configured. Published portable build options live in
BUILDING.md. Use .bazelrc.local for checkout-specific Bazel overrides.""",
            )
        return CommandHelp(
            description="Configure the CMake package/install-test build tree.",
            arguments="CMake configure options.",
            epilog="""Examples:
  python dev.py cmake configure
  python dev.py cmake configure --fresh
  python dev.py cmake configure -DCMAKE_BUILD_TYPE=Debug
  python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python dev.py --cmake-build-dir build/cmake-asan cmake configure -DIREE_ENABLE_ASAN=ON

The build tree lives outside the checkout at ../builds/<checkout-name>/.
Use --cmake-build-dir to select a different tree for CI, experiments, or
parallel configurations.
Published project build options live in BUILDING.md.""",
        )
    if command == "build":
        if lane == "bazel":
            return CommandHelp(
                description="Build Bazel targets.",
                arguments="Bazel targets or build options. Defaults to //runtime/... and //libhrx/....",
                epilog="""Examples:
  python dev.py bazel build
  python dev.py bazel build //libhrx/...
  python dev.py bazel build //runtime/src/iree/base/...

With no explicit target, this builds //runtime/... and //libhrx/....""",
            )
        return CommandHelp(
            description="Build CMake targets in the configured build tree.",
            arguments="Target names followed by native CMake build options.",
            epilog="""Examples:
  python dev.py cmake build hrx
  python dev.py cmake build libhrx_src_libhrx_hrx
  python dev.py cmake build hrx --parallel 8
  python dev.py cmake build --parallel 8

Positional arguments are target names and become cmake --build ... --target
<name>. Option-looking arguments are forwarded to CMake.""",
        )
    if command == "test":
        if lane == "bazel":
            return CommandHelp(
                description="Run Bazel tests with the presubmit configuration.",
                arguments="Bazel test targets. Defaults to //runtime/... and //libhrx/....",
                epilog="""Examples:
  python dev.py bazel test
  python dev.py bazel test //build_tools/devtools:cli_test
  python dev.py bazel test //libhrx/...""",
            )
        return CommandHelp(
            description="Run CTest in the configured CMake build tree.",
            arguments="CTest options.",
            epilog="""Examples:
  python dev.py cmake test
  python dev.py cmake test -R hrx
  python dev.py cmake test --rerun-failed

CTest runs in ../builds/<checkout-name>/ with --output-on-failure.""",
        )
    if command == "presubmit":
        return CommandHelp(
            description=f"Run CI-shaped acceptance checks for the {lane} lane.",
            epilog=f"""Examples:
  python dev.py {lane} presubmit
  python dev.py {lane} presubmit --profile paranoid
  python dev.py {lane} presubmit --no-project-tests
  python dev.py {lane} presubmit --verbose

Profiles:
  default   Repository hygiene.
  paranoid  Hygiene plus affected project tests and static-analysis providers.
  ci        Full tracked-tree hygiene plus all configured project tests.

The default profile is ci. This is intentionally the expensive full-tree check
that approximates the project test workflows a developer can run locally. Use
--no-project-tests only for CI jobs that have separate project test workflows.
Use `python dev.py {lane} precommit` for local changes only.""",
        )
    if command == "precommit":
        default_profile = {
            "bazel": "paranoid",
            "cmake": "default",
        }[lane]
        if lane == "bazel":
            lane_scope = (
                "The Bazel lane defaults to the paranoid profile: repository "
                "hygiene, affected project tests, and configured "
                "static-analysis providers."
            )
        else:
            lane_scope = (
                "The CMake lane defaults to the default profile. Select "
                "paranoid or ci to add affected project CMake/CTest checks."
            )
        return CommandHelp(
            description=f"Run non-mutating checks for local {lane} changes.",
            epilog=f"""Examples:
  python dev.py {lane} precommit
  python dev.py {lane} precommit --profile {default_profile}
  python dev.py {lane} precommit --base origin/main
  python dev.py {lane} precommit --staged
  python dev.py {lane} precommit README.md CONTRIBUTING.md
  python dev.py {lane} precommit --verbose

With no input option, precommit checks staged, unstaged, and untracked files.
`--base` checks branch changes from the merge base with the given ref through
HEAD, plus local staged, unstaged, and untracked files.
Explicit paths check only those files and are used by the generated Git hook.
The default profile is {default_profile}. Use --profile to select default,
paranoid, or ci for this run.

{lane_scope}""",
        )
    if command == "fix":
        return CommandHelp(
            description="Apply staged mechanical fixups.",
            epilog=f"""Examples:
  python dev.py {lane} fix

Fix runs the staged-file formatter/generated-file repair path. The Git hook and
presubmit commands stay non-mutating.""",
        )
    if command == "doctor":
        return CommandHelp(
            description=f"Check tools required by the {lane} lane.",
            epilog=f"""Examples:
  python dev.py {lane} doctor
  python dev.py {lane} doctor --system
  python dev.py {lane} doctor --tool-root ../.tools/iree-x""",
        )
    if command in ("run", "try", "fuzz"):
        return CommandHelp(
            description=f"{command} support for the {lane} lane.",
            arguments="Command arguments.",
            epilog="This command slot is reserved but not wired yet.",
        )
    return CommandHelp(description=f"Run {lane} {command}.")
