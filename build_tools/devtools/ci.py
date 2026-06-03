#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Local runner for source-tree IREE CI jobs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__:
    from . import ci_config
else:
    sys.path.insert(0, str(REPO_ROOT))
    from build_tools.devtools import ci_config


@dataclass(frozen=True)
class CiStep:
    name: str
    argv: tuple[str, ...]

    def command_line(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class StepResult:
    step: CiStep
    returncode: int
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def command_targets(explicit_targets: list[str] | None = None) -> tuple[str, ...]:
    if explicit_targets:
        return tuple(explicit_targets)
    targets = []
    for directory in ci_config.IREE_TARGET_DIRECTORIES:
        if (REPO_ROOT / directory).is_dir():
            targets.append(f"//{directory}/...")
    if not targets:
        raise RuntimeError("no IREE target directories exist in this checkout")
    return tuple(targets)


def dev_command(*args: str) -> tuple[str, ...]:
    return (os.environ.get("IREE_CI_PYTHON", "python3"), "dev.py", *args)


def bazel_configure_step(enable_amdgpu: bool = False) -> CiStep:
    command = ["bazel", "configure"]
    if enable_amdgpu:
        command.append("-DIREE_HAL_DRIVER_AMDGPU=ON")
    return CiStep("Configure Bazel", dev_command(*command))


def bazel_build_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
) -> CiStep:
    command = ["bazel", "build", *targets]
    if config is not None:
        command.append(f"--config={config}")
    return CiStep(name, dev_command(*command))


def bazel_test_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
    test_tag_filters: tuple[str, ...] = (),
) -> CiStep:
    options = []
    if config is not None:
        options.append(f"--config={config}")
    if test_tag_filters:
        options.append("--test_tag_filters=" + ",".join(test_tag_filters))
    command = ["bazel", "test", *options]
    if any(target.startswith("-") for target in targets):
        command.append("--")
    command.extend(targets)
    return CiStep(name, dev_command(*command))


def cpu_steps(targets: tuple[str, ...]) -> list[CiStep]:
    return [
        bazel_configure_step(),
        bazel_build_step("Build IREE", targets),
        bazel_test_step("Test IREE", targets + ci_config.CPU_XFAIL_TARGETS),
    ]


def cpu_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step()]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.append(
            bazel_test_step(
                f"Test IREE with {config.upper()}",
                targets + ci_config.CPU_SANITIZERS_XFAIL_TARGETS,
                config=config,
            )
        )
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        steps.append(
            bazel_build_step(
                f"Build IREE with {config.upper()}",
                targets,
                config=config,
            )
        )
    return steps


def amdgpu_test_steps(
    targets: tuple[str, ...],
    config: str | None = None,
    xfail_targets: tuple[str, ...] = (),
) -> list[CiStep]:
    config_name = f" and {config.upper()}" if config is not None else ""
    return [
        bazel_test_step(
            f"Test IREE AMDGPU resources{config_name}",
            targets + xfail_targets,
            config=config,
            test_tag_filters=(ci_config.AMDGPU_RESOURCE_TAG,),
        ),
    ]


def amdgpu_steps(targets: tuple[str, ...]) -> list[CiStep]:
    return [
        bazel_configure_step(enable_amdgpu=True),
        bazel_build_step(
            "Build IREE with AMDGPU",
            ci_config.AMDGPU_DRIVER_TARGETS,
        ),
        *amdgpu_test_steps(targets, xfail_targets=ci_config.AMDGPU_XFAIL_TARGETS),
    ]


def amdgpu_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step(enable_amdgpu=True)]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.extend(
            amdgpu_test_steps(
                targets,
                config=config,
                xfail_targets=ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS,
            )
        )
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        steps.append(
            bazel_build_step(
                f"Build IREE with AMDGPU and {config.upper()}",
                ci_config.AMDGPU_DRIVER_TARGETS,
                config=config,
            )
        )
    return steps


def steps_from_args(args: argparse.Namespace) -> list[CiStep]:
    targets = command_targets(args.target)
    if args.command == "iree-cpu":
        return cpu_steps(targets)
    if args.command == "iree-cpu-sanitizers":
        return cpu_sanitizer_steps(targets)
    if args.command == "iree-amdgpu":
        return amdgpu_steps(targets)
    if args.command == "iree-amdgpu-sanitizers":
        return amdgpu_sanitizer_steps(targets)
    raise ValueError(f"unknown CI command: {args.command}")


def github_actions_enabled() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def print_group_start(name: str) -> None:
    if github_actions_enabled():
        print(f"::group::{name}")


def print_group_end() -> None:
    if github_actions_enabled():
        print("::endgroup::")


def run_step(step: CiStep, verbose: bool) -> StepResult:
    print(f"[run] {step.name}", flush=True)
    if verbose or github_actions_enabled():
        print("  " + step.command_line(), flush=True)
    start_time = time.monotonic()
    returncode = subprocess.run(step.argv, cwd=REPO_ROOT).returncode
    elapsed_seconds = time.monotonic() - start_time
    result = StepResult(step, returncode, elapsed_seconds)
    if result.ok:
        print(f"[ok] {step.name} ({elapsed_seconds:.1f}s)", flush=True)
    else:
        print(
            f"[fail] {step.name} ({elapsed_seconds:.1f}s, exit {returncode})",
            flush=True,
        )
        print("  " + step.command_line(), flush=True)
    return result


def write_step_summary(results: list[StepResult]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## IREE CI",
        "",
        "| Phase | Result | Time |",
        "| --- | --- | ---: |",
    ]
    for result in results:
        outcome = "pass" if result.ok else f"fail ({result.returncode})"
        lines.append(
            f"| {result.step.name} | {outcome} | {result.elapsed_seconds:.1f}s |"
        )
    with Path(summary_path).open("a", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def run_steps(
    steps: list[CiStep],
    *,
    dry_run: bool,
    keep_going: bool,
    verbose: bool,
) -> int:
    if dry_run:
        for step in steps:
            print(step.command_line())
        return 0

    print("== IREE CI ==", flush=True)
    results = []
    for step in steps:
        print_group_start(step.name)
        try:
            result = run_step(step, verbose=verbose)
        finally:
            print_group_end()
        results.append(result)
        if not result.ok and not keep_going:
            write_step_summary(results)
            return result.returncode

    write_step_summary(results)
    failures = [result for result in results if not result.ok]
    if failures:
        print("", flush=True)
        print("Failed phases:", flush=True)
        for failure in failures:
            print(f"  {failure.step.name}: {failure.returncode}", flush=True)
        return failures[-1].returncode
    return 0


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-tree IREE CI command groups locally.",
    )
    parser.add_argument(
        "command",
        choices=(
            "iree-cpu",
            "iree-cpu-sanitizers",
            "iree-amdgpu",
            "iree-amdgpu-sanitizers",
        ),
        help="CI command group to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying dev.py commands without executing them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Run every phase and report failures at the end.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each underlying command before running it.",
    )
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "Bazel target pattern to build/test. Defaults to the IREE target "
            "directories present in the checkout."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    return run_steps(
        steps_from_args(args),
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
