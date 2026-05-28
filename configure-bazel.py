#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates local Bazel configuration for this checkout."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


HOST_DRIVERS = ("local-sync", "local-task", "null")

SDK_DRIVER_PACKAGES = {
    "amdgpu": (
        "runtime/src/iree/hal/drivers/amdgpu",
        "runtime/src/iree/hal/drivers/amdgpu/abi",
        "runtime/src/iree/hal/drivers/amdgpu/cts",
        "runtime/src/iree/hal/drivers/amdgpu/device",
        "runtime/src/iree/hal/drivers/amdgpu/device/binaries",
        "runtime/src/iree/hal/drivers/amdgpu/device/binaries/prebuilt",
        "runtime/src/iree/hal/drivers/amdgpu/device/binaries/source",
        "runtime/src/iree/hal/drivers/amdgpu/registration",
        "runtime/src/iree/hal/drivers/amdgpu/util",
    ),
    "cuda": (
        "runtime/src/iree/hal/drivers/cuda",
        "runtime/src/iree/hal/drivers/cuda/cts",
        "runtime/src/iree/hal/drivers/cuda/registration",
    ),
    "hip": (
        "runtime/src/iree/hal/drivers/hip",
        "runtime/src/iree/hal/drivers/hip/cts",
        "runtime/src/iree/hal/drivers/hip/registration",
        "runtime/src/iree/hal/drivers/hip/util",
    ),
    "vulkan": (
        "runtime/src/iree/hal/drivers/vulkan",
        "runtime/src/iree/hal/drivers/vulkan/cts",
        "runtime/src/iree/hal/drivers/vulkan/registration",
        "runtime/src/iree/hal/drivers/vulkan/util",
    ),
    "webgpu": (
        "runtime/src/iree/hal/drivers/webgpu",
        "runtime/src/iree/hal/drivers/webgpu/cts",
        "runtime/src/iree/hal/drivers/webgpu/registration",
    ),
}

ROCM_DRIVERS = frozenset(("amdgpu", "hip"))
SUPPORTED_ENABLE_DRIVERS = frozenset((*HOST_DRIVERS, "amdgpu"))
ALL_DRIVERS = tuple(HOST_DRIVERS) + tuple(SDK_DRIVER_PACKAGES)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Writes .bazelrc.configured for this checkout. Defaults to a "
            "host-only SDK-free recursive build scope."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".bazelrc.configured"),
        help="Path to write. Defaults to .bazelrc.configured.",
    )
    parser.add_argument(
        "--enable-driver",
        "--include-driver",
        action="append",
        choices=ALL_DRIVERS,
        default=[],
        dest="enabled_drivers",
        help="Driver to include in the runtime driver registry and package scope.",
    )
    parser.add_argument(
        "--exclude-driver",
        action="append",
        choices=ALL_DRIVERS,
        default=[],
        dest="excluded_drivers",
        help="Driver to exclude from the runtime driver registry and package scope.",
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        help="ROCm or TheRock SDK root. Required when enabling AMDGPU or HIP.",
    )
    return parser.parse_args()


def resolve_rocm_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"--rocm-path does not exist: {resolved}")
    if not (resolved / "include").is_dir():
        raise SystemExit(f"--rocm-path has no include directory: {resolved}")
    return str(resolved)


def ordered_driver_set(values: set[str]) -> list[str]:
    return [driver for driver in ALL_DRIVERS if driver in values]


def bazelrc_line(command: str, option: str) -> str:
    return f"{command} {shlex.quote(option)}"


def deleted_package_lines(enabled_drivers: set[str]) -> list[str]:
    lines = []
    for driver, packages in SDK_DRIVER_PACKAGES.items():
        if driver in enabled_drivers:
            continue
        lines.append(bazelrc_line(
            "common",
            "--deleted_packages=" + ",".join(packages),
        ))
    return lines


def generate_config(args: argparse.Namespace) -> str:
    enabled_drivers = set(HOST_DRIVERS)
    enabled_drivers.update(args.enabled_drivers)
    enabled_drivers.difference_update(args.excluded_drivers)

    unsupported_enabled_drivers = enabled_drivers.difference(SUPPORTED_ENABLE_DRIVERS)
    if unsupported_enabled_drivers:
        raise SystemExit(
            "Driver enablement is not yet configured for: {}. "
            "Leave these drivers excluded until their repositories have been "
            "ported.".format(", ".join(sorted(unsupported_enabled_drivers)))
        )

    rocm_path = resolve_rocm_path(args.rocm_path)
    rocm_enabled_drivers = enabled_drivers.intersection(ROCM_DRIVERS)
    if rocm_enabled_drivers and not rocm_path:
        raise SystemExit(
            "--rocm-path is required when enabling {}".format(
                ", ".join(sorted(rocm_enabled_drivers))
            )
        )

    lines = [
        "# AUTO-GENERATED by ./configure-bazel.py; do not edit.",
        "# Re-run ./configure-bazel.py or use .bazelrc.local for local overrides.",
        "",
        "# Runtime driver registry scope.",
        bazelrc_line(
            "build",
            "--//runtime/config/hal:drivers=" + ",".join(
                ordered_driver_set(enabled_drivers)
            ),
        ),
        "",
        "# Optional AMDGPU device compiler toolchain.",
    ]
    if "amdgpu" in enabled_drivers:
        lines.append(bazelrc_line(
            "common",
            "--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",
        ))
    else:
        lines.append(bazelrc_line(
            "common",
            "--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none",
        ))

    if rocm_path:
        lines.extend([
            bazelrc_line("common", "--repo_env=IREE_ROCM_PATH=" + rocm_path),
            "",
        ])
    else:
        lines.append("")

    lines.extend([
        "# Recursive build package scope.",
        *deleted_package_lines(enabled_drivers),
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_arguments()
    config = generate_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
