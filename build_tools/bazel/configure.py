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
from dataclasses import dataclass, field
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
DRIVER_DEFINES = {
    "IREE_HAL_DRIVER_AMDGPU": "amdgpu",
    "IREE_HAL_DRIVER_LOCAL_SYNC": "local-sync",
    "IREE_HAL_DRIVER_LOCAL_TASK": "local-task",
    "IREE_HAL_DRIVER_NULL": "null",
}
UNSUPPORTED_DRIVER_DEFINES = {
    "IREE_HAL_DRIVER_CUDA": "cuda",
    "IREE_HAL_DRIVER_HIP": "hip",
    "IREE_HAL_DRIVER_METAL": "metal",
    "IREE_HAL_DRIVER_VULKAN": "vulkan",
    "IREE_HAL_DRIVER_WEBGPU": "webgpu",
}
REMOVED_OPTIONS = frozenset(
    ("--enable-driver", "--include-driver", "--exclude-driver", "--rocm-path")
)
NATIVE_DRIVER_FLAG = "--//runtime/config/hal:drivers"
NATIVE_REPO_ENV_PREFIX = "--repo_env="
TRUE_VALUES = frozenset(("1", "ON", "TRUE", "YES"))
FALSE_VALUES = frozenset(("0", "OFF", "FALSE", "NO"))


@dataclass
class ConfigRequest:
    output: Path
    enabled_drivers: set[str] = field(default_factory=lambda: set(HOST_DRIVERS))
    driver_source: str | None = None
    rocm_path: str | None = None

    def set_driver(self, driver: str, enabled: bool) -> None:
        if self.driver_source == "native":
            raise SystemExit(
                "Do not mix portable -DIREE_HAL_DRIVER_* options with the "
                f"native {NATIVE_DRIVER_FLAG}=... Bazel option."
            )
        self.driver_source = "portable"
        if enabled:
            self.enabled_drivers.add(driver)
        else:
            self.enabled_drivers.discard(driver)

    def set_driver_list(self, drivers: set[str]) -> None:
        if self.driver_source == "portable":
            raise SystemExit(
                "Do not mix portable -DIREE_HAL_DRIVER_* options with the "
                f"native {NATIVE_DRIVER_FLAG}=... Bazel option."
            )
        unknown_drivers = drivers.difference(ALL_DRIVERS)
        if unknown_drivers:
            raise SystemExit(
                "Unknown runtime HAL driver(s): {}".format(
                    ", ".join(sorted(unknown_drivers))
                )
            )
        self.driver_source = "native"
        self.enabled_drivers = set(drivers)

    def set_rocm_path(self, path: str) -> None:
        rocm_path = resolve_rocm_path(path)
        if self.rocm_path is not None and self.rocm_path != rocm_path:
            raise SystemExit(
                "Conflicting ROCm roots: {} and {}".format(self.rocm_path, rocm_path)
            )
        self.rocm_path = rocm_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Writes .bazelrc.configured for this checkout. Defaults to a "
            "host-only SDK-free recursive build scope."
        ),
        epilog="""Examples:
  python build_tools/bazel/configure.py
  python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python build_tools/bazel/configure.py --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm

Portable -D project options are documented in BUILDING.md. Other Bazel-native
overrides belong in .bazelrc.local.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".bazelrc.configured"),
        help="Path to write. Defaults to .bazelrc.configured.",
    )
    parser.add_argument(
        "-D",
        action="append",
        default=[],
        dest="defines",
        metavar="NAME=VALUE",
        help="Portable project configuration option documented in BUILDING.md.",
    )
    args, native_bazel_args = parser.parse_known_args(argv)
    args.native_bazel_args = native_bazel_args
    return args


def option_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def check_removed_option(arg: str) -> None:
    removed_option = option_name(arg)
    if removed_option not in REMOVED_OPTIONS:
        return
    raise SystemExit(
        "{} was removed. Use -DIREE_HAL_DRIVER_AMDGPU=ON and "
        "-DIREE_ROCM_PATH=/path/to/rocm, or the native Bazel form "
        "{}=amdgpu,local-sync,local-task,null "
        "--repo_env=IREE_ROCM_PATH=/path/to/rocm.".format(
            removed_option, NATIVE_DRIVER_FLAG
        )
    )


def parse_bool(name: str, value: str) -> bool:
    normalized_value = value.upper()
    if normalized_value in TRUE_VALUES:
        return True
    if normalized_value in FALSE_VALUES:
        return False
    raise SystemExit(f"{name} expects ON or OFF, got {value!r}.")


def parse_define(define: str) -> tuple[str, str]:
    if "=" not in define:
        raise SystemExit(f"-D{define} must use NAME=VALUE.")
    name, value = define.split("=", 1)
    name = name.split(":", 1)[0]
    if not name:
        raise SystemExit(f"-D{define} has no option name.")
    return name, value


def resolve_rocm_path(path: str) -> str:
    if not path:
        raise SystemExit("IREE_ROCM_PATH must not be empty.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"IREE_ROCM_PATH does not exist: {resolved}")
    if not (resolved / "include").is_dir():
        raise SystemExit(f"IREE_ROCM_PATH has no include directory: {resolved}")
    return str(resolved)


def apply_define(request: ConfigRequest, define: str) -> None:
    name, value = parse_define(define)
    if name in DRIVER_DEFINES:
        request.set_driver(DRIVER_DEFINES[name], parse_bool(name, value))
        return
    if name in UNSUPPORTED_DRIVER_DEFINES:
        if parse_bool(name, value):
            raise SystemExit(f"{name}=ON is not configured in the Bazel graph yet.")
        return
    if name == "IREE_ROCM_PATH":
        request.set_rocm_path(value)
        return
    raise SystemExit(
        f"Unsupported Bazel configure option -D{name}. Published portable "
        "configuration options are documented in BUILDING.md."
    )


def parse_driver_list(value: str) -> set[str]:
    if not value:
        return set()
    return {driver for driver in value.split(",") if driver}


def apply_native_bazel_arg(request: ConfigRequest, arg: str) -> None:
    check_removed_option(arg)
    if arg == "--":
        raise SystemExit("build_tools/bazel/configure.py does not use a -- separator.")
    if arg.startswith(NATIVE_DRIVER_FLAG + "="):
        request.set_driver_list(parse_driver_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_DRIVER_FLAG:
        raise SystemExit(f"{NATIVE_DRIVER_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_REPO_ENV_PREFIX):
        repo_env = arg[len(NATIVE_REPO_ENV_PREFIX) :]
        if "=" not in repo_env:
            raise SystemExit("--repo_env must use --repo_env=NAME=VALUE syntax.")
        name, value = repo_env.split("=", 1)
        if name == "IREE_ROCM_PATH":
            request.set_rocm_path(value)
            return
        raise SystemExit(
            "build_tools/bazel/configure.py only accepts "
            "--repo_env=IREE_ROCM_PATH=... "
            "for generated configuration. Put other Bazel-native overrides "
            "in .bazelrc.local."
        )
    raise SystemExit(
        f"Unsupported Bazel configure argument {arg!r}. Use documented -D "
        "project options, the native runtime driver build setting, or "
        ".bazelrc.local for checkout-specific Bazel overrides."
    )


def request_from_args(args: argparse.Namespace) -> ConfigRequest:
    request = ConfigRequest(output=args.output)
    for define in args.defines:
        apply_define(request, define)
    for native_bazel_arg in args.native_bazel_args:
        apply_native_bazel_arg(request, native_bazel_arg)
    return request


def ordered_driver_set(values: set[str]) -> list[str]:
    return [driver for driver in ALL_DRIVERS if driver in values]


def bazelrc_line(command: str, option: str) -> str:
    return f"{command} {shlex.quote(option)}"


def deleted_package_lines(enabled_drivers: set[str]) -> list[str]:
    lines = []
    for driver, packages in SDK_DRIVER_PACKAGES.items():
        if driver in enabled_drivers:
            continue
        lines.append(
            bazelrc_line(
                "common",
                "--deleted_packages=" + ",".join(packages),
            )
        )
    return lines


def generate_config(args: argparse.Namespace) -> str:
    request = request_from_args(args)
    unsupported_enabled_drivers = request.enabled_drivers.difference(
        SUPPORTED_ENABLE_DRIVERS
    )
    if unsupported_enabled_drivers:
        raise SystemExit(
            "Driver enablement is not yet configured for: {}. "
            "Leave these drivers disabled until their repositories have been "
            "ported.".format(", ".join(sorted(unsupported_enabled_drivers)))
        )

    rocm_enabled_drivers = request.enabled_drivers.intersection(ROCM_DRIVERS)
    if rocm_enabled_drivers and not request.rocm_path:
        raise SystemExit(
            "-DIREE_ROCM_PATH=/path/to/rocm or "
            "--repo_env=IREE_ROCM_PATH=/path/to/rocm is required when enabling "
            "{}.".format(", ".join(sorted(rocm_enabled_drivers)))
        )

    lines = [
        "# AUTO-GENERATED by build_tools/bazel/configure.py; do not edit.",
        "# Re-run python build_tools/bazel/configure.py or use .bazelrc.local for local overrides.",
        "",
        "# Runtime driver registry scope.",
        bazelrc_line(
            "build",
            "--//runtime/config/hal:drivers="
            + ",".join(ordered_driver_set(request.enabled_drivers)),
        ),
        "",
        "# Optional AMDGPU device compiler toolchain.",
    ]
    if "amdgpu" in request.enabled_drivers:
        lines.append(
            bazelrc_line(
                "common",
                "--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",
            )
        )
    else:
        lines.append(
            bazelrc_line(
                "common",
                "--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none",
            )
        )

    if request.rocm_path:
        lines.extend(
            [
                bazelrc_line(
                    "common", "--repo_env=IREE_ROCM_PATH=" + request.rocm_path
                ),
                "",
            ]
        )
    else:
        lines.append("")

    lines.extend(
        [
            "# Recursive build package scope.",
            *deleted_package_lines(request.enabled_drivers),
            "",
        ]
    )
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
