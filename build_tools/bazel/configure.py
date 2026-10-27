#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates local Bazel configuration for this checkout."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePath

from build_tools.bazel import configure_project

LOOM_EXECUTE_SUBSTRATES = configure_project.LOOM_EXECUTION_SUBSTRATES
LOOM_IMPORTERS = ("cxx", "mlir", "tilelang")
LOOM_IMAGE_CODECS = ("png", "jpeg", "gif", "webp")
LOOM_AUDIO_CODECS = ("wav", "mp3", "flac")
LOOM_ARCHITECTURES = configure_project.LOOM_ARCHITECTURES
LOOM_ARTIFACT_FORMATS = configure_project.LOOM_ARTIFACT_FORMATS
AMDF_FAMILIES = ("rdna", "cdna", "xdna")
HOST_DRIVERS = ("task",)
DEFAULT_DRIVERS = (*HOST_DRIVERS, "amd")

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
ROCM_DRIVERS = frozenset(("amdgpu",))
AMD_ENGINES = ("amd-gpu", "amd-xdna")
DEPENDENCY_MODES = frozenset(("pinned", "package", "auto"))
SUPPORTED_ENABLE_DRIVERS = frozenset(
    (*HOST_DRIVERS, *AMD_ENGINES, "amdgpu", "vulkan", "webgpu")
)
ALL_DRIVERS = tuple(HOST_DRIVERS) + tuple(SDK_DRIVER_PACKAGES) + ("amd", *AMD_ENGINES)
DRIVER_DEFINES = {
    "IREE_HAL_DRIVER_AMDGPU": "amdgpu",
    "IREE_HAL_DRIVER_TASK": "task",
    "IREE_HAL_DRIVER_VULKAN": "vulkan",
    "IREE_HAL_DRIVER_WEBGPU": "webgpu",
    "IREE_HAL_DRIVER_AMD": "amd",
    "IREE_HAL_DRIVER_AMD_GPU": "amd-gpu",
    "IREE_HAL_DRIVER_AMD_XDNA": "amd-xdna",
}
LOOM_IMPORT_DEFINES = {
    "LOOM_IMPORT_CXX": "cxx",
    "LOOM_IMPORT_MLIR": "mlir",
    "LOOM_IMPORT_TILELANG": "tilelang",
}
AMDF_FAMILY_DEFINES = {
    "AMDF_FAMILY_RDNA": "rdna",
    "AMDF_FAMILY_CDNA": "cdna",
    "AMDF_FAMILY_XDNA": "xdna",
}
REMOVED_OPTIONS = frozenset(
    ("--enable-driver", "--include-driver", "--exclude-driver", "--rocm-path")
)
NATIVE_DRIVER_FLAG = "--//runtime/config/hal:drivers"
NATIVE_RUNTIME_AMDGPU_TARGETS_FLAG = "--//runtime/src/iree/hal/drivers/amdgpu:targets"
NATIVE_LOOM_ARCH_FLAG = "--//loom/config/arch:enable"
NATIVE_LOOM_AMDGPU_TARGETS_FLAG = "--//loom/config/arch/amdgpu:targets"
NATIVE_LOOM_EMIT_FLAG = "--//loom/config/emit:enable"
NATIVE_LOOM_EXECUTE_FLAG = "--//loom/config/execute:enable"
NATIVE_LOOM_IMPORT_FLAG = "--//loom/config/import:enable"
NATIVE_LOOM_IMAGE_INPUT_FLAG = "--//loom/config/serve:image_input"
NATIVE_LOOM_IMAGE_CODECS_FLAG = "--//loom/config/serve:image_codecs"
NATIVE_LOOM_AUDIO_CODECS_FLAG = "--//loom/config/serve:audio_codecs"
NATIVE_AMDF_ENABLED_FLAG = "--//libamdf/config:enabled"
NATIVE_AMDF_FAMILIES_FLAG = "--//libamdf/config:families"
NATIVE_VULKAN_ENABLED_FLAG = "--//build_tools/vulkan/config:enabled"
NATIVE_D3D12_ENABLED_FLAG = "--//build_tools/d3d12/config:enabled"
NATIVE_RDMA_ENABLED_FLAG = "--//runtime/config/net:rdma"
NATIVE_REPO_ENV_PREFIX = "--repo_env="
TRUE_VALUES = frozenset(("1", "ON", "TRUE", "YES"))
FALSE_VALUES = frozenset(("0", "OFF", "FALSE", "NO"))


@dataclass
class ConfigRequest:
    output: Path
    enabled_drivers: set[str] = field(default_factory=lambda: set(DEFAULT_DRIVERS))
    driver_source: str | None = None
    # Explicit engine selections override the AMD umbrella in either order.
    amd_engine_overrides: dict[str, bool] = field(default_factory=dict)
    dependency_mode: str = "pinned"
    rocm_dependency_mode: str | None = None
    enabled_loom_architectures: set[str] = field(default_factory=set)
    runtime_amdgpu_targets: set[str] | None = None
    enabled_loom_execute: set[str] = field(default_factory=set)
    loom_execute_source: str | None = None
    loom_emit_source: str | None = None
    loom_import_source: str | None = None
    # Image input selection, independent of serving's other modalities.
    loom_image_input: bool = True
    # Spelling used to select image input.
    loom_image_input_source: str | None = None
    # Codec providers present in image-enabled builds.
    loom_image_codecs: set[str] = field(default_factory=lambda: set(LOOM_IMAGE_CODECS))
    # Spelling used to select image providers.
    loom_image_codecs_source: str | None = None
    # Optional encoded-audio providers; an empty set retains raw PCM input.
    loom_audio_codecs: set[str] = field(default_factory=lambda: set(LOOM_AUDIO_CODECS))
    # Spelling used to select encoded-audio providers.
    loom_audio_codecs_source: str | None = None
    loom_arch_source: str | None = None
    loom_amdgpu_targets: set[str] | None = None
    enabled_loom_artifact_formats: set[str] = field(default_factory=set)
    enabled_loom_importers: set[str] = field(default_factory=set)
    amdf_build: bool = False
    amdf_build_source: str | None = None
    enabled_amdf_families: set[str] = field(default_factory=lambda: set(AMDF_FAMILIES))
    amdf_family_source: str | None = None
    # Explicit Vulkan API selection, independent of HAL driver selection.
    vulkan_enabled: bool = True
    # Option spelling used for the Vulkan API selection.
    vulkan_source: str | None = None
    # Explicit D3D12 API selection; target-OS compatibility is evaluated by Bazel.
    d3d12_enabled: bool = True
    # Option spelling used for the D3D12 API selection.
    d3d12_source: str | None = None
    # Explicit native RDMA support, independent of host carrier composition.
    rdma_enabled: bool = False
    # Option spelling used for native RDMA selection.
    rdma_source: str | None = None
    rocm_path: str | None = None
    # Opaque state owned by repository project configuration contributors.
    project_state: dict[str, object] = field(default_factory=dict)

    def set_driver(self, driver: str, enabled: bool) -> None:
        if self.driver_source == "native":
            raise SystemExit(
                "Do not mix portable -DIREE_HAL_DRIVER_* options with the "
                f"native {NATIVE_DRIVER_FLAG}=... Bazel option."
            )
        self.driver_source = "portable"
        if driver in AMD_ENGINES:
            self.amd_engine_overrides[driver] = enabled
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

    def set_runtime_amdgpu_target_list(self, targets: set[str]) -> None:
        self.runtime_amdgpu_targets = set(targets)

    def set_loom_arch(self, architecture: str, enabled: bool) -> None:
        if self.loom_arch_source == "native":
            raise SystemExit(
                "Do not mix portable -DLOOM_ARCH_* options with the "
                f"native {NATIVE_LOOM_ARCH_FLAG}=... Bazel option."
            )
        self.loom_arch_source = "portable"
        if enabled:
            self.enabled_loom_architectures.add(architecture)
        else:
            self.enabled_loom_architectures.discard(architecture)

    def set_loom_arch_list(self, architectures: set[str]) -> None:
        if self.loom_arch_source == "portable":
            raise SystemExit(
                "Do not mix portable -DLOOM_ARCH_* options with the "
                f"native {NATIVE_LOOM_ARCH_FLAG}=... Bazel option."
            )
        unknown_architectures = architectures.difference(LOOM_ARCHITECTURES)
        if unknown_architectures:
            raise SystemExit(
                "Unknown Loom architecture(s): {}".format(
                    ", ".join(sorted(unknown_architectures))
                )
            )
        self.loom_arch_source = "native"
        self.enabled_loom_architectures = set(architectures)

    def set_loom_amdgpu_target_list(self, targets: set[str]) -> None:
        self.loom_amdgpu_targets = set(targets)

    def set_loom_emit(self, artifact_format: str, enabled: bool) -> None:
        if self.loom_emit_source == "native":
            raise SystemExit(
                "Do not mix portable -DLOOM_EMIT_* options with the "
                f"native {NATIVE_LOOM_EMIT_FLAG}=... Bazel option."
            )
        self.loom_emit_source = "portable"
        if enabled:
            self.enabled_loom_artifact_formats.add(artifact_format)
        else:
            self.enabled_loom_artifact_formats.discard(artifact_format)

    def set_loom_emit_list(self, artifact_formats: set[str]) -> None:
        if self.loom_emit_source == "portable":
            raise SystemExit(
                "Do not mix portable -DLOOM_EMIT_* options with the "
                f"native {NATIVE_LOOM_EMIT_FLAG}=... Bazel option."
            )
        unknown_formats = artifact_formats.difference(LOOM_ARTIFACT_FORMATS)
        if unknown_formats:
            raise SystemExit(
                "Unknown Loom artifact format(s): {}".format(
                    ", ".join(sorted(unknown_formats))
                )
            )
        self.loom_emit_source = "native"
        self.enabled_loom_artifact_formats = set(artifact_formats)

    def set_loom_execute(self, execute: str, enabled: bool) -> None:
        if self.loom_execute_source == "native":
            raise SystemExit(
                "Do not mix portable -DLOOM_EXECUTE_* options with the "
                f"native {NATIVE_LOOM_EXECUTE_FLAG}=... Bazel option."
            )
        self.loom_execute_source = "portable"
        if enabled:
            self.enabled_loom_execute.add(execute)
        else:
            self.enabled_loom_execute.discard(execute)

    def set_loom_execute_list(self, execute_values: set[str]) -> None:
        if self.loom_execute_source == "portable":
            raise SystemExit(
                "Do not mix portable -DLOOM_EXECUTE_* options with the "
                f"native {NATIVE_LOOM_EXECUTE_FLAG}=... Bazel option."
            )
        unknown_execute_values = execute_values.difference(LOOM_EXECUTE_SUBSTRATES)
        if unknown_execute_values:
            raise SystemExit(
                "Unknown Loom execute substrate(s): {}".format(
                    ", ".join(sorted(unknown_execute_values))
                )
            )
        self.loom_execute_source = "native"
        self.enabled_loom_execute = set(execute_values)

    def set_loom_importer(self, importer: str, enabled: bool) -> None:
        if self.loom_import_source == "native":
            raise SystemExit(
                "Do not mix portable -DLOOM_IMPORT_* options with the "
                f"native {NATIVE_LOOM_IMPORT_FLAG}=... Bazel option."
            )
        self.loom_import_source = "portable"
        if enabled:
            self.enabled_loom_importers.add(importer)
        else:
            self.enabled_loom_importers.discard(importer)

    def set_loom_importer_list(self, importers: set[str]) -> None:
        if self.loom_import_source == "portable":
            raise SystemExit(
                "Do not mix portable -DLOOM_IMPORT_* options with the "
                f"native {NATIVE_LOOM_IMPORT_FLAG}=... Bazel option."
            )
        unknown_importers = importers.difference(LOOM_IMPORTERS)
        if unknown_importers:
            raise SystemExit(
                "Unknown Loom importer(s): {}".format(
                    ", ".join(sorted(unknown_importers))
                )
            )
        self.loom_import_source = "native"
        self.enabled_loom_importers = set(importers)

    def set_vulkan_enabled(self, enabled: bool, source: str) -> None:
        if self.vulkan_source is not None and self.vulkan_source != source:
            raise SystemExit(
                "Do not mix portable -DIREE_ENABLE_VULKAN with the native "
                f"{NATIVE_VULKAN_ENABLED_FLAG}=... Bazel option."
            )
        self.vulkan_source = source
        self.vulkan_enabled = enabled

    def set_d3d12_enabled(self, enabled: bool, source: str) -> None:
        if self.d3d12_source is not None and self.d3d12_source != source:
            raise SystemExit(
                "Do not mix portable -DIREE_ENABLE_D3D12 with the native "
                f"{NATIVE_D3D12_ENABLED_FLAG}=... Bazel option."
            )
        self.d3d12_source = source
        self.d3d12_enabled = enabled

    def set_amdf_build(self, enabled: bool, source: str) -> None:
        if self.amdf_build_source is not None and self.amdf_build_source != source:
            raise SystemExit(
                "Do not mix portable -DAMDF_BUILD with the native "
                f"{NATIVE_AMDF_ENABLED_FLAG}=... Bazel option."
            )
        self.amdf_build_source = source
        self.amdf_build = enabled

    def set_rdma_enabled(self, enabled: bool, source: str) -> None:
        if self.rdma_source is not None and self.rdma_source != source:
            raise SystemExit(
                "Do not mix portable -DIREE_NET_RDMA with the native "
                f"{NATIVE_RDMA_ENABLED_FLAG}=... Bazel option."
            )
        self.rdma_source = source
        self.rdma_enabled = enabled

    def set_loom_image_input(self, enabled: bool, source: str) -> None:
        if self.loom_image_input_source not in (None, source):
            raise SystemExit("Do not mix portable and native Loom image-input options.")
        self.loom_image_input_source = source
        self.loom_image_input = enabled

    def set_loom_image_codecs(self, codecs: set[str], source: str) -> None:
        if self.loom_image_codecs_source not in (None, source):
            raise SystemExit("Do not mix portable and native Loom image-codec options.")
        unknown = codecs.difference(LOOM_IMAGE_CODECS)
        if unknown:
            raise SystemExit(
                "Unknown Loom image codec(s): " + ", ".join(sorted(unknown))
            )
        self.loom_image_codecs_source = source
        self.loom_image_codecs = codecs

    def set_loom_audio_codecs(self, codecs: set[str], source: str) -> None:
        if self.loom_audio_codecs_source not in (None, source):
            raise SystemExit("Do not mix portable and native Loom audio-codec options.")
        unknown = codecs.difference(LOOM_AUDIO_CODECS)
        if unknown:
            raise SystemExit(
                "Unknown Loom audio codec(s): " + ", ".join(sorted(unknown))
            )
        self.loom_audio_codecs_source = source
        self.loom_audio_codecs = codecs

    def set_amdf_family(self, family: str, enabled: bool) -> None:
        if self.amdf_family_source == "native":
            raise SystemExit(
                "Do not mix portable -DAMDF_FAMILY_* options with the native "
                f"{NATIVE_AMDF_FAMILIES_FLAG}=... Bazel option."
            )
        self.amdf_family_source = "portable"
        if enabled:
            self.enabled_amdf_families.add(family)
        else:
            self.enabled_amdf_families.discard(family)

    def set_amdf_family_list(self, families: set[str]) -> None:
        if self.amdf_family_source == "portable":
            raise SystemExit(
                "Do not mix portable -DAMDF_FAMILY_* options with the native "
                f"{NATIVE_AMDF_FAMILIES_FLAG}=... Bazel option."
            )
        unknown_families = families.difference(AMDF_FAMILIES)
        if unknown_families:
            raise SystemExit(
                "Unknown libamdf family or families: {}".format(
                    ", ".join(sorted(unknown_families))
                )
            )
        self.amdf_family_source = "native"
        self.enabled_amdf_families = set(families)

    def set_rocm_path(self, path: str) -> None:
        rocm_path = resolve_rocm_path(path)
        if self.rocm_path is not None and self.rocm_path != rocm_path:
            raise SystemExit(
                "Conflicting ROCm roots: {} and {}".format(self.rocm_path, rocm_path)
            )
        self.rocm_path = rocm_path

    def set_dependency_mode(self, mode: str) -> None:
        dependency_mode = mode.lower()
        if dependency_mode not in DEPENDENCY_MODES:
            raise SystemExit(
                "IREE_DEPENDENCY_MODE must be one of pinned, package, or auto; "
                f"got {mode!r}."
            )
        self.dependency_mode = dependency_mode

    def set_rocm_dependency_mode(self, mode: str) -> None:
        dependency_mode = mode.lower()
        if dependency_mode not in DEPENDENCY_MODES:
            raise SystemExit(
                "IREE_ROCM_DEPENDENCY_MODE must be one of pinned, package, or auto; "
                f"got {mode!r}."
            )
        self.rocm_dependency_mode = dependency_mode

    def effective_rocm_dependency_mode(self) -> str:
        if self.rocm_dependency_mode is not None:
            return self.rocm_dependency_mode
        if self.rocm_path is not None:
            return "package"
        return self.dependency_mode


def windows_symbolic_links_available() -> bool:
    """Returns whether this process can create Windows symbolic links."""
    with tempfile.TemporaryDirectory(prefix="hrx-bazel-symlink-") as temp_directory:
        directory = Path(temp_directory)
        target = directory / "target"
        link = directory / "link"
        target.write_text("", encoding="utf-8")
        try:
            link.symlink_to(target)
        except OSError:
            return False
        return link.is_symlink()


def require_windows_bazel_host(
    *,
    platform_name: str | None = None,
    symbolic_link_probe: Callable[[], bool] | None = None,
) -> None:
    """Fails with setup instructions for missing Windows Bazel capabilities."""
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "win32":
        return
    if symbolic_link_probe is None:
        symbolic_link_probe = windows_symbolic_links_available
    if symbolic_link_probe():
        return
    raise SystemExit(
        "Windows Bazel host requirements are not satisfied:\n\n"
        "Symbolic-link creation is unavailable. rules_python constructs "
        "Windows runfiles and runtime virtual environments with symbolic "
        "links. Enable Windows Developer Mode or grant this account the "
        "'Create symbolic links' user right.\n\n"
        "CI jobs without administrator rights must use a base image where "
        "symbolic-link creation is already provisioned."
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Writes .bazelrc.configured for this checkout. Defaults to a "
            "SDK-free recursive build scope."
        ),
        epilog="""Examples:
  python build_tools/bazel/configure.py
  python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON
  python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_DEPENDENCY_MODE=pinned
  python build_tools/bazel/configure.py --//runtime/config/hal:drivers=amdgpu,task --repo_env=IREE_ROCM_PATH=/opt/rocm
  python build_tools/bazel/configure.py -DAMDF_BUILD=ON -DAMDF_FAMILY_CDNA=OFF
  python build_tools/bazel/configure.py -DLOOM_ARCH_SPIRV=OFF
  python build_tools/bazel/configure.py -DLOOM_ARCH_AMDGPU=ON -DLOOM_EMIT_ELF=ON -DLOOM_EXECUTE_IREE_HAL=ON -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python build_tools/bazel/configure.py -DLOOM_ARCH_AIE2P=ON
  python build_tools/bazel/configure.py -DLOOM_IMPORT_TILELANG=ON

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
    parser.add_argument(
        "--check-windows-host",
        action="store_true",
        help=argparse.SUPPRESS,
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
        "{} was removed. Use portable -DIREE_HAL_DRIVER_* options and "
        "-DIREE_ROCM_PATH=/path/to/rocm when package-mode ROCm headers are "
        "required. The native Bazel form is "
        "{}=amdgpu,task "
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


def as_bazel_path(path: PurePath) -> str:
    """Returns a filesystem path in Bazel's platform-neutral spelling."""
    return path.as_posix()


def resolve_rocm_path(path: str) -> str:
    if not path:
        raise SystemExit("IREE_ROCM_PATH must not be empty.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"IREE_ROCM_PATH does not exist: {resolved}")
    if not (resolved / "include").is_dir():
        raise SystemExit(f"IREE_ROCM_PATH has no include directory: {resolved}")
    return as_bazel_path(resolved)


def apply_define(request: ConfigRequest, define: str) -> None:
    name, value = parse_define(define)
    if configure_project.apply_define(request, name, value, parse_bool):
        return
    if name in DRIVER_DEFINES:
        request.set_driver(DRIVER_DEFINES[name], parse_bool(name, value))
        return
    if name in LOOM_IMPORT_DEFINES:
        request.set_loom_importer(LOOM_IMPORT_DEFINES[name], parse_bool(name, value))
        return
    if name == "AMDF_BUILD":
        request.set_amdf_build(parse_bool(name, value), "portable")
        return
    if name == "LOOM_SERVE_IMAGE_INPUT":
        request.set_loom_image_input(parse_bool(name, value), "portable")
        return
    if name == "LOOM_SERVE_IMAGE_CODECS":
        request.set_loom_image_codecs(set(filter(None, value.split(";"))), "portable")
        return
    if name == "LOOM_SERVE_AUDIO_CODECS":
        request.set_loom_audio_codecs(set(filter(None, value.split(";"))), "portable")
        return
    if name == "IREE_ENABLE_VULKAN":
        request.set_vulkan_enabled(parse_bool(name, value), "portable")
        return
    if name == "IREE_ENABLE_D3D12":
        request.set_d3d12_enabled(parse_bool(name, value), "portable")
        return
    if name == "IREE_NET_RDMA":
        request.set_rdma_enabled(parse_bool(name, value), "portable")
        return
    if name in AMDF_FAMILY_DEFINES:
        request.set_amdf_family(AMDF_FAMILY_DEFINES[name], parse_bool(name, value))
        return
    if name == "IREE_ROCM_PATH":
        request.set_rocm_path(value)
        return
    if name == "IREE_DEPENDENCY_MODE":
        request.set_dependency_mode(value)
        return
    if name == "IREE_ROCM_DEPENDENCY_MODE":
        request.set_rocm_dependency_mode(value)
        return
    raise SystemExit(
        f"Unsupported Bazel configure option -D{name}. Published portable "
        "configuration options are documented in BUILDING.md."
    )


def parse_driver_list(value: str) -> set[str]:
    if not value:
        return set()
    return {driver for driver in value.split(",") if driver}


def parse_string_list(value: str) -> set[str]:
    if not value:
        return set()
    return {part for part in value.split(",") if part}


def apply_native_bazel_arg(request: ConfigRequest, arg: str) -> None:
    check_removed_option(arg)
    if arg == "--":
        raise SystemExit("build_tools/bazel/configure.py does not use a -- separator.")
    if configure_project.apply_native_bazel_arg(request, arg):
        return
    if arg.startswith(NATIVE_LOOM_IMAGE_INPUT_FLAG + "="):
        request.set_loom_image_input(
            parse_bool(NATIVE_LOOM_IMAGE_INPUT_FLAG, arg.split("=", 1)[1]), "native"
        )
        return
    if arg.startswith(NATIVE_LOOM_IMAGE_CODECS_FLAG + "="):
        request.set_loom_image_codecs(parse_string_list(arg.split("=", 1)[1]), "native")
        return
    if arg.startswith(NATIVE_LOOM_AUDIO_CODECS_FLAG + "="):
        request.set_loom_audio_codecs(parse_string_list(arg.split("=", 1)[1]), "native")
        return
    if arg.startswith(NATIVE_DRIVER_FLAG + "="):
        request.set_driver_list(parse_driver_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_DRIVER_FLAG:
        raise SystemExit(f"{NATIVE_DRIVER_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_RUNTIME_AMDGPU_TARGETS_FLAG + "="):
        request.set_runtime_amdgpu_target_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_RUNTIME_AMDGPU_TARGETS_FLAG:
        raise SystemExit(
            f"{NATIVE_RUNTIME_AMDGPU_TARGETS_FLAG} must use --flag=value syntax."
        )
    if arg.startswith(NATIVE_LOOM_ARCH_FLAG + "="):
        request.set_loom_arch_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_ARCH_FLAG:
        raise SystemExit(f"{NATIVE_LOOM_ARCH_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_LOOM_AMDGPU_TARGETS_FLAG + "="):
        request.set_loom_amdgpu_target_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_AMDGPU_TARGETS_FLAG:
        raise SystemExit(
            f"{NATIVE_LOOM_AMDGPU_TARGETS_FLAG} must use --flag=value syntax."
        )
    if arg.startswith(NATIVE_LOOM_EMIT_FLAG + "="):
        request.set_loom_emit_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_EMIT_FLAG:
        raise SystemExit(f"{NATIVE_LOOM_EMIT_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_LOOM_EXECUTE_FLAG + "="):
        request.set_loom_execute_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_EXECUTE_FLAG:
        raise SystemExit(f"{NATIVE_LOOM_EXECUTE_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_LOOM_IMPORT_FLAG + "="):
        request.set_loom_importer_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_IMPORT_FLAG:
        raise SystemExit(f"{NATIVE_LOOM_IMPORT_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_AMDF_ENABLED_FLAG + "="):
        request.set_amdf_build(
            parse_bool(
                NATIVE_AMDF_ENABLED_FLAG,
                arg.split("=", 1)[1],
            ),
            "native",
        )
        return
    if arg == NATIVE_AMDF_ENABLED_FLAG:
        raise SystemExit(f"{NATIVE_AMDF_ENABLED_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_AMDF_FAMILIES_FLAG + "="):
        request.set_amdf_family_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_AMDF_FAMILIES_FLAG:
        raise SystemExit(f"{NATIVE_AMDF_FAMILIES_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_VULKAN_ENABLED_FLAG + "="):
        request.set_vulkan_enabled(
            parse_bool(NATIVE_VULKAN_ENABLED_FLAG, arg.split("=", 1)[1]), "native"
        )
        return
    if arg == NATIVE_VULKAN_ENABLED_FLAG:
        raise SystemExit(f"{NATIVE_VULKAN_ENABLED_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_D3D12_ENABLED_FLAG + "="):
        request.set_d3d12_enabled(
            parse_bool(NATIVE_D3D12_ENABLED_FLAG, arg.split("=", 1)[1]), "native"
        )
        return
    if arg == NATIVE_D3D12_ENABLED_FLAG:
        raise SystemExit(f"{NATIVE_D3D12_ENABLED_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_RDMA_ENABLED_FLAG + "="):
        request.set_rdma_enabled(
            parse_bool(NATIVE_RDMA_ENABLED_FLAG, arg.split("=", 1)[1]), "native"
        )
        return
    if arg == NATIVE_RDMA_ENABLED_FLAG:
        raise SystemExit(f"{NATIVE_RDMA_ENABLED_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_REPO_ENV_PREFIX):
        repo_env = arg[len(NATIVE_REPO_ENV_PREFIX) :]
        if "=" not in repo_env:
            raise SystemExit("--repo_env must use --repo_env=NAME=VALUE syntax.")
        name, value = repo_env.split("=", 1)
        if name == "IREE_ROCM_PATH":
            request.set_rocm_path(value)
            return
        if name == "IREE_DEPENDENCY_MODE":
            request.set_dependency_mode(value)
            return
        if name == "IREE_ROCM_DEPENDENCY_MODE":
            request.set_rocm_dependency_mode(value)
            return
        raise SystemExit(
            "build_tools/bazel/configure.py only accepts "
            "--repo_env=IREE_ROCM_PATH=..., "
            "--repo_env=IREE_DEPENDENCY_MODE=..., and "
            "--repo_env=IREE_ROCM_DEPENDENCY_MODE=... "
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
    configure_project.initialize_request(request)
    for define in args.defines:
        apply_define(request, define)
    for native_bazel_arg in args.native_bazel_args:
        apply_native_bazel_arg(request, native_bazel_arg)
    configure_project.finalize_request(request)
    if "amd" in request.enabled_drivers:
        request.enabled_drivers.remove("amd")
        request.enabled_drivers.update(AMD_ENGINES)
    for engine, enabled in request.amd_engine_overrides.items():
        if enabled:
            request.enabled_drivers.add(engine)
        else:
            request.enabled_drivers.discard(engine)
    return request


def ordered_driver_set(values: set[str]) -> list[str]:
    return [driver for driver in ALL_DRIVERS if driver in values]


def ordered_loom_arch_set(values: set[str]) -> list[str]:
    return [
        architecture for architecture in LOOM_ARCHITECTURES if architecture in values
    ]


def ordered_loom_emit_set(values: set[str]) -> list[str]:
    return [
        artifact_format
        for artifact_format in LOOM_ARTIFACT_FORMATS
        if artifact_format in values
    ]


def ordered_loom_execute_set(values: set[str]) -> list[str]:
    return [execute for execute in LOOM_EXECUTE_SUBSTRATES if execute in values]


def ordered_loom_importer_set(values: set[str]) -> list[str]:
    return [importer for importer in LOOM_IMPORTERS if importer in values]


def ordered_amdf_family_set(values: set[str]) -> list[str]:
    return [family for family in AMDF_FAMILIES if family in values]


def bazelrc_line(command: str, option: str) -> str:
    return f"{command} {shlex.quote(option)}"


def generate_config(args: argparse.Namespace) -> str:
    request = request_from_args(args)
    if request.enabled_drivers.intersection(AMD_ENGINES):
        if request.amdf_build_source is not None and not request.amdf_build:
            raise SystemExit(
                "AMDF_BUILD=OFF conflicts with an enabled native AMD HAL engine"
            )
        request.amdf_build = True
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
        rocm_path = os.environ.get("IREE_ROCM_PATH")
        if rocm_path:
            request.set_rocm_path(rocm_path)

    effective_rocm_dependency_mode = request.effective_rocm_dependency_mode()
    if (
        rocm_enabled_drivers
        and effective_rocm_dependency_mode == "package"
        and not request.rocm_path
    ):
        raise SystemExit(
            "ROCm package dependency mode requires -DIREE_ROCM_PATH=/path/to/rocm, "
            "--repo_env=IREE_ROCM_PATH=/path/to/rocm, or IREE_ROCM_PATH in the "
            "environment when enabling {}. Set IREE_ROCM_DEPENDENCY_MODE=pinned "
            "or auto to use pinned ROCm headers without a ROCm root.".format(
                ", ".join(sorted(rocm_enabled_drivers))
            )
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
        *(
            [
                bazelrc_line(
                    "build",
                    NATIVE_RUNTIME_AMDGPU_TARGETS_FLAG
                    + "="
                    + ",".join(sorted(request.runtime_amdgpu_targets)),
                )
            ]
            if request.runtime_amdgpu_targets is not None
            else []
        ),
        "",
        "# libamdf library and implementation-family scope.",
        bazelrc_line(
            "build",
            "--//libamdf/config:enabled=" + str(request.amdf_build).lower(),
        ),
        bazelrc_line(
            "build",
            "--//libamdf/config:families="
            + ",".join(ordered_amdf_family_set(request.enabled_amdf_families)),
        ),
        "",
        "# Vulkan and D3D12 API clients; neither selection enables a HAL driver.",
        bazelrc_line(
            "build",
            NATIVE_VULKAN_ENABLED_FLAG + "=" + str(request.vulkan_enabled).lower(),
        ),
        bazelrc_line(
            "build",
            NATIVE_D3D12_ENABLED_FLAG + "=" + str(request.d3d12_enabled).lower(),
        ),
        "",
        "# Native RDMA resources; no host carrier is implicitly registered.",
        bazelrc_line(
            "build", NATIVE_RDMA_ENABLED_FLAG + "=" + str(request.rdma_enabled).lower()
        ),
        "",
        "# Source dependency mode.",
        bazelrc_line(
            "common",
            "--repo_env=IREE_DEPENDENCY_MODE=" + request.dependency_mode,
        ),
        "",
        "# Optional encoded audio providers; raw PCM is always available.",
        bazelrc_line(
            "build",
            NATIVE_LOOM_AUDIO_CODECS_FLAG
            + "="
            + ",".join(
                codec
                for codec in LOOM_AUDIO_CODECS
                if codec in request.loom_audio_codecs
            ),
        ),
        "",
        "# Loom architecture, artifact, execution, and importer scope.",
        *("# " + line for line in configure_project.request_summary(request)),
        bazelrc_line(
            "build",
            NATIVE_LOOM_IMAGE_INPUT_FLAG + "=" + str(request.loom_image_input).lower(),
        ),
        bazelrc_line(
            "build",
            NATIVE_LOOM_IMAGE_CODECS_FLAG
            + "="
            + ",".join(
                codec
                for codec in LOOM_IMAGE_CODECS
                if codec in request.loom_image_codecs
            ),
        ),
        bazelrc_line(
            "build",
            "--//loom/config/arch:enable="
            + ",".join(ordered_loom_arch_set(request.enabled_loom_architectures)),
        ),
        *(
            [
                bazelrc_line(
                    "build",
                    NATIVE_LOOM_AMDGPU_TARGETS_FLAG
                    + "="
                    + ",".join(sorted(request.loom_amdgpu_targets)),
                )
            ]
            if request.loom_amdgpu_targets is not None
            else []
        ),
        bazelrc_line(
            "build",
            "--//loom/config/execute:enable="
            + ",".join(ordered_loom_execute_set(request.enabled_loom_execute)),
        ),
        bazelrc_line(
            "build",
            "--//loom/config/import:enable="
            + ",".join(ordered_loom_importer_set(request.enabled_loom_importers)),
        ),
        bazelrc_line(
            "build",
            "--//loom/config/emit:enable="
            + ",".join(ordered_loom_emit_set(request.enabled_loom_artifact_formats)),
        ),
        "",
    ]
    if request.rocm_dependency_mode or request.rocm_path:
        lines.append(
            bazelrc_line(
                "common",
                "--repo_env=IREE_ROCM_DEPENDENCY_MODE="
                + effective_rocm_dependency_mode,
            )
        )
    lines.extend(
        [
            "",
            "# Optional AMDGPU device compiler toolchain.",
        ]
    )
    if "amdgpu" in request.enabled_drivers and request.rocm_path:
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

    return "\n".join(lines)


def main() -> int:
    args = parse_arguments()
    require_windows_bazel_host()
    if args.check_windows_host:
        return 0
    config = generate_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
