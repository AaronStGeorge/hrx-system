# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom target capability presets and Bazel-configure expansion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

ARCHITECTURES = ("aie2p", "amdgpu", "spirv", "vm", "wasm", "x86")
ARTIFACT_FORMATS = ("elf", "spirv", "vm", "wasm")
EXECUTION_SUBSTRATES = ("iree_hal", "iree_vm", "wasm_host")

ARCH_DEFINES = {
    "LOOM_ARCH_AIE2P": "aie2p",
    "LOOM_ARCH_AMDGPU": "amdgpu",
    "LOOM_ARCH_SPIRV": "spirv",
    "LOOM_ARCH_VM": "vm",
    "LOOM_ARCH_WASM": "wasm",
    "LOOM_ARCH_X86": "x86",
}
EMIT_DEFINES = {
    "LOOM_EMIT_ELF": "elf",
    "LOOM_EMIT_SPIRV": "spirv",
    "LOOM_EMIT_VM": "vm",
    "LOOM_EMIT_WASM": "wasm",
}
EXECUTE_DEFINES = {
    "LOOM_EXECUTE_IREE_HAL": "iree_hal",
    "LOOM_EXECUTE_IREE_VM": "iree_vm",
    "LOOM_EXECUTE_WASM_HOST": "wasm_host",
}


@dataclass(frozen=True)
class TargetCapabilities:
    architectures: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    execution: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetPreset:
    key: str
    option: str
    description: str
    includes: tuple[str, ...] = ()
    capabilities: TargetCapabilities = field(default_factory=TargetCapabilities)

    @property
    def bazel_config(self) -> str:
        return "loom-target-" + self.key.replace("_", "-")


class ConfigurationRequest(Protocol):
    project_state: dict[str, object]
    enabled_loom_architectures: set[str]
    enabled_loom_artifact_formats: set[str]
    enabled_loom_execute: set[str]
    loom_arch_source: str | None
    loom_emit_source: str | None
    loom_execute_source: str | None

    def set_loom_arch(self, architecture: str, enabled: bool) -> None: ...

    def set_loom_emit(self, artifact_format: str, enabled: bool) -> None: ...

    def set_loom_execute(self, execute: str, enabled: bool) -> None: ...


TARGET_PRESETS = (
    TargetPreset(
        key="vm",
        option="LOOM_TARGET_ALL_VM",
        description="Build Loom VM compilation and execution support.",
        capabilities=TargetCapabilities(("vm",), ("vm",), ("iree_vm",)),
    ),
    TargetPreset(
        key="cpu",
        option="LOOM_TARGET_ALL_CPU",
        description="Build Loom CPU and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("x86",), ("elf",), ("iree_hal",)),
    ),
    TargetPreset(
        key="amd_gpu",
        option="LOOM_TARGET_ALL_AMD_GPU",
        description="Build Loom AMD GPU and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("amdgpu",), ("elf",), ("iree_hal",)),
    ),
    TargetPreset(
        key="amd_npu",
        option="LOOM_TARGET_ALL_AMD_NPU",
        description="Build Loom AMD NPU and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("aie2p",), ("elf",), ("iree_hal",)),
    ),
    TargetPreset(
        key="amd",
        option="LOOM_TARGET_ALL_AMD",
        description="Build Loom AMD GPU, NPU, and supporting VM capabilities.",
        includes=("amd_gpu", "amd_npu"),
    ),
    TargetPreset(
        key="spirv",
        option="LOOM_TARGET_ALL_SPIRV",
        description="Build Loom SPIR-V and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("spirv",), ("spirv",), ("iree_hal",)),
    ),
    TargetPreset(
        key="wasm",
        option="LOOM_TARGET_ALL_WASM",
        description="Build Loom WebAssembly and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("wasm",), ("wasm",), ("wasm_host",)),
    ),
    TargetPreset(
        key="default",
        option="LOOM_TARGET_DEFAULT",
        description="Build the ordinary non-WebAssembly Loom target set.",
        includes=("cpu", "amd", "spirv"),
    ),
    TargetPreset(
        key="all",
        option="LOOM_TARGET_ALL",
        description="Build every Loom target capability.",
        includes=("default", "wasm"),
    ),
)

_PRESETS_BY_KEY = {preset.key: preset for preset in TARGET_PRESETS}
_PRESETS_BY_OPTION = {preset.option: preset for preset in TARGET_PRESETS}
_PRESETS_BY_BAZEL_CONFIG = {preset.bazel_config: preset for preset in TARGET_PRESETS}
_STATE_KEY = "loom_capability_presets"


@dataclass
class TargetPresetState:
    enabled: set[str] = field(default_factory=lambda: {"default"})
    explicit: set[str] = field(default_factory=set)
    provenance: tuple[str, ...] = ()
    exact: bool = False


def _ordered(values: set[str], order: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in order if value in values)


def resolve_presets(keys: set[str] | tuple[str, ...]) -> TargetCapabilities:
    architectures: set[str] = set()
    artifacts: set[str] = set()
    execution: set[str] = set()
    resolved: set[str] = set()
    active: set[str] = set()

    def add(key: str) -> None:
        if key in resolved:
            return
        if key in active:
            raise ValueError(f"Loom target preset cycle at {key}")
        preset = _PRESETS_BY_KEY.get(key)
        if preset is None:
            raise ValueError(f"unknown Loom target preset: {key}")
        active.add(key)
        for included_key in preset.includes:
            add(included_key)
        active.remove(key)
        architectures.update(preset.capabilities.architectures)
        artifacts.update(preset.capabilities.artifacts)
        execution.update(preset.capabilities.execution)
        resolved.add(key)

    for key in keys:
        add(key)
    return TargetCapabilities(
        _ordered(architectures, ARCHITECTURES),
        _ordered(artifacts, ARTIFACT_FORMATS),
        _ordered(execution, EXECUTION_SUBSTRATES),
    )


def _validate_presets() -> None:
    if len(_PRESETS_BY_KEY) != len(TARGET_PRESETS):
        raise ValueError("duplicate Loom target preset key")
    if len(_PRESETS_BY_OPTION) != len(TARGET_PRESETS):
        raise ValueError("duplicate Loom target preset option")
    if len(_PRESETS_BY_BAZEL_CONFIG) != len(TARGET_PRESETS):
        raise ValueError("duplicate Loom target preset Bazel config")
    for preset in TARGET_PRESETS:
        unknown_includes = set(preset.includes).difference(_PRESETS_BY_KEY)
        unknown_architectures = set(preset.capabilities.architectures).difference(
            ARCHITECTURES
        )
        unknown_artifacts = set(preset.capabilities.artifacts).difference(
            ARTIFACT_FORMATS
        )
        unknown_execution = set(preset.capabilities.execution).difference(
            EXECUTION_SUBSTRATES
        )
        if unknown_includes:
            raise ValueError(
                f"Loom target preset {preset.key} includes unknown presets: "
                + ", ".join(sorted(unknown_includes))
            )
        if unknown_architectures or unknown_artifacts or unknown_execution:
            raise ValueError(
                f"Loom target preset {preset.key} has unknown capabilities"
            )
        resolve_presets((preset.key,))


def _state(request: ConfigurationRequest) -> TargetPresetState:
    return cast(TargetPresetState, request.project_state[_STATE_KEY])


def initialize_request(request: ConfigurationRequest) -> None:
    request.project_state[_STATE_KEY] = TargetPresetState()


def apply_define(
    request: ConfigurationRequest,
    name: str,
    value: str,
    parse_bool: Callable[[str, str], bool],
) -> bool:
    preset = _PRESETS_BY_OPTION.get(name)
    if preset is not None:
        state = _state(request)
        state.explicit.add(preset.key)
        if parse_bool(name, value):
            state.enabled.add(preset.key)
        else:
            state.enabled.discard(preset.key)
        return True
    if name in ARCH_DEFINES:
        request.set_loom_arch(ARCH_DEFINES[name], parse_bool(name, value))
        return True
    if name in EMIT_DEFINES:
        request.set_loom_emit(EMIT_DEFINES[name], parse_bool(name, value))
        return True
    if name in EXECUTE_DEFINES:
        request.set_loom_execute(EXECUTE_DEFINES[name], parse_bool(name, value))
        return True
    return False


def apply_native_bazel_arg(request: ConfigurationRequest, arg: str) -> bool:
    prefix = "--config="
    if not arg.startswith(prefix):
        return False
    preset = _PRESETS_BY_BAZEL_CONFIG.get(arg[len(prefix) :])
    if preset is None:
        return False
    state = _state(request)
    state.explicit.add(preset.key)
    state.enabled.add(preset.key)
    return True


def finalize_request(request: ConfigurationRequest) -> None:
    state = _state(request)
    sources = (
        request.loom_arch_source,
        request.loom_emit_source,
        request.loom_execute_source,
    )
    native_count = sum(source == "native" for source in sources)
    if native_count not in (0, len(sources)):
        raise SystemExit(
            "Native Loom capability selection must name complete architecture, "
            "artifact, and execution lists together."
        )
    exact = native_count == len(sources)
    if exact and state.explicit:
        raise SystemExit(
            "Do not mix Loom target presets with native complete capability lists."
        )
    enabled = set() if exact else set(state.enabled)
    family_presets = enabled.difference(("default",))
    if family_presets:
        enabled = family_presets
    resolved = resolve_presets(enabled)
    request.enabled_loom_architectures.update(resolved.architectures)
    request.enabled_loom_artifact_formats.update(resolved.artifacts)
    request.enabled_loom_execute.update(resolved.execution)
    state.provenance = tuple(
        preset.key for preset in TARGET_PRESETS if preset.key in enabled
    )
    state.exact = exact


def request_summary(request: ConfigurationRequest) -> tuple[str, ...]:
    state = _state(request)
    provenance = state.provenance
    if provenance:
        selection = ",".join(provenance)
    elif state.exact:
        selection = "exact native lists"
    else:
        selection = "explicit leaves"
    return (
        "Loom target selection: " + selection,
        "Loom architectures: "
        + ",".join(_ordered(request.enabled_loom_architectures, ARCHITECTURES)),
        "Loom artifact formats: "
        + ",".join(_ordered(request.enabled_loom_artifact_formats, ARTIFACT_FORMATS)),
        "Loom execution substrates: "
        + ",".join(_ordered(request.enabled_loom_execute, EXECUTION_SUBSTRATES)),
    )


_validate_presets()
