# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared construction for kernel-shaped Loom importer outputs."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass

import loom
from loom.builder import ValueRef
from loom.dsl import Op
from loom.importers.core.names import NameAllocator, sanitize_symbol
from loom.ir import BUFFER_TYPE, INDEX, Block, Module, Region, Type


@dataclass(frozen=True, slots=True)
class KernelArgumentSpec:
    """One imported kernel ABI argument."""

    ordinal: int
    name: str
    type: Type = BUFFER_TYPE


@dataclass(frozen=True, slots=True)
class KernelModuleSpec:
    """Top-level target/kernel facts shared by foreign importers."""

    target_preset: str
    export_symbol: str
    callee: str
    arguments: Sequence[KernelArgumentSpec]
    target_symbol: str | None = None
    export_ordinal: int | None = None
    workgroup_size: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True, slots=True)
class KernelModuleShell:
    """Partially-built Loom module plus handles for filling the kernel body."""

    module: Module
    builder: loom.LoomBuilder
    config: Region
    body: Region
    arguments_by_ordinal: dict[int, ValueRef]
    target_symbol: str
    kernel_symbol: str

    @property
    def body_block(self) -> Block:
        return self.body.blocks[0]

    @property
    def config_block(self) -> Block:
        return self.config.blocks[0]


def create_kernel_module(spec: KernelModuleSpec) -> KernelModuleShell:
    """Create a Loom module containing one target record and one kernel.def."""
    module, builder = loom.module_builder(ops=kernel_module_ops(spec.target_preset))
    target_symbol = spec.target_symbol or sanitize_symbol(
        spec.target_preset, fallback="target"
    )
    _build_target_record(builder, target_symbol, spec.target_preset)
    names = NameAllocator()
    arguments_by_ordinal = {
        argument.ordinal: builder.value(
            names.reserve_or_fresh(argument.name, fallback="arg"),
            argument.type,
        )
        for argument in spec.arguments
    }
    workgroup = normalize_workgroup_size(spec.workgroup_size)
    config = builder.region()
    body = builder.region()
    builder.kernel.def_(
        target=target_symbol,
        export_symbol=spec.export_symbol,
        export_ordinal=spec.export_ordinal,
        callee=spec.callee,
        args=[arguments_by_ordinal[argument.ordinal] for argument in spec.arguments],
        config=config,
        body=body,
    )
    _build_launch_config(builder, config, workgroup)
    return KernelModuleShell(
        module=module,
        builder=builder,
        config=config,
        body=body,
        arguments_by_ordinal=arguments_by_ordinal,
        target_symbol=target_symbol,
        kernel_symbol=spec.callee,
    )


def _build_launch_config(
    builder: loom.LoomBuilder,
    config: Region,
    workgroup_size: tuple[int, int, int],
) -> None:
    with builder.insertion_block(config.blocks[0]):
        count_x = builder.index.constant(value=1, results=[INDEX], name="wg_count_x")
        count_y = builder.index.constant(value=1, results=[INDEX], name="wg_count_y")
        count_z = builder.index.constant(value=1, results=[INDEX], name="wg_count_z")
        size_x = builder.index.constant(
            value=workgroup_size[0],
            results=[INDEX],
            name="wg_size_x",
        )
        size_y = builder.index.constant(
            value=workgroup_size[1],
            results=[INDEX],
            name="wg_size_y",
        )
        size_z = builder.index.constant(
            value=workgroup_size[2],
            results=[INDEX],
            name="wg_size_z",
        )
        builder.kernel.config(
            workgroup_count_x=count_x,
            workgroup_count_y=count_y,
            workgroup_count_z=count_z,
            workgroup_size_x=size_x,
            workgroup_size_y=size_y,
            workgroup_size_z=size_z,
        )


def normalize_workgroup_size(workgroup_size: tuple[int, ...]) -> tuple[int, int, int]:
    """Normalize source workgroup size facts to Loom's x/y/z tuple."""
    if len(workgroup_size) > 3:
        raise ValueError(f"unsupported workgroup rank: {workgroup_size}")
    values = list(workgroup_size)
    while len(values) < 3:
        values.append(1)
    return (int(values[0]), int(values[1]), int(values[2]))


def kernel_module_ops(target_preset: str) -> tuple[Op, ...]:
    """Return the op declarations needed for a kernel module target preset."""
    ops = loom.default_ops()
    if _amdgpu_target_kind(target_preset) is None:
        return ops
    from loom.target.arch.amdgpu.dialect import ALL_AMDGPU_OPS

    return (*ops, *ALL_AMDGPU_OPS)


def _build_target_record(
    builder: loom.LoomBuilder,
    target_symbol: str,
    target_preset: str,
) -> None:
    amdgpu_kind = _amdgpu_target_kind(target_preset)
    if amdgpu_kind is not None:
        from loom.target.arch.amdgpu.dialect import amdgpu_target

        builder.ir.build(
            amdgpu_target.name,
            attributes={"symbol": target_symbol, "kind": amdgpu_kind},
        )
        return
    builder.target.generic(symbol=target_symbol, kind="reference")


def _amdgpu_target_kind(target_preset: str) -> str | None:
    target_cpu = _target_cpu(target_preset)
    if target_cpu is None:
        return None
    from loom.target.arch.amdgpu.dialect import AmdgpuTargetKind

    available_kinds = {case.keyword for case in AmdgpuTargetKind.cases}
    if target_cpu in available_kinds:
        return target_cpu
    if target_cpu.startswith("gfx11") and "gfx1100" in available_kinds:
        return "gfx1100"
    if target_cpu.startswith("gfx12") and "gfx1200" in available_kinds:
        return "gfx1200"
    return None


def _target_cpu(target_preset: str) -> str | None:
    try:
        tokens = shlex.split(target_preset)
    except ValueError:
        tokens = target_preset.split()
    for index, token in enumerate(tokens):
        if token.startswith("-mcpu="):
            return token.split("=", 1)[1]
        if token.startswith("--mcpu="):
            return token.split("=", 1)[1]
        if token in ("-mcpu", "--mcpu") and index + 1 < len(tokens):
            return tokens[index + 1]
    return None
