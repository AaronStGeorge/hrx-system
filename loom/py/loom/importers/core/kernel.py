# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared construction for kernel-shaped Loom importer outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import loom
from loom.builder import ValueRef
from loom.importers.core.names import NameAllocator, sanitize_symbol
from loom.ir import BUFFER_TYPE, Block, Module, Region, Type


@dataclass(frozen=True, slots=True)
class KernelArgumentSpec:
    """One imported kernel ABI argument."""

    ordinal: int
    name: str
    type: Type = BUFFER_TYPE


@dataclass(frozen=True, slots=True)
class KernelModuleSpec:
    """Top-level target/profile/kernel facts shared by foreign importers."""

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
    body: Region
    arguments_by_ordinal: dict[int, ValueRef]
    target_symbol: str
    kernel_symbol: str

    @property
    def body_block(self) -> Block:
        return self.body.blocks[0]


def create_kernel_module(spec: KernelModuleSpec) -> KernelModuleShell:
    """Create a Loom module containing one target profile and one kernel.def."""
    module, builder = loom.module_builder()
    target_symbol = spec.target_symbol or sanitize_symbol(
        spec.target_preset, fallback="target"
    )
    builder.target.profile(symbol=target_symbol, preset=spec.target_preset)
    names = NameAllocator()
    arguments_by_ordinal = {
        argument.ordinal: builder.value(
            names.reserve_or_fresh(argument.name, fallback="arg"),
            argument.type,
        )
        for argument in spec.arguments
    }
    body = builder.region()
    workgroup = normalize_workgroup_size(spec.workgroup_size)
    builder.kernel.def_(
        target=target_symbol,
        export_symbol=spec.export_symbol,
        export_ordinal=spec.export_ordinal,
        workgroup_size_x=workgroup[0],
        workgroup_size_y=workgroup[1],
        workgroup_size_z=workgroup[2],
        callee=spec.callee,
        args=[arguments_by_ordinal[argument.ordinal] for argument in spec.arguments],
        body=body,
    )
    return KernelModuleShell(
        module=module,
        builder=builder,
        body=body,
        arguments_by_ordinal=arguments_by_ordinal,
        target_symbol=target_symbol,
        kernel_symbol=spec.callee,
    )


def normalize_workgroup_size(workgroup_size: tuple[int, ...]) -> tuple[int, int, int]:
    """Normalize source workgroup size facts to Loom's x/y/z tuple."""
    if len(workgroup_size) > 3:
        raise ValueError(f"unsupported workgroup rank: {workgroup_size}")
    values = list(workgroup_size)
    while len(values) < 3:
        values.append(1)
    return (int(values[0]), int(values[1]), int(values[2]))
