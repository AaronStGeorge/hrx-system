# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-low legality policy for source operation kinds.

This module intentionally lives under ``loom.target`` instead of the dialect
declarations. The classification is a lowering-stage policy: target-low accepts
some ops directly, requires a target provider for others, and rejects source
structure that must lower away before executable code reaches this stage.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class TargetLowLegality(Enum):
    """Generated C class for target-low legality after provider verification."""

    UNSUPPORTED = "LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED"
    CORE = "LOOM_TARGET_LOW_LEGALITY_CORE"
    PROVIDER = "LOOM_TARGET_LOW_LEGALITY_PROVIDER"
    SOURCE_ONLY = "LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY"
    MODULE_METADATA = "LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA"


TARGET_LOW_CORE_OPS = (
    "buffer.alloca",
    "buffer.assume.memory_space",
    "buffer.view",
    "cfg.br",
    "cfg.cond_br",
    "encoding.assume.spec",
    "encoding.define",
    "encoding.layout.assume.dense",
    "encoding.layout.assume.strided",
    "encoding.layout.dense",
    "encoding.layout.strided",
    "func.call",
    "func.decl",
    "func.def",
    "func.return",
    "index.add",
    "index.cast",
    "index.cmp",
    "index.constant",
    "index.div",
    "index.madd",
    "index.mul",
    "index.rem",
    "index.sub",
    "kernel.workgroup.id",
    "kernel.workitem.id",
    "low.br",
    "low.cond_br",
    "low.concat",
    "low.const",
    "low.copy",
    "low.frame_index",
    "low.func.decl",
    "low.func.def",
    "low.invoke",
    "low.op",
    "low.reload",
    "low.resource",
    "low.return",
    "low.slot",
    "low.spill",
    "scalar.addf",
    "scalar.addi",
    "scalar.andi",
    "scalar.bitcast",
    "scalar.cmpf",
    "scalar.cmpi",
    "scalar.constant",
    "scalar.divf",
    "scalar.divsi",
    "scalar.divui",
    "scalar.extf",
    "scalar.extsi",
    "scalar.extui",
    "scalar.fmaf",
    "scalar.fmai",
    "scalar.fptosi",
    "scalar.fptoui",
    "scalar.fptrunc",
    "scalar.maxnumf",
    "scalar.minnumf",
    "scalar.mulf",
    "scalar.muli",
    "scalar.negf",
    "scalar.ori",
    "scalar.remf",
    "scalar.remsi",
    "scalar.remui",
    "scalar.shli",
    "scalar.shrsi",
    "scalar.shrui",
    "scalar.sitofp",
    "scalar.subf",
    "scalar.subi",
    "scalar.trunci",
    "scalar.uitofp",
    "scalar.xori",
    "scf.select",
    "vector.addf",
    "vector.addi",
    "vector.andi",
    "vector.bitcast",
    "vector.bitfield.extracts",
    "vector.bitfield.extractu",
    "vector.bitfield.insert",
    "vector.bitpack",
    "vector.bitunpacks",
    "vector.bitunpacku",
    "vector.cmpf",
    "vector.cmpi",
    "vector.constant",
    "vector.divf",
    "vector.divsi",
    "vector.divui",
    "vector.dot2f",
    "vector.dot4f8",
    "vector.dot4i",
    "vector.dot8i4",
    "vector.extf",
    "vector.extract",
    "vector.extsi",
    "vector.extui",
    "vector.fmaf",
    "vector.fmai",
    "vector.fptosi",
    "vector.fptoui",
    "vector.fptrunc",
    "vector.from_elements",
    "vector.insert",
    "vector.load",
    "vector.maxnumf",
    "vector.maxsi",
    "vector.maxui",
    "vector.minnumf",
    "vector.minsi",
    "vector.minui",
    "vector.mulf",
    "vector.muli",
    "vector.negf",
    "vector.ori",
    "vector.poison",
    "vector.reduce",
    "vector.remf",
    "vector.remsi",
    "vector.remui",
    "vector.select",
    "vector.shli",
    "vector.shrsi",
    "vector.shrui",
    "vector.shuffle",
    "vector.sitofp",
    "vector.splat",
    "vector.store",
    "vector.subf",
    "vector.subi",
    "vector.trunci",
    "vector.uitofp",
    "vector.xori",
    "view.load",
    "view.prefetch",
    "view.refine",
    "view.store",
    "view.subview",
)

TARGET_LOW_PROVIDER_OPS = (
    "kernel.async.cluster.gather",
    "kernel.async.cluster.gather.mask",
    "kernel.async.copy",
    "kernel.async.copy.mask",
    "kernel.async.gather",
    "kernel.async.gather.mask",
    "kernel.async.group",
    "kernel.async.tensor.load.to.lds",
    "kernel.async.tensor.store.from.lds",
    "kernel.async.wait",
    "kernel.barrier",
    "vector.dotf",
    "vector.iota",
    "vector.slice",
    "vector.table.lookup",
    "view.atomic.cmpxchg",
    "view.atomic.reduce",
    "view.atomic.rmw",
)

TARGET_LOW_SOURCE_ONLY_OPS = (
    "scf.condition",
    "scf.for",
    "scf.if",
    "scf.switch",
    "scf.while",
    "scf.yield",
)

TARGET_LOW_MODULE_METADATA_OPS = (
    "target.artifact",
    "target.profile",
)

TARGET_LOW_LEGALITY_OPS: tuple[tuple[TargetLowLegality, tuple[str, ...]], ...] = (
    (TargetLowLegality.CORE, TARGET_LOW_CORE_OPS),
    (TargetLowLegality.PROVIDER, TARGET_LOW_PROVIDER_OPS),
    (TargetLowLegality.SOURCE_ONLY, TARGET_LOW_SOURCE_ONLY_OPS),
    (TargetLowLegality.MODULE_METADATA, TARGET_LOW_MODULE_METADATA_OPS),
)


def target_low_legality_by_name() -> dict[str, TargetLowLegality]:
    """Returns the declared target-low legality class by op name."""

    result: dict[str, TargetLowLegality] = {}
    for legality, names in TARGET_LOW_LEGALITY_OPS:
        for name in names:
            if name in result:
                raise ValueError(f"op '{name}' has duplicate target-low legality")
            result[name] = legality
    return result
