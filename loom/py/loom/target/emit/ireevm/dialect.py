# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE VM target-family record dialect."""

from loom.assembly import AttrDict, SymbolRef, TemplateParam
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    SYMBOL_DEFINE,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpPhase,
    SymbolDefinition,
    TargetLikeInterface,
)

ireevm_ops = Dialect(
    "ireevm",
    dialect_id=0x1A,
    doc="IREE VM target-family records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/emit/ireevm/ops",
    register_by_default=False,
)

IreeVmTargetKind = EnumDef(
    "IreeVmTargetKind",
    [
        EnumCase("core", 1, doc="IREE VM core target row."),
    ],
    doc="IREE VM target row selected by ireevm.target.",
)

ireevm_target = Op(
    "ireevm.target",
    group=ireevm_ops,
    doc=(
        "IREE VM target-family record. The selector chooses a VM emission row; "
        "optional attrs structurally override authored common target fields."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_ireevm_target_bundles",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(IreeVmTargetKind),
    verify="loom_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "ireevm.target<core> @vm",
    ],
)

ALL_IREEVM_OPS: tuple[Op, ...] = (ireevm_target,)
