# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE VM target-family record dialect."""

from typing import Any

from loom.assembly import (
    ARROW,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    AttrDict,
    FormatElement,
    FuncArgs,
    OptionalGroup,
    PredicateList,
    ResultTypeList,
    Scope,
    SymbolRef,
    TemplateParam,
    kw,
)
from loom.dialect.func.defs import CallingConv, Purity, Visibility
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ANY,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    Op,
    Operand,
    OpPhase,
    Result,
    SymbolDefinition,
    TargetLikeInterface,
)

ireevm_ops = Dialect(
    "ireevm",
    dialect_id=0x1A,
    doc="IREE VM target-family records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/ireevm/ops",
    register_by_default=False,
)

IreeVmTargetKind = EnumDef(
    "IreeVmTargetKind",
    [
        EnumCase("core", 1, doc="IREE VM core target row."),
    ],
    doc="IREE VM target row selected by ireevm.target.",
)

_IMPORT_DECL_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("import_module", "string"),
    AttrDef("import_symbol", "string"),
    AttrDef("cc", "enum", enum_def=CallingConv, optional=True),
    AttrDef("purity", "enum", enum_def=Purity, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_IMPORT_DECL_MODIFIER_FORMAT: list[FormatElement] = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    kw("module"),
    GLUE,
    LPAREN,
    Attr("import_module"),
    GLUE,
    RPAREN,
    kw("symbol"),
    GLUE,
    LPAREN,
    Attr("import_symbol"),
    GLUE,
    RPAREN,
    OptionalGroup([Attr("cc")], anchor="cc"),
    OptionalGroup([Attr("purity")], anchor="purity"),
]

_IMPORT_DECL_SIGNATURE_FORMAT: list[FormatElement] = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs("args"),
            OptionalGroup(
                [ARROW, ResultTypeList("results")],
                anchor="results",
            ),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_IMPORT_DECL_FUNC_LIKE: dict[str, Any] = dict(
    callee="callee",
    import_module="import_module",
    import_symbol="import_symbol",
    visibility="visibility",
    cc="cc",
    purity="purity",
    predicates="predicates",
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

ireevm_import_decl = Op(
    "ireevm.import.decl",
    group=ireevm_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "IREE VM imported function declaration. Callable by name via func.call "
        "and lowered to a VM module import during target materialization."
    ),
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    attrs=list(_IMPORT_DECL_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
        fact_domain="loom_func_symbol_fact_domain",
    ),
    results=[Result("results", ANY, variadic=True)],
    interfaces=[FuncLikeInterface(**_IMPORT_DECL_FUNC_LIKE, args_as_operands=True)],
    verify="loom_ireevm_import_decl_verify",
    format=[
        *_IMPORT_DECL_MODIFIER_FORMAT,
        *_IMPORT_DECL_SIGNATURE_FORMAT,
    ],
    examples=[
        'ireevm.import.decl module("hal") symbol("buffer.length") '
        "@hal_buffer_length(%buffer: i32) -> (i64)",
        'ireevm.import.decl public module("hal") symbol("buffer.map") '
        "@hal_buffer_map(%buffer: i32) -> (i64)",
    ],
)

ALL_IREEVM_OPS: tuple[Op, ...] = (
    ireevm_target,
    ireevm_import_decl,
)
