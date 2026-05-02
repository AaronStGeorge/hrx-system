# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Op declarations -> C op tables, accessors, and builders.

Reads Op declarations from the Python DSL and emits three C files per
dialect:

  ops.h      — enum + ISA macros + accessor macros + builder declarations
  builders.c — builder implementations (macros for common, explicit for complex)
  tables.c   — .rodata: B-string names, format arrays, descriptors, vtables

The generated files are checked into the repository. The C build never
requires Python.

Usage:
    python3 loom/py/loom/gen/run.py c_tables
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from loom.assembly import (
    Attr,
    AttrDict,
    AttrTable,
    BindingList,
    BlockArgs,
    BlockRef,
    Clause,
    DescriptorRef,
    Flags,
    FormatElement,
    FuncArgs,
    Glue,
    IndexList,
    Keyword,
    OperandDict,
    OpRef,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    RegionTable,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TemplateParam,
    TypedRefs,
    TypeOf,
    TypesOf,
)
from loom.assembly import (
    Region as RegionFmt,
)
from loom.dsl import (
    ATTR_TYPE_FLAGS,
    AttrDef,
    CallLikeInterface,
    CallLikeKind,
    ContractFamily,
    EffectKind,
    EnumDef,
    FuncLikeInterface,
    LoopLikeInterface,
    MemoryAccessInterface,
    Op,
    Operand,
    RegionBranchInterface,
    RegionDef,
    TargetLikeInterface,
    TiedResult,
    TypeConstraint,
    TypeDef,
    TypeSemantic,
)
from loom.fields import FieldKind, FieldLayout, compute_layout
from loom.gen import bootstrap as _bootstrap
from loom.gen.generated_file import line_comment_header

# ============================================================================
# Constants
# ============================================================================

# Maps Python Keyword text to C keyword enum name.
# This is the single source of truth for all format keywords. The
# generator produces keyword_enum.inc and keyword_table.inc from
# this dict. Ordinals are assigned by position (do not reorder
# existing entries — append new keywords at the end).
KEYWORD_MAP: dict[str, str] = {
    ",": "LOOM_KW_COMMA",
    ":": "LOOM_KW_COLON",
    "->": "LOOM_KW_ARROW",
    "=": "LOOM_KW_EQUALS",
    "(": "LOOM_KW_LPAREN",
    ")": "LOOM_KW_RPAREN",
    "[": "LOOM_KW_LBRACKET",
    "]": "LOOM_KW_RBRACKET",
    "{": "LOOM_KW_LBRACE",
    "}": "LOOM_KW_RBRACE",
    "to": "LOOM_KW_TO",
    "step": "LOOM_KW_STEP",
    "else": "LOOM_KW_ELSE",
    "iter_args": "LOOM_KW_ITER_ARGS",
    "where": "LOOM_KW_WHERE",
    "as": "LOOM_KW_AS",
    "public": "LOOM_KW_PUBLIC",
    "host": "LOOM_KW_HOST",
    "device": "LOOM_KW_DEVICE",
    "priority": "LOOM_KW_PRIORITY",
    "x": "LOOM_KW_X",
    "import": "LOOM_KW_IMPORT",
    "layout": "LOOM_KW_LAYOUT",
    "into": "LOOM_KW_INTO",
    "default": "LOOM_KW_DEFAULT",
    "case": "LOOM_KW_CASE",
    "do": "LOOM_KW_DO",
    "using": "LOOM_KW_USING",
    "dgroups": "LOOM_KW_DGROUPS",
    "target": "LOOM_KW_TARGET",
    "allocation": "LOOM_KW_ALLOCATION",
    "schedule": "LOOM_KW_SCHEDULE",
    "source": "LOOM_KW_SOURCE",
    "preset": "LOOM_KW_PRESET",
    "abi": "LOOM_KW_ABI",
    "export": "LOOM_KW_EXPORT",
    "for": "LOOM_KW_FOR",
    "value": "LOOM_KW_VALUE",
    "seed": "LOOM_KW_SEED",
    "range": "LOOM_KW_RANGE",
    "path": "LOOM_KW_PATH",
    "actual": "LOOM_KW_ACTUAL",
    "expected": "LOOM_KW_EXPECTED",
    "atol": "LOOM_KW_ATOL",
    "attrs": "LOOM_KW_ATTRS",
    "base": "LOOM_KW_BASE",
    "bounds": "LOOM_KW_BOUNDS",
    "callee": "LOOM_KW_CALLEE",
    "count": "LOOM_KW_COUNT",
    "inputs": "LOOM_KW_INPUTS",
    "mode": "LOOM_KW_MODE",
    "nan": "LOOM_KW_NAN",
    "offset": "LOOM_KW_OFFSET",
    "reason": "LOOM_KW_REASON",
    "rtol": "LOOM_KW_RTOL",
    "shape": "LOOM_KW_SHAPE",
    "values": "LOOM_KW_VALUES",
    "artifact": "LOOM_KW_ARTIFACT",
    "ordinal": "LOOM_KW_ORDINAL",
    "linkage": "LOOM_KW_LINKAGE",
    "workgroup_size": "LOOM_KW_WORKGROUP_SIZE",
    "from": "LOOM_KW_FROM",
    "axes": "LOOM_KW_AXES",
}

# Maps Region(..., syntax=...) names to C parser/printer selector IDs. The
# empty name is the canonical braced region form.
REGION_SYNTAX_MAP: dict[str, str] = {
    "": "LOOM_REGION_SYNTAX_DEFAULT",
    "test.do": "LOOM_REGION_SYNTAX_TEST_DO",
    "low.asm": "LOOM_REGION_SYNTAX_LOW_ASM",
    "low.asm.optional": "LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL",
    "pipeline": "LOOM_REGION_SYNTAX_PIPELINE",
}

# Maps Python TypeConstraint enum to C constraint enum name.
TYPE_CONSTRAINT_MAP: dict[TypeConstraint, str] = {
    TypeConstraint.TILE: "LOOM_TYPE_CONSTRAINT_TILE",
    TypeConstraint.TENSOR: "LOOM_TYPE_CONSTRAINT_TENSOR",
    TypeConstraint.VECTOR: "LOOM_TYPE_CONSTRAINT_VECTOR",
    TypeConstraint.RANK_ONE_VECTOR: "LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR",
    TypeConstraint.ALL_STATIC_VECTOR: "LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR",
    TypeConstraint.ALL_STATIC_RANK_ONE_VECTOR: "LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR",
    TypeConstraint.VIEW: "LOOM_TYPE_CONSTRAINT_VIEW",
    TypeConstraint.BUFFER: "LOOM_TYPE_CONSTRAINT_BUFFER",
    TypeConstraint.INTEGER: "LOOM_TYPE_CONSTRAINT_INTEGER",
    TypeConstraint.FLOAT: "LOOM_TYPE_CONSTRAINT_FLOAT",
    TypeConstraint.INDEX_OR_NON_I1_INTEGER_SCALAR: "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR",
    TypeConstraint.INTEGER_ELEMENT: "LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT",
    TypeConstraint.FLOAT_ELEMENT: "LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT",
    TypeConstraint.INDEX_OR_NON_I1_INTEGER_ELEMENT: "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT",
    TypeConstraint.I1_ELEMENT: "LOOM_TYPE_CONSTRAINT_I1_ELEMENT",
    TypeConstraint.I8_ELEMENT: "LOOM_TYPE_CONSTRAINT_I8_ELEMENT",
    TypeConstraint.I32_ELEMENT: "LOOM_TYPE_CONSTRAINT_I32_ELEMENT",
    TypeConstraint.F16_OR_BF16_ELEMENT: "LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT",
    TypeConstraint.F32_ELEMENT: "LOOM_TYPE_CONSTRAINT_F32_ELEMENT",
    TypeConstraint.SCALAR: "LOOM_TYPE_CONSTRAINT_SCALAR",
    TypeConstraint.INDEX: "LOOM_TYPE_CONSTRAINT_INDEX",
    TypeConstraint.OFFSET: "LOOM_TYPE_CONSTRAINT_OFFSET",
    TypeConstraint.ADDRESS: "LOOM_TYPE_CONSTRAINT_ADDRESS",
    TypeConstraint.ANY: "LOOM_TYPE_CONSTRAINT_ANY",
    TypeConstraint.GROUP: "LOOM_TYPE_CONSTRAINT_GROUP",
    TypeConstraint.ANY_ENCODING: "LOOM_TYPE_CONSTRAINT_ANY_ENCODING",
    TypeConstraint.ENCODING_LAYOUT: "LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT",
    TypeConstraint.ENCODING_SCHEMA: "LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA",
    TypeConstraint.ENCODING_STORAGE: "LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE",
    TypeConstraint.ENCODING_TRANSFORM: "LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM",
    TypeConstraint.POOL: "LOOM_TYPE_CONSTRAINT_POOL",
    TypeConstraint.REGISTER: "LOOM_TYPE_CONSTRAINT_REGISTER",
    TypeConstraint.STORAGE: "LOOM_TYPE_CONSTRAINT_STORAGE",
    TypeConstraint.I1: "LOOM_TYPE_CONSTRAINT_I1",
}

CALL_LIKE_KIND_MAP: dict[CallLikeKind, str] = {
    CallLikeKind.SEMANTIC: "LOOM_CALL_LIKE_KIND_SEMANTIC",
    CallLikeKind.TEMPLATE: "LOOM_CALL_LIKE_KIND_TEMPLATE",
    CallLikeKind.LOW_INTERNAL: "LOOM_CALL_LIKE_KIND_LOW_INTERNAL",
    CallLikeKind.LOW_INVOKE: "LOOM_CALL_LIKE_KIND_LOW_INVOKE",
}

_ERROR_REF_CODE_BITS = 10
_ERROR_REF_MAX_CODE = (1 << _ERROR_REF_CODE_BITS) - 1
_ERROR_REF_MAX_DOMAIN_VALUE = (1 << (16 - _ERROR_REF_CODE_BITS)) - 2


def _error_ref_literal(error: Any) -> str:
    if error.code > _ERROR_REF_MAX_CODE:
        raise ValueError(f"{error.error_id}: code {error.code} exceeds LOOM_ERROR_REF 10-bit max")
    if error.domain.value > _ERROR_REF_MAX_DOMAIN_VALUE:
        raise ValueError(f"{error.error_id}: domain {error.domain.name} exceeds LOOM_ERROR_REF 6-bit max")
    return f"LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_{error.domain.name}, {error.code})"


# Maps Python trait names to C trait bit names.
TRAIT_MAP: dict[str, str] = {
    "Pure": "LOOM_TRAIT_PURE",
    "Commutative": "LOOM_TRAIT_COMMUTATIVE",
    "Idempotent": "LOOM_TRAIT_IDEMPOTENT",
    "Involution": "LOOM_TRAIT_INVOLUTION",
    "Terminator": "LOOM_TRAIT_TERMINATOR",
    "ConstantLike": "LOOM_TRAIT_CONSTANT_LIKE",
    "Elementwise": "LOOM_TRAIT_ELEMENTWISE",
    "Decomposable": "LOOM_TRAIT_DECOMPOSABLE",
    "SymbolDefine": "LOOM_TRAIT_SYMBOL_DEFINE",
    "IsolatedFromAbove": "LOOM_TRAIT_ISOLATED_FROM_ABOVE",
    "NonDeterministic": "LOOM_TRAIT_NON_DETERMINISTIC",
    "UnknownEffects": "LOOM_TRAIT_UNKNOWN_EFFECTS",
    "Convergent": "LOOM_TRAIT_CONVERGENT",
    "UniqueIdentity": "LOOM_TRAIT_UNIQUE_IDENTITY",
    "Hint": "LOOM_TRAIT_HINT",
    "SafeToSpeculate": "LOOM_TRAIT_SAFE_TO_SPECULATE",
    "RefinableResultTypeRefs": "LOOM_TRAIT_REFINABLE_RESULT_TYPE_REFS",
    "PoisonBoundary": "LOOM_TRAIT_POISON_BOUNDARY",
    "FactIdentity": "LOOM_TRAIT_FACT_IDENTITY",
    "ValueAlias": "LOOM_TRAIT_VALUE_ALIAS",
}

# Maps Python constraint names to (relation, property) C enum pairs.
CONSTRAINT_MAP: dict[str, tuple[str, str]] = {
    "SameType": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_TYPE"),
    "SameKind": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_KIND"),
    "SameRegisterClass": (
        "LOOM_RELATION_PAIRWISE_EQ",
        "LOOM_PROPERTY_REGISTER_CLASS",
    ),
    "SameElementType": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_ELEMENT_TYPE"),
    "SameEncoding": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_ENCODING"),
    "SameShape": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_SHAPE"),
    "RanksMatch": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_RANK"),
    "HasIntegerElement": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT",
    ),
    "HasFloatElement": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT",
    ),
    "HasIndexOrNonI1IntegerScalar": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR",
    ),
    "HasIndexOrNonI1IntegerElement": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT",
    ),
    "HasI1Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_I1_ELEMENT",
    ),
    "HasI8Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_I8_ELEMENT",
    ),
    "HasI32Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_I32_ELEMENT",
    ),
    "HasF16OrBf16Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT",
    ),
    "HasF32Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_F32_ELEMENT",
    ),
    "HasRegister": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_REGISTER",
    ),
    "HasRankOneVector": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR",
    ),
    "HasAllStaticVector": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR",
    ),
    "HasAllStaticRankOneVector": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR",
    ),
    "ElementWidthGreaterThan": (
        "LOOM_RELATION_ELEMENT_WIDTH_ORDER",
        "LOOM_PROPERTY_ELEMENT_WIDTH_GREATER_THAN",
    ),
    "ElementWidthLessThan": (
        "LOOM_RELATION_ELEMENT_WIDTH_ORDER",
        "LOOM_PROPERTY_ELEMENT_WIDTH_LESS_THAN",
    ),
    "ElementWidthAtLeastAttr": (
        "LOOM_RELATION_ELEMENT_WIDTH_AT_LEAST_ATTR",
        "LOOM_PROPERTY_ELEMENT_WIDTH_AT_LEAST_ATTR",
    ),
    "BitRangeWithinElementWidth": (
        "LOOM_RELATION_BIT_RANGE_WITHIN_ELEMENT_WIDTH",
        "LOOM_PROPERTY_BIT_RANGE_WITHIN_ELEMENT_WIDTH",
    ),
    "PositiveBitWidthAttr": (
        "LOOM_RELATION_ATTR_I64_PREDICATE",
        "LOOM_PROPERTY_BIT_WIDTH_POSITIVE",
    ),
    "AttrMatchesElementType": (
        "LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "LiteralMatchesElementType": (
        "LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "RegisterUnitsSumTo": (
        "LOOM_RELATION_REGISTER_UNIT_COUNT_SUM",
        "LOOM_PROPERTY_REGISTER_UNIT_COUNT",
    ),
    "TotalBitCountEqual": (
        "LOOM_RELATION_TOTAL_BIT_COUNT_EQUAL",
        "LOOM_PROPERTY_TOTAL_BIT_COUNT",
    ),
    "PackedPayloadBitCountMatchesStorage": (
        "LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
        "LOOM_PROPERTY_PACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
    ),
    "UnpackedPayloadBitCountMatchesStorage": (
        "LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
        "LOOM_PROPERTY_UNPACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
    ),
    "OffsetCountMatchesRank": (
        "LOOM_RELATION_COUNT_MATCHES_RANK",
        "LOOM_PROPERTY_RANK",
    ),
    "ValueCountMatchesStaticElementCount": (
        "LOOM_RELATION_COUNT_MATCHES_STATIC_ELEMENT_COUNT",
        "LOOM_PROPERTY_TYPE",
    ),
    "DimIndexInBounds": ("LOOM_RELATION_ATTR_IN_RANGE_RANK", "LOOM_PROPERTY_RANK"),
    "AllShapesMatch": ("LOOM_RELATION_ALL_SAME", "LOOM_PROPERTY_SHAPE"),
    "LastAxisGroupedBy": ("LOOM_RELATION_LAST_AXIS_GROUPED_BY", "$data"),
    "BlockArgCount": ("LOOM_RELATION_REGION_ARG_COUNT", "LOOM_PROPERTY_TYPE"),
    "BlockArgsSatisfy": (
        "LOOM_RELATION_REGION_ARGS_SATISFY",
        "$type_constraint_data",
    ),
    "BlockArgsMatchTypes": (
        "LOOM_RELATION_REGION_ARG_MATCH",
        "LOOM_PROPERTY_TYPE",
    ),
    "BlockArgsMatchElementTypes": (
        "LOOM_RELATION_REGION_ARG_MATCH",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "YieldCountMatchesResults": ("LOOM_RELATION_YIELD_COUNT", "LOOM_PROPERTY_TYPE"),
    "YieldTypesMatchResults": ("LOOM_RELATION_YIELD_MATCH", "LOOM_PROPERTY_TYPE"),
    "YieldElementTypesMatchResults": (
        "LOOM_RELATION_YIELD_MATCH",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "IterArgsMatchResults": (
        "LOOM_RELATION_VARIADIC_MATCH",
        "LOOM_PROPERTY_TYPE",
    ),
}

# Maps FieldKind to C field category value.
FIELD_CATEGORY_MAP: dict[int, int] = {
    FieldKind.OPERAND: 0,  # LOOM_FIELD_OPERAND
    FieldKind.RESULT: 1,  # LOOM_FIELD_RESULT
    FieldKind.ATTR: 2,  # LOOM_FIELD_ATTR
    FieldKind.REGION: 3,  # LOOM_FIELD_REGION
}

# LOOM_FIELD_REF stores a 6-bit index in the generated constraint-table
# encoding. Printer callbacks use a wider representation, but constraint args
# still need a hard generator-side guard against silent truncation.
LOOM_FIELD_REF_MAX_INDEX = 63

# Maps Python attr_type strings to C attr kind enum names.
ATTR_KIND_MAP: dict[str, str] = {
    "i64": "LOOM_ATTR_I64",
    "f64": "LOOM_ATTR_F64",
    "string": "LOOM_ATTR_STRING",
    "bool": "LOOM_ATTR_BOOL",
    "enum": "LOOM_ATTR_ENUM",
    "i64_array": "LOOM_ATTR_I64_ARRAY",
    "symbol": "LOOM_ATTR_SYMBOL",
    "type": "LOOM_ATTR_TYPE",
    "encoding": "LOOM_ATTR_ENCODING",
    "predicate_list": "LOOM_ATTR_PREDICATE_LIST",
    "dict": "LOOM_ATTR_DICT",
    "any": "LOOM_ATTR_ANY",
}

COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

GENERATED_HEADER = COPYRIGHT + "\n" + "\n".join(line_comment_header("//", generator="loom.gen.c_tables")) + "\n// clang-format off"


# ============================================================================
# Naming helpers
# ============================================================================


def _c_prefix(op: Op) -> str:
    """Returns the C function/variable prefix for an op.

    test.addi -> loom_test_addi
    """
    return "loom_" + op.name.replace(".", "_")


def _c_identifier(value: str) -> str:
    """Returns a C identifier fragment for a DSL key."""
    return re.sub(r"[^0-9A-Za-z_]", "_", value)


def _c_enum_name(op: Op) -> str:
    """Returns the C enum constant name for an op kind.

    test.addi -> LOOM_OP_TEST_ADDI
    """
    return "LOOM_OP_" + op.name.replace(".", "_").upper()


def _c_dialect_enum(dialect_name: str) -> str:
    """Returns the C dialect ID enum name.

    test -> LOOM_DIALECT_TEST
    """
    return "LOOM_DIALECT_" + dialect_name.upper()


def _c_dialect_path(dialect: Any) -> str:
    """Returns the generated C source path under loom/src/loom."""
    return dialect.c_path or f"ops/{dialect.name}"


def _c_dialect_include_path(dialect: Any) -> str:
    """Returns the generated C include path rooted at loom/src."""
    return f"loom/{_c_dialect_path(dialect)}"


def _guard_name(dialect_name: str) -> str:
    """Returns the header guard name."""
    return f"LOOM_OPS_{dialect_name.upper()}_OPS_H_"


def _contract_family_mask(contracts: Sequence[ContractFamily]) -> str:
    """Returns a stable C bitmask expression for contract families."""

    unique_contracts = set(contracts)
    if len(unique_contracts) != len(contracts):
        duplicate_names = sorted(family.name for family in contracts if contracts.count(family) > 1)
        raise ValueError(f"duplicate contract families in semantic metadata: {duplicate_names}")
    ordered_contracts = [family for family in ContractFamily if family in unique_contracts]
    if not ordered_contracts:
        return "0"
    return " | ".join(family.c_name for family in ordered_contracts)


def _op_phase_c_name(op: Op) -> str:
    """Returns the C phase enum for an op after applying its dialect default."""

    phase = op.effective_phase
    if phase is None:
        return "LOOM_OP_PHASE_UNSPECIFIED"
    return phase.c_name


def _type_semantic_c_name(semantic: TypeSemantic) -> str:
    """Returns the C enum for a type semantic role."""

    return semantic.c_name


def _emit_op_semantics(lines: list[str], op: Op) -> None:
    """Appends a sparse initializer for one op semantic metadata row."""

    contract_families = _contract_family_mask(op.contracts)
    lines.append("    {")
    lines.append(f"        .phase = {_op_phase_c_name(op)},")
    if contract_families != "0":
        lines.append(f"        .contract_families = {contract_families},")
    lines.append("    },")


def _emit_type_semantics(lines: list[str], type_def: TypeDef) -> None:
    """Appends a sparse initializer for one type semantic metadata row."""

    semantic = _type_semantic_c_name(type_def.semantic)
    contract_families = _contract_family_mask(type_def.contracts)
    if semantic == "LOOM_TYPE_SEMANTIC_ORDINARY" and contract_families == "0":
        return
    lines.append("    .semantics = {")
    if semantic != "LOOM_TYPE_SEMANTIC_ORDINARY":
        lines.append(f"        .semantic = {semantic},")
    if contract_families != "0":
        lines.append(f"        .contract_families = {contract_families},")
    lines.append("    },")


_T = TypeVar("_T")


def _find_interface(op: Op, iface_class: type[_T]) -> _T | None:
    """Returns the interface instance from op.interfaces matching |iface_class|, or None.

    Generic over the interface class so callers retain type information
    on the result. Used by both the per-op interface helpers below and
    the table-driven interface vtable emitter (see _INTERFACES).
    """
    for iface in op.interfaces:
        if isinstance(iface, iface_class):
            return iface
    return None


def _func_args_are_operands(op: Op) -> bool:
    """Returns true if FuncArgs should be stored in the op's operand array.

    Bodyless func-like ops (func.decl, func.ukernel) have no entry block, so
    their signature args live as op operands. Bodyful funcs keep signature args
    as body entry block args and must not duplicate them as operands.
    """
    func_like_iface = _find_interface(op, FuncLikeInterface)
    return bool(func_like_iface and func_like_iface.args_as_operands)


def _func_args_field_name(op: Op) -> str:
    """Returns the FuncArgs field name declared by a func-like op format."""

    def _walk(elements: Sequence[FormatElement]) -> str | None:
        for element in elements:
            match element:
                case FuncArgs(field=name):
                    return name
                case OptionalGroup(elements=inner) | Scope(elements=inner) | Clause(elements=inner):
                    nested = _walk(inner)
                    if nested is not None:
                        return nested
                case _:
                    continue
        return None

    return _walk(op.format) or "args"


def _explicit_func_args_operand(op: Op) -> Operand | None:
    """Returns the operand descriptor that backs FuncArgs, if declared."""
    if not _func_args_are_operands(op):
        return None
    field_name = _func_args_field_name(op)
    for operand in op.operands:
        if operand.name != field_name:
            continue
        if not operand.variadic:
            raise ValueError(f"Op '{op.name}': FuncArgs field '{field_name}' must be a variadic operand when declared explicitly")
        return operand
    return None


# Maps Python symbol interface names to C interface flag constants.
SYMBOL_INTERFACE_MAP: dict[str, str] = {
    "func_like": "LOOM_SYMBOL_INTERFACE_FUNC_LIKE",
    "global": "LOOM_SYMBOL_INTERFACE_GLOBAL",
    "executable": "LOOM_SYMBOL_INTERFACE_EXECUTABLE",
    "record": "LOOM_SYMBOL_INTERFACE_RECORD",
    "target": "LOOM_SYMBOL_INTERFACE_TARGET",
}


def _symbol_interface_flags(interfaces: Sequence[str]) -> str:
    flags = [SYMBOL_INTERFACE_MAP[interface] for interface in interfaces]
    return " | ".join(flags) if flags else "0"


def _symbol_kind(op: Op) -> str:
    """Returns the legacy C bytecode symbol kind constant for an op."""
    return op.symbol_def.bytecode_kind if op.symbol_def is not None else "LOOM_SYMBOL_NONE"


def _constraint_arg_ref(
    op: Op,
    constraint_name: str,
    arg_name: str,
    category: int,
    field_index: int,
) -> str:
    """Returns the LOOM_FIELD_REF(...) initializer for one constraint arg."""
    if field_index > LOOM_FIELD_REF_MAX_INDEX:
        raise ValueError(f"Op '{op.name}' constraint {constraint_name}: field '{arg_name}' index {field_index} exceeds LOOM_FIELD_REF 6-bit max {LOOM_FIELD_REF_MAX_INDEX}")
    return f"LOOM_FIELD_REF({category}, {field_index})"


def _resolve_attr_index(op: Op, attr_name: str | None, interface_name: str = "interface") -> int:
    """Resolves an attr name to its non-flags attr index.

    Returns 0xFF (LOOM_ATTR_INDEX_NONE) if attr_name is None.
    Raises ValueError if the attr name is not found on the op.
    |interface_name| is used in error messages to identify which
    interface declaration is referencing this attr.
    """
    if attr_name is None:
        return 0xFF
    index = 0
    for attr_def in op.attrs:
        if attr_def.attr_type == ATTR_TYPE_FLAGS:
            continue
        if attr_def.name == attr_name:
            return index
        index += 1
    raise ValueError(f"{interface_name} on {op.name!r}: attr {attr_name!r} not found. Available: {[a.name for a in op.attrs if a.attr_type != ATTR_TYPE_FLAGS]}")


def _resolve_region_index(op: Op, region_name: str | None, interface_name: str = "interface") -> int:
    """Resolves a region name to its region index.

    Returns 0xFF (LOOM_REGION_INDEX_NONE) if region_name is None.
    Raises ValueError if the region name is not found on the op.
    |interface_name| is used in error messages.
    """
    if region_name is None:
        return 0xFF
    for i, region_def in enumerate(op.regions):
        if region_def.name == region_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: region {region_name!r} not found. Available: {[r.name for r in op.regions]}")


def _resolve_operand_index(op: Op, operand_name: str | None, interface_name: str = "interface") -> int:
    """Resolves an operand name to its index in the op's operand list.

    Returns 0xFF (LOOM_OPERAND_INDEX_NONE) if operand_name is None.
    Raises ValueError if the operand name is not found on the op.
    |interface_name| is used in error messages.

    The returned index is the position in op.operands. This includes
    fixed operands and the variadic operand (which always comes last
    when present). For interfaces that need the offset of a variadic
    operand specifically, the returned index is exactly that offset.
    """
    if operand_name is None:
        return 0xFF
    for i, operand in enumerate(op.operands):
        if operand.name == operand_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: operand {operand_name!r} not found. Available: {[o.name for o in op.operands]}")


def _resolve_result_index(op: Op, result_name: str | None, interface_name: str = "interface") -> int:
    """Resolves a result name to its index in the op's result list.

    Returns 0xFF (LOOM_RESULT_INDEX_NONE) if result_name is None.
    Raises ValueError if the result name is not found on the op.
    |interface_name| is used in error messages.

    The returned index is the position in op.results. This includes
    fixed results and the variadic result tail.
    """
    if result_name is None:
        return 0xFF
    for i, result in enumerate(op.results):
        if result.name == result_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: result {result_name!r} not found. Available: {[r.name for r in op.results]}")


def _resolve_block_arg_index(
    op: Op,
    region_name: str,
    arg_name: str | None,
    interface_name: str = "interface",
) -> int:
    """Resolves a block argument name to its index in the entry block.

    The lookup considers a region's implicit block arguments first
    (declared via implicit_args=(("name", "type"),) on RegionDef),
    then any block args derived from BindingList format elements
    (which the format pipeline appends after implicit args).

    Returns 0xFF (LOOM_BLOCK_ARG_INDEX_NONE) if arg_name is None.
    Raises ValueError if the region or arg name is not found.
    """
    if arg_name is None:
        return 0xFF
    region_def: RegionDef | None = None
    for r in op.regions:
        if r.name == region_name:
            region_def = r
            break
    if region_def is None:
        raise ValueError(f"{interface_name} on {op.name!r}: region {region_name!r} not found. Available: {[r.name for r in op.regions]}")
    # Implicit block args come first.
    for i, (name, _type) in enumerate(region_def.implicit_args):
        if name == arg_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: block arg {arg_name!r} not found in region {region_name!r}. Available implicit_args: {[name for name, _ in region_def.implicit_args]}")


# ============================================================================
# Interface vtable emission (table-driven)
# ============================================================================
#
# Interfaces declared in the Python DSL (FuncLikeInterface, etc.) are
# emitted as per-op .rodata vtable structs on the C side. Each interface
# also has a pointer slot on cache line 3 of loom_op_vtable_t.
#
# Adding a new interface is four steps:
#   1. Declare the Python NamedTuple in dsl.py.
#   2. Add the C struct, fat reference, cast function, and inline
#      accessors to ir.h / op_defs.{h,c}.
#   3. Add a pointer slot on cache line 3 of loom_op_vtable_t in ir.h.
#   4. Add an InterfaceSpec entry to _INTERFACES below.
#
# The generator code is entirely table-driven off _INTERFACES — adding
# an interface does not require any changes to the emission loops.


@dataclass(frozen=True, slots=True)
class InterfaceFieldSpec:
    """Describes one field in an interface's C vtable struct.

    py_field: Attribute name on the Python NamedTuple (e.g., "body",
        "callee", "iv"). The generator reads this via getattr.
    c_field: Field name in the C struct (e.g., "body_region_index",
        "callee_attr_index"). Emitted as the designated initializer key.
    kind: How to resolve the Python value to a C integer index or
        literal. One of:
          "attr"       — resolve to an op attribute index
          "region"     — resolve to an op region index
          "operand"    — resolve to an op operand index
          "result"     — resolve to an op result index
          "block_arg"  — resolve to a block argument index within a
                         region (requires region_field)
          "bool"       — render as a C boolean literal (true/false)
          "call_kind"  — render a loom_call_like_kind_t constant
          "c_ptr"      — render NULL or an address-of C data symbol
    region_field: For kind="block_arg", the name of the interface field
        that names the region the block arg belongs to. Unused for
        other kinds.
    required: True when a soft default is still semantically required and
        must resolve to a declared op field.
    c_pointee_type: For kind="c_ptr", the const pointee type used to emit an
        extern declaration for the referenced C symbol.
    """

    py_field: str
    c_field: str
    kind: str
    region_field: str = ""
    required: bool = False
    c_pointee_type: str = ""


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    """Generator metadata for one interface type.

    python_class: The Python NamedTuple class declared in dsl.py
        (e.g., FuncLikeInterface).
    name: Human-readable interface name used in error messages
        (e.g., "FuncLikeInterface").
    c_struct: Name of the C vtable struct type (e.g.,
        "loom_func_like_vtable_t").
    vtable_field: Field name on loom_op_vtable_t AND the per-op .rodata
        constant suffix (e.g., "func_like" → vtable->func_like and
        loom_<op>_func_like). These share a name by convention so the
        generator only needs one identifier per interface.
    fields: Ordered tuple of InterfaceFieldSpec for every field on the
        C struct. Order matches the desired .rodata emission order
        (cosmetic; designated initializers don't require any specific
        order).
    """

    python_class: type
    name: str
    c_struct: str
    vtable_field: str
    fields: tuple[InterfaceFieldSpec, ...]


# Registry of all interfaces known to the generator. Adding a new
# interface is a single entry here (plus the C-side struct/cast code).
_INTERFACES: tuple[InterfaceSpec, ...] = (
    InterfaceSpec(
        python_class=CallLikeInterface,
        name="CallLikeInterface",
        c_struct="loom_call_like_vtable_t",
        vtable_field="call_like",
        fields=(
            InterfaceFieldSpec("callee", "callee_attr_index", "attr"),
            InterfaceFieldSpec("purity", "purity_attr_index", "attr"),
            InterfaceFieldSpec("operands", "operand_offset", "operand"),
            InterfaceFieldSpec("results", "result_offset", "result"),
            InterfaceFieldSpec("kind", "kind", "call_kind"),
        ),
    ),
    InterfaceSpec(
        python_class=FuncLikeInterface,
        name="FuncLikeInterface",
        c_struct="loom_func_like_vtable_t",
        vtable_field="func_like",
        fields=(
            InterfaceFieldSpec("callee", "callee_attr_index", "attr"),
            InterfaceFieldSpec("import_module", "import_module_attr_index", "attr"),
            InterfaceFieldSpec("import_symbol", "import_symbol_attr_index", "attr"),
            InterfaceFieldSpec("target", "target_attr_index", "attr"),
            InterfaceFieldSpec("abi", "abi_attr_index", "attr"),
            InterfaceFieldSpec("abi_attrs", "abi_attrs_attr_index", "attr"),
            InterfaceFieldSpec("export_symbol", "export_symbol_attr_index", "attr"),
            InterfaceFieldSpec("export_attrs", "export_attrs_attr_index", "attr"),
            InterfaceFieldSpec("artifact", "artifact_attr_index", "attr"),
            InterfaceFieldSpec("export_ordinal", "export_ordinal_attr_index", "attr"),
            InterfaceFieldSpec("export_linkage", "export_linkage_attr_index", "attr"),
            InterfaceFieldSpec("workgroup_size_x", "workgroup_size_x_attr_index", "attr"),
            InterfaceFieldSpec("workgroup_size_y", "workgroup_size_y_attr_index", "attr"),
            InterfaceFieldSpec("workgroup_size_z", "workgroup_size_z_attr_index", "attr"),
            InterfaceFieldSpec("visibility", "visibility_attr_index", "attr"),
            InterfaceFieldSpec("cc", "cc_attr_index", "attr"),
            InterfaceFieldSpec("purity", "purity_attr_index", "attr"),
            InterfaceFieldSpec("predicates", "predicates_attr_index", "attr"),
            InterfaceFieldSpec("body", "body_region_index", "region"),
            InterfaceFieldSpec("implements", "implements_attr_index", "attr"),
            InterfaceFieldSpec("priority", "priority_attr_index", "attr"),
            InterfaceFieldSpec("args_as_operands", "args_as_operands", "bool"),
        ),
    ),
    InterfaceSpec(
        python_class=TargetLikeInterface,
        name="TargetLikeInterface",
        c_struct="loom_target_like_vtable_t",
        vtable_field="target_like",
        fields=(
            InterfaceFieldSpec("symbol", "symbol_attr_index", "attr"),
            InterfaceFieldSpec("selector", "selector_attr_index", "attr"),
            InterfaceFieldSpec("extensions", "extension_attrs_attr_index", "attr"),
            InterfaceFieldSpec(
                "descriptor",
                "descriptor",
                "c_ptr",
                c_pointee_type="loom_target_like_descriptor_t",
            ),
        ),
    ),
    InterfaceSpec(
        python_class=LoopLikeInterface,
        name="LoopLikeInterface",
        c_struct="loom_loop_like_vtable_t",
        vtable_field="loop_like",
        fields=(
            InterfaceFieldSpec("body", "body_region_index", "region"),
            InterfaceFieldSpec("condition_region", "condition_region_index", "region"),
            InterfaceFieldSpec("iv", "iv_block_arg_index", "block_arg", region_field="body"),
            InterfaceFieldSpec("iter_args", "iter_args_operand_offset", "operand"),
            InterfaceFieldSpec("lower_bound", "lower_bound_operand_index", "operand"),
            InterfaceFieldSpec("upper_bound", "upper_bound_operand_index", "operand"),
            InterfaceFieldSpec("step", "step_operand_index", "operand"),
        ),
    ),
    InterfaceSpec(
        python_class=RegionBranchInterface,
        name="RegionBranchInterface",
        c_struct="loom_region_branch_vtable_t",
        vtable_field="region_branch",
        fields=(InterfaceFieldSpec("selector", "selector_operand_index", "operand"),),
    ),
    InterfaceSpec(
        python_class=MemoryAccessInterface,
        name="MemoryAccessInterface",
        c_struct="loom_memory_access_vtable_t",
        vtable_field="memory_access",
        fields=(
            InterfaceFieldSpec("view", "view_operand_index", "operand", required=True),
            InterfaceFieldSpec("value", "value_operand_index", "operand"),
            InterfaceFieldSpec("expected", "expected_operand_index", "operand"),
            InterfaceFieldSpec("replacement", "replacement_operand_index", "operand"),
            InterfaceFieldSpec("mask", "mask_operand_index", "operand"),
            InterfaceFieldSpec("passthrough", "passthrough_operand_index", "operand"),
            InterfaceFieldSpec("offsets", "offsets_operand_index", "operand"),
            InterfaceFieldSpec("indices", "indices_operand_offset", "operand"),
            InterfaceFieldSpec("static_indices", "static_indices_attr_index", "attr"),
            InterfaceFieldSpec("cache_scope", "cache_scope_attr_index", "attr"),
            InterfaceFieldSpec("cache_temporal", "cache_temporal_attr_index", "attr"),
            InterfaceFieldSpec("atomic_kind", "atomic_kind_attr_index", "attr"),
            InterfaceFieldSpec("atomic_ordering", "atomic_ordering_attr_index", "attr"),
            InterfaceFieldSpec("atomic_success_ordering", "atomic_success_ordering_attr_index", "attr"),
            InterfaceFieldSpec("atomic_failure_ordering", "atomic_failure_ordering_attr_index", "attr"),
            InterfaceFieldSpec("atomic_scope", "atomic_scope_attr_index", "attr"),
        ),
    ),
)


_TARGET_PROJECTION_FIELDS: dict[str, tuple[str, str]] = {
    "codegen_format": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U32", "snapshot.codegen_format"),
    "target_triple": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "snapshot.target_triple"),
    "data_layout": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "snapshot.data_layout"),
    "artifact_format": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U32", "snapshot.artifact_format"),
    "target_cpu": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "snapshot.target_cpu"),
    "target_features": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "snapshot.target_features"),
    "default_pointer_bitwidth": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.default_pointer_bitwidth"),
    "index_bitwidth": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.index_bitwidth"),
    "offset_bitwidth": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.offset_bitwidth"),
    "max_workgroup_size_x": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_size.x"),
    "max_workgroup_size_y": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_size.y"),
    "max_workgroup_size_z": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_size.z"),
    "max_flat_workgroup_size": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_flat_workgroup_size"),
    "subgroup_size": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.subgroup_size"),
    "max_workgroup_count_x": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_count.x"),
    "max_workgroup_count_y": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_count.y"),
    "max_workgroup_count_z": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_count.z"),
    "memory_space_generic": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.generic"),
    "memory_space_global": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.global"),
    "memory_space_workgroup": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.workgroup"),
    "memory_space_constant": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.constant"),
    "memory_space_private": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.private_memory"),
    "memory_space_host": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.host"),
    "memory_space_descriptor": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.descriptor"),
    "abi": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U32", "export_plan.abi_kind"),
    "export_symbol": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "export_plan.export_symbol"),
    "linkage": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U32", "export_plan.linkage"),
    "hal_binding_alignment": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "export_plan.hal_kernel.binding_alignment"),
    "hal_buffer_resource_flags": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "export_plan.hal_kernel.buffer_resource_flags"),
    "contract_set_key": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "config.contract_set_key"),
    "contract_feature_bits": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64", "config.contract_feature_bits"),
}


def _interface_field_is_explicit(iface: Any, field_name: str) -> bool:
    """Returns true when the declaration explicitly supplied an interface field."""
    explicit_fields = getattr(iface, "_explicit_fields", None)
    return explicit_fields is None or field_name in explicit_fields


def _interface_reference_exists(
    op: Op,
    iface: Any,
    field_spec: InterfaceFieldSpec,
    py_value: Any,
) -> bool:
    """Returns true if an implicitly defaulted interface reference exists."""
    if py_value is None:
        return True
    if field_spec.kind == "attr":
        return any(attr_def.name == py_value and attr_def.attr_type != ATTR_TYPE_FLAGS for attr_def in op.attrs)
    if field_spec.kind == "region":
        return any(region_def.name == py_value for region_def in op.regions)
    if field_spec.kind == "operand":
        return any(operand.name == py_value for operand in op.operands)
    if field_spec.kind == "result":
        return any(result.name == py_value for result in op.results)
    if field_spec.kind == "block_arg":
        region_value = getattr(iface, field_spec.region_field)
        region_def = next(
            (region for region in op.regions if region.name == region_value),
            None,
        )
        if region_def is None:
            return False
        return any(name == py_value for name, _type in region_def.implicit_args)
    return True


def _interface_soft_default_is_absent(
    op: Op,
    iface: Any,
    field_spec: InterfaceFieldSpec,
    py_value: Any,
) -> bool:
    """Returns true when a non-required soft default should emit none."""
    if field_spec.required:
        return False
    if _interface_field_is_explicit(iface, field_spec.py_field):
        return False
    return not _interface_reference_exists(op, iface, field_spec, py_value)


def _resolve_interface_field(
    op: Op,
    iface: Any,
    field_spec: InterfaceFieldSpec,
    interface_name: str,
) -> str:
    """Resolves one interface field to its emitted C initializer value.

    Dispatches on field_spec.kind to the right name-to-index resolver,
    then formats the result as a string ready to follow an `= ` in a
    designated initializer.
    """
    py_value = getattr(iface, field_spec.py_field)
    if _interface_soft_default_is_absent(op, iface, field_spec, py_value):
        return "255"
    if field_spec.kind == "attr":
        return str(_resolve_attr_index(op, py_value, interface_name))
    if field_spec.kind == "region":
        return str(_resolve_region_index(op, py_value, interface_name))
    if field_spec.kind == "operand":
        return str(_resolve_operand_index(op, py_value, interface_name))
    if field_spec.kind == "result":
        return str(_resolve_result_index(op, py_value, interface_name))
    if field_spec.kind == "block_arg":
        if not field_spec.region_field:
            raise ValueError(f"{interface_name} field {field_spec.py_field!r}: kind='block_arg' requires region_field to name the region this block arg belongs to")
        region_value = getattr(iface, field_spec.region_field)
        return str(_resolve_block_arg_index(op, region_value, py_value, interface_name))
    if field_spec.kind == "bool":
        return "true" if py_value else "false"
    if field_spec.kind == "call_kind":
        if not isinstance(py_value, CallLikeKind):
            raise ValueError(f"{interface_name} field {field_spec.py_field!r}: expected CallLikeKind, got {py_value!r}")
        return CALL_LIKE_KIND_MAP[py_value]
    if field_spec.kind == "c_ptr":
        if isinstance(iface, TargetLikeInterface) and field_spec.py_field == "descriptor" and iface.bundle_table is not None:
            descriptor = iface.descriptor or f"{_c_prefix(op)}_target_like_descriptor"
            return f"&{_normalize_c_symbol_reference(descriptor)}"
        if py_value is None:
            return "NULL"
        if not isinstance(py_value, str) or not py_value:
            raise ValueError(f"{interface_name} field {field_spec.py_field!r}: expected C symbol name or None, got {py_value!r}")
        return f"&{_normalize_c_symbol_reference(py_value)}"
    raise ValueError(f"{interface_name} field {field_spec.py_field!r}: unknown kind {field_spec.kind!r}")


def _resolve_soft_memory_field(
    op: Op,
    iface: MemoryAccessInterface,
    field_name: str,
    field_kind: str,
    interface_name: str,
) -> int | None:
    """Resolves a MemoryAccessInterface field, returning None for absent defaults."""
    field_spec = InterfaceFieldSpec(field_name, "", field_kind)
    py_value = getattr(iface, field_name)
    if _interface_soft_default_is_absent(op, iface, field_spec, py_value):
        return None
    if field_kind == "attr":
        index = _resolve_attr_index(op, py_value, interface_name)
    elif field_kind == "operand":
        index = _resolve_operand_index(op, py_value, interface_name)
    else:
        raise ValueError(f"unsupported MemoryAccessInterface field kind {field_kind!r}")
    return None if index == 0xFF else index


def _validate_call_like_interface(op: Op, iface: CallLikeInterface, interface_name: str) -> None:
    """Validates CallLikeInterface's trailing variadic slice contract."""
    operand_index = _resolve_operand_index(op, iface.operands, interface_name)
    operand = op.operands[operand_index]
    if not operand.variadic:
        raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.operands!r} must be variadic")
    if operand_index + 1 != len(op.operands):
        raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.operands!r} must be the trailing operand field")

    result_index = _resolve_result_index(op, iface.results, interface_name)
    result = op.results[result_index]
    if not result.variadic:
        raise ValueError(f"{interface_name} on {op.name!r}: result {iface.results!r} must be variadic")
    if result_index + 1 != len(op.results):
        raise ValueError(f"{interface_name} on {op.name!r}: result {iface.results!r} must be the trailing result field")


def _validate_memory_access_interface(op: Op, iface: MemoryAccessInterface, interface_name: str) -> None:
    """Validates MemoryAccessInterface's optional role coherence."""
    if iface.view is None:
        raise ValueError(f"{interface_name} on {op.name!r}: view operand is required")
    _resolve_operand_index(op, iface.view, interface_name)
    indices_index = _resolve_soft_memory_field(op, iface, "indices", "operand", interface_name)
    if indices_index is not None:
        indices_operand = op.operands[indices_index]
        if not indices_operand.variadic:
            raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.indices!r} must be variadic")
        if indices_index + 1 != len(op.operands):
            raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.indices!r} must be the trailing operand field")

    cache_scope_index = _resolve_soft_memory_field(op, iface, "cache_scope", "attr", interface_name)
    cache_temporal_index = _resolve_soft_memory_field(op, iface, "cache_temporal", "attr", interface_name)
    if (cache_scope_index is None) != (cache_temporal_index is None):
        raise ValueError(f"{interface_name} on {op.name!r}: cache_scope and cache_temporal must be declared together")

    atomic_ordering_index = _resolve_soft_memory_field(op, iface, "atomic_ordering", "attr", interface_name)
    atomic_success_ordering_index = _resolve_soft_memory_field(op, iface, "atomic_success_ordering", "attr", interface_name)
    atomic_failure_ordering_index = _resolve_soft_memory_field(op, iface, "atomic_failure_ordering", "attr", interface_name)
    if (atomic_success_ordering_index is None) != (atomic_failure_ordering_index is None):
        raise ValueError(f"{interface_name} on {op.name!r}: atomic_success_ordering and atomic_failure_ordering must be declared together")
    if atomic_ordering_index is not None and atomic_success_ordering_index is not None:
        raise ValueError(f"{interface_name} on {op.name!r}: use either atomic_ordering or success/failure orderings, not both")


def _target_like_projection_entries(op: Op) -> list[tuple[int, str, str]]:
    entries: list[tuple[int, str, str]] = []
    for attr_def in op.attrs:
        projection = _TARGET_PROJECTION_FIELDS.get(attr_def.name)
        if projection is None:
            continue
        value_kind, storage_field = projection
        attr_index = _resolve_attr_index(op, attr_def.name, "TargetLikeInterface")
        entries.append((attr_index, value_kind, storage_field))
    return entries


def _emit_target_like_descriptor(op: Op, iface: TargetLikeInterface, lines: list[str]) -> None:
    if iface.bundle_table is None:
        return
    descriptor = _normalize_c_symbol_reference(iface.descriptor or f"{_c_prefix(op)}_target_like_descriptor")
    bundle_table = _normalize_c_symbol_reference(iface.bundle_table)
    prefix = _c_prefix(op)
    projections = _target_like_projection_entries(op)
    projection_array = "NULL"
    if projections:
        projection_array = f"{prefix}_target_projections"
        lines.append(f"static const loom_target_projection_t {projection_array}[] = {{")
        for attr_index, value_kind, storage_field in projections:
            lines.append(f"    {{offsetof(loom_target_bundle_storage_t, {storage_field}), {attr_index}, {value_kind}}},")
        lines.append("};")
    lines.append(f"static const loom_target_like_descriptor_t {descriptor} = {{")
    lines.append(f"    .bundle_table = &{bundle_table},")
    if projection_array != "NULL":
        lines.append(f"    .projections = {projection_array},")
        lines.append(f"    .projection_count = IREE_ARRAYSIZE({projection_array}),")
    lines.append("};")
    lines.append("")


def _emit_interface_vtable(op: Op, spec: InterfaceSpec, lines: list[str]) -> None:
    """Appends the .rodata struct declaration for |op|'s |spec| interface.

    No-op if |op| does not implement |spec|. When emitted, the
    resulting C initializer is pointed to from the main op vtable's
    cache line 3 slot named by spec.vtable_field.
    """
    iface = _find_interface(op, spec.python_class)
    if iface is None:
        return
    if isinstance(iface, CallLikeInterface):
        _validate_call_like_interface(op, iface, spec.name)
    if isinstance(iface, MemoryAccessInterface):
        _validate_memory_access_interface(op, iface, spec.name)
    prefix = _c_prefix(op)
    lines.append(f"static const {spec.c_struct} {prefix}_{spec.vtable_field} = {{")
    for field_spec in spec.fields:
        value_str = _resolve_interface_field(op, iface, field_spec, spec.name)
        lines.append(f"    .{field_spec.c_field} = {value_str},")
    lines.append("};")
    lines.append("")


def _interface_vtable_ptr(op: Op, spec: InterfaceSpec) -> str:
    """Returns the C expression for the interface pointer on the main vtable.

    Evaluates to `&<op>_<vtable_field>` when the op implements the
    interface, or `NULL` otherwise.
    """
    if _find_interface(op, spec.python_class) is None:
        return "NULL"
    return f"&{_c_prefix(op)}_{spec.vtable_field}"


def _collect_shared_enums(
    dialect_name: str,
    ops: Sequence[Op],
) -> dict[int, tuple[str, str, EnumDef]]:
    """Identifies EnumDef objects shared across multiple ops in a dialect.

    Returns a mapping from id(enum_def) to (c_prefix, const_prefix, enum_def)
    for each EnumDef used by more than one op. The naming uses the dialect
    prefix + attr name (e.g., loom_func_cc for the func dialect's cc attr).

    EnumDefs used by only one op are not included — they keep per-op naming.
    """
    from collections import defaultdict

    # Map id(enum_def) -> list of (op, attr_def) pairs.
    usage: dict[int, list[tuple[Op, AttrDef]]] = defaultdict(list)
    for op in ops:
        for attr_def in op.attrs:
            if attr_def.attr_type == "enum" and attr_def.enum_def is not None:
                if attr_def.enum_def.c_type is not None:
                    continue
                usage[id(attr_def.enum_def)].append((op, attr_def))

    shared: dict[int, tuple[str, str, EnumDef]] = {}
    for enum_id, pairs in usage.items():
        if len(pairs) <= 1:
            continue
        # All pairs reference the same EnumDef and should have the same
        # attr name (since they share the object). Verify this.
        attr_names = {attr_def.name for _, attr_def in pairs}
        if len(attr_names) != 1:
            # Same EnumDef used under different attr names — can't share
            # the C name. Fall back to per-op naming for all.
            continue
        attr_name = next(iter(attr_names))
        c_prefix = f"loom_{dialect_name}_{attr_name}"
        const_prefix = c_prefix.upper()
        enum_def = pairs[0][1].enum_def
        assert enum_def is not None  # Filtered at collection time.
        shared[enum_id] = (c_prefix, const_prefix, enum_def)

    return shared


def _enum_c_prefix_from_type(c_type: str) -> str:
    """Returns the C prefix for an enum typedef name."""
    if c_type.endswith("_t"):
        return c_type[:-2]
    return c_type


def _enum_c_prefix(
    op: Op,
    attr_def: AttrDef,
    shared_enums: dict[int, tuple[str, str, EnumDef]],
) -> tuple[str, str]:
    """Returns (c_prefix, const_prefix) for an enum attr.

    Uses the shared dialect-level name if the EnumDef is shared across
    multiple ops, otherwise falls back to the per-op prefix.
    """
    if attr_def.enum_def is not None and attr_def.enum_def.c_type is not None:
        assert attr_def.enum_def.c_const_prefix is not None
        return (
            _enum_c_prefix_from_type(attr_def.enum_def.c_type),
            attr_def.enum_def.c_const_prefix,
        )
    if attr_def.enum_def is not None and id(attr_def.enum_def) in shared_enums:
        c_prefix, const_prefix, _ = shared_enums[id(attr_def.enum_def)]
        return c_prefix, const_prefix
    c_prefix = _c_prefix(op) + "_" + attr_def.name
    return c_prefix, c_prefix.upper()


def _enum_c_type(
    op: Op,
    attr_def: AttrDef,
    shared_enums: dict[int, tuple[str, str, EnumDef]],
) -> str:
    """Returns the C type exposed for an enum attribute."""
    if attr_def.enum_def is not None and attr_def.enum_def.c_type is not None:
        return attr_def.enum_def.c_type
    c_prefix, _ = _enum_c_prefix(op, attr_def, shared_enums)
    return f"{c_prefix}_t"


def _enum_names_array_name(
    op: Op,
    attr_def: AttrDef,
    shared_enums: dict[int, tuple[str, str, EnumDef]],
) -> str:
    """Returns the enum keyword table symbol for an enum attribute."""
    if attr_def.enum_def is not None and attr_def.enum_def.c_type is not None:
        c_prefix, _ = _enum_c_prefix(op, attr_def, shared_enums)
        return f"{c_prefix}_names"
    shared = shared_enums.get(id(attr_def.enum_def)) if attr_def.enum_def is not None else None
    if shared:
        return f"{shared[0]}_names"
    return f"{_c_prefix(op)}_{attr_def.name}_names"


def _enum_case_c_ident(keyword: str) -> str:
    """Converts an enum assembly keyword to a C enum/macro suffix."""
    return keyword.replace(".", "_").upper()


# ============================================================================
# Format element translation
# ============================================================================


def _translate_format_elements(
    op: Op,
) -> list[tuple[str, int, str]]:
    """Translates an op's format spec to C format element initializers.

    Returns a list of (kind_str, field_index, data_str) triples that
    become C struct initializers: {kind, field_index, data}.
    """
    layout = compute_layout(op)
    elements: list[tuple[str, int, str]] = []

    # Implicit fields are created by the printer/parser from context,
    # not declared as operands/results/attrs/regions.
    implicit_fields = {"iv", "args"}

    def _resolve_field(name: str) -> tuple[FieldKind, int]:
        desc = layout.fields.get(name)
        if desc is None:
            raise ValueError(f"Op '{op.name}': format references undeclared field '{name}'")
        if desc.kind == FieldKind.ATTR:
            return desc.kind, _resolve_attr_index(op, name, "format")
        return desc.kind, desc.index

    def _resolve_region_syntax(syntax: str) -> str:
        c_name = REGION_SYNTAX_MAP.get(syntax)
        if c_name is None:
            known = ", ".join(repr(name) for name in sorted(REGION_SYNTAX_MAP))
            raise ValueError(f"Op '{op.name}': unknown region syntax {syntax!r}. Known syntaxes: {known}")
        return c_name

    def _walk(fmt_elements: tuple[FormatElement, ...]) -> None:
        for element in fmt_elements:
            match element:
                case Ref(field=name):
                    if name in implicit_fields:
                        # Implicit field (iv, etc.) — emitted as operand ref
                        # with a special index. The printer/parser handles
                        # implicit fields specially (e.g., IV is the first
                        # block arg in a loop body).
                        elements.append(("LOOM_FORMAT_KIND_OPERAND_REF", 0xFF, "0"))
                    else:
                        ref_kind, ref_index = _resolve_field(name)
                        if ref_kind == FieldKind.OPERAND:
                            elements.append(("LOOM_FORMAT_KIND_OPERAND_REF", ref_index, "0"))
                        else:
                            raise ValueError(f"Op '{op.name}': Ref('{name}') references {ref_kind.name}, expected OPERAND")

                case Refs(field=name):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': Refs('{name}') references {kind.name}, expected OPERAND")
                    elements.append(("LOOM_FORMAT_KIND_OPERAND_REFS", index, "0"))

                case TypedRefs(field=name):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': TypedRefs('{name}') references {kind.name}, expected OPERAND")
                    elements.append(("LOOM_FORMAT_KIND_OPERAND_TYPED_REFS", index, "0"))

                case BlockRef(field=name):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.SUCCESSOR:
                        raise ValueError(f"Op '{op.name}': BlockRef('{name}') references {kind.name}, expected SUCCESSOR")
                    elements.append(("LOOM_FORMAT_KIND_SUCCESSOR_REF", index, "0"))

                case Attr(field=name):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': Attr('{name}') references {kind.name}, expected ATTR")
                    elements.append(("LOOM_FORMAT_KIND_ATTR_VALUE", index, "0"))

                case SymbolRef(field=name):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': SymbolRef('{name}') references {kind.name}, expected ATTR")
                    elements.append(("LOOM_FORMAT_KIND_SYMBOL_REF", index, "0"))

                case TypeOf(field=name):
                    kind, index = _resolve_field(name)
                    if kind == FieldKind.OPERAND:
                        elements.append(("LOOM_FORMAT_KIND_OPERAND_TYPE", index, "0"))
                    elif kind == FieldKind.RESULT:
                        elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE", index, "0"))
                    else:
                        raise ValueError(f"Op '{op.name}': TypeOf('{name}') references {kind.name}")

                case TypesOf(field=name):
                    kind, index = _resolve_field(name)
                    if kind == FieldKind.OPERAND:
                        elements.append(("LOOM_FORMAT_KIND_OPERAND_TYPES", index, "0"))
                    elif kind == FieldKind.RESULT:
                        elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE_LIST", index, "0"))
                    else:
                        raise ValueError(f"Op '{op.name}': TypesOf('{name}') references {kind.name}")

                case ResultType(field=name):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.RESULT:
                        raise ValueError(f"Op '{op.name}': ResultType('{name}') references {kind.name}, expected RESULT")
                    elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE", index, "0"))

                case ResultTypeList(field=name, parens=parens):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.RESULT:
                        raise ValueError(f"Op '{op.name}': ResultTypeList('{name}') references {kind.name}, expected RESULT")
                    payload = "LOOM_RESULT_TYPE_LIST_PARENS" if parens else "0"
                    elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE_LIST", index, payload))

                case Clause(name=name, elements=inner):
                    kw_enum = KEYWORD_MAP.get(name)
                    if kw_enum is None:
                        raise ValueError(f"Op '{op.name}': unknown clause name '{name}'. Add it to KEYWORD_MAP and the C keyword enum.")
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, kw_enum))
                    elements.append(("LOOM_FORMAT_KIND_GLUE", 0, "0"))
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, KEYWORD_MAP["("]))
                    _walk(inner)
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, KEYWORD_MAP[")"]))

                case Keyword(text=text):
                    kw_enum = KEYWORD_MAP.get(text)
                    if kw_enum is None:
                        raise ValueError(f"Op '{op.name}': unknown keyword '{text}'. Add it to KEYWORD_MAP and the C keyword enum.")
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, kw_enum))

                case AttrDict(field=field_name):
                    attr_index = 0
                    attr_dict_flags = "0"
                    if field_name:
                        non_flags = _non_flags_attrs(op)
                        for idx, ad in enumerate(non_flags):
                            if ad.name == field_name:
                                attr_index = idx
                                break
                    else:
                        attr_dict_flags = "LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS"
                    elements.append(("LOOM_FORMAT_KIND_ATTR_DICT", attr_index, attr_dict_flags))

                case OperandDict(operands=operand_field, names=names_field):
                    operand_kind, operand_index = _resolve_field(operand_field)
                    names_kind, names_index = _resolve_field(names_field)
                    if operand_kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': OperandDict operands field '{operand_field}' is not an operand field")
                    if names_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': OperandDict names field '{names_field}' is not an attr field")
                    operand_desc = layout.fields[operand_field]
                    if not operand_desc.variadic:
                        raise ValueError(f"Op '{op.name}': OperandDict operands field '{operand_field}' must be variadic")
                    attr_def = op.attr(names_field)
                    if attr_def is None or attr_def.attr_type != "dict":
                        raise ValueError(f"Op '{op.name}': OperandDict names field '{names_field}' must be a dict attr")
                    elements.append(("LOOM_FORMAT_KIND_OPERAND_DICT", operand_index, str(names_index)))

                case AttrTable(keys=keys_field, values=values_field):
                    values_kind, values_index = _resolve_field(values_field)
                    keys_kind, keys_index = _resolve_field(keys_field)
                    if values_kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' is not an operand field")
                    if keys_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': AttrTable keys field '{keys_field}' is not an attr field")
                    values_desc = layout.fields[values_field]
                    if not values_desc.variadic:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' must be variadic")
                    attr_def = op.attr(keys_field)
                    if attr_def is None or attr_def.attr_type != "i64_array":
                        raise ValueError(f"Op '{op.name}': AttrTable keys field '{keys_field}' must be an i64_array attr")
                    elements.append(("LOOM_FORMAT_KIND_ATTR_TABLE", values_index, str(keys_index)))

                case RegionTable(keys=keys_field, case_regions=case_regions_field, default_region=default_region_field):
                    case_kind, case_index = _resolve_field(case_regions_field)
                    default_kind, default_index = _resolve_field(default_region_field)
                    keys_kind, keys_index = _resolve_field(keys_field)
                    if case_kind != FieldKind.REGION:
                        raise ValueError(f"Op '{op.name}': RegionTable case field '{case_regions_field}' is not a region field")
                    if default_kind != FieldKind.REGION:
                        raise ValueError(f"Op '{op.name}': RegionTable default field '{default_region_field}' is not a region field")
                    if keys_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': RegionTable keys field '{keys_field}' is not an attr field")
                    case_desc = layout.fields[case_regions_field]
                    if not case_desc.variadic:
                        raise ValueError(f"Op '{op.name}': RegionTable case field '{case_regions_field}' must be variadic")
                    attr_def = op.attr(keys_field)
                    if attr_def is None or attr_def.attr_type != "i64_array":
                        raise ValueError(f"Op '{op.name}': RegionTable keys field '{keys_field}' must be an i64_array attr")
                    payload = f"LOOM_FORMAT_REGION_TABLE_DATA({keys_index}, {default_index})"
                    elements.append(("LOOM_FORMAT_KIND_REGION_TABLE", case_index, payload))

                case RegionFmt(field=name, syntax=syntax):
                    kind, index = _resolve_field(name)
                    if layout.fields[name].variadic:
                        raise ValueError(f"Op '{op.name}': variadic Region '{name}' must use RegionTable or another table format")
                    elements.append(("LOOM_FORMAT_KIND_REGION", index, _resolve_region_syntax(syntax)))

                case IndexList(dynamic=dynamic_field, static=static_field, glue=glue):
                    dyn_kind, dyn_index = _resolve_field(dynamic_field)
                    sta_kind, sta_index = _resolve_field(static_field)
                    if sta_index > 0x7FFF:
                        raise ValueError(f"Op '{op.name}': IndexList static attr index {sta_index} exceeds packed format limit")
                    payload = str(sta_index) if glue else f"LOOM_FORMAT_INDEX_LIST_DATA({sta_index}, false)"
                    elements.append(("LOOM_FORMAT_KIND_INDEX_LIST", dyn_index, payload))

                case BindingList(field=name, kind=binding_kind):
                    fld_kind, index = _resolve_field(name)
                    bk = "LOOM_BINDING_ELEMENT" if binding_kind == "element" else "LOOM_BINDING_CAPTURE"
                    elements.append(("LOOM_FORMAT_KIND_BINDING_LIST", index, bk))

                case BlockArgs(region=name):
                    kind, index = _resolve_field(name)
                    if kind != FieldKind.REGION:
                        raise ValueError(f"Op '{op.name}': BlockArgs region field '{name}' is not a region field")
                    elements.append(("LOOM_FORMAT_KIND_BLOCK_ARGS", index, "0"))

                case FuncArgs(field=name):
                    elements.append(("LOOM_FORMAT_KIND_FUNC_ARGS", 0, "0"))

                case PredicateList(field=name):
                    fld_kind, index = _resolve_field(name)
                    elements.append(("LOOM_FORMAT_KIND_PREDICATE_LIST", index, "0"))

                case OptionalGroup(elements=inner, anchor=anchor):
                    # Resolve anchor to determine category.
                    anchor_desc = layout.fields.get(anchor)
                    if anchor_desc is None:
                        # Implicit field (predicates, etc.) with no backing
                        # storage on the op. Use LOOM_ANCHOR_ATTR with an
                        # index beyond any real attr so the printer's bounds
                        # check evaluates to "not present."
                        anchor_category = 1  # LOOM_ANCHOR_ATTR
                        anchor_index = 0xFF
                    else:
                        anchor_index = anchor_desc.index
                        if anchor_desc.kind == FieldKind.OPERAND:
                            anchor_category = 0  # LOOM_ANCHOR_OPERAND
                        elif anchor_desc.kind == FieldKind.ATTR:
                            anchor_category = 1  # LOOM_ANCHOR_ATTR
                        elif anchor_desc.kind == FieldKind.REGION:
                            anchor_category = 2  # LOOM_ANCHOR_REGION
                        elif anchor_desc.kind == FieldKind.RESULT:
                            anchor_category = 3  # LOOM_ANCHOR_RESULTS
                        else:
                            anchor_category = 1

                    # Count inner elements (recursively).
                    inner_start = len(elements)
                    # Placeholder for the OPTIONAL_GROUP element itself.
                    elements.append(("LOOM_FORMAT_KIND_OPTIONAL_GROUP", 0, "0"))
                    _walk(inner)
                    inner_count = len(elements) - inner_start - 1
                    # Patch the placeholder with actual skip count.
                    data = f"({inner_count} << 2) | {anchor_category}"
                    elements[inner_start] = (
                        "LOOM_FORMAT_KIND_OPTIONAL_GROUP",
                        anchor_index,
                        data,
                    )

                case Scope(elements=inner):
                    # Scoped group: push scope, process children, pop scope.
                    # Encoded like OptionalGroup: placeholder + children.
                    inner_start = len(elements)
                    elements.append(("LOOM_FORMAT_KIND_SCOPE", 0, "0"))
                    _walk(inner)
                    inner_count = len(elements) - inner_start - 1
                    elements[inner_start] = (
                        "LOOM_FORMAT_KIND_SCOPE",
                        0,
                        str(inner_count),
                    )

                case Flags(field=name):
                    # Flags are stored in instance_flags, not in the
                    # attribute array. field_index is unused (0).
                    elements.append(("LOOM_FORMAT_KIND_FLAGS", 0, "0"))

                case OpRef(field=name):
                    kind, index = _resolve_field(name)
                    elements.append(("LOOM_FORMAT_KIND_OP_REF", index, "0"))

                case DescriptorRef(key=key_name, stable_id=stable_id_name):
                    key_kind, key_index = _resolve_field(key_name)
                    stable_id_kind, stable_id_index = _resolve_field(stable_id_name)
                    key_attr = op.attr(key_name)
                    stable_id_attr = op.attr(stable_id_name)
                    if key_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': DescriptorRef key field '{key_name}' is not an attr field")
                    if key_attr is None or key_attr.attr_type != "string":
                        raise ValueError(f"Op '{op.name}': DescriptorRef key field '{key_name}' must be a string attr")
                    if stable_id_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': DescriptorRef stable ID field '{stable_id_name}' is not an attr field")
                    if stable_id_attr is None or stable_id_attr.attr_type != "i64":
                        raise ValueError(f"Op '{op.name}': DescriptorRef stable ID field '{stable_id_name}' must be an i64 attr")
                    elements.append(
                        (
                            "LOOM_FORMAT_KIND_DESCRIPTOR_REF",
                            key_index,
                            str(stable_id_index),
                        )
                    )

                case TemplateParam(field=name):
                    kind, index = _resolve_field(name)
                    elements.append(("LOOM_FORMAT_KIND_TEMPLATE_PARAM", index, "0"))

                case Glue():
                    elements.append(("LOOM_FORMAT_KIND_GLUE", 0, "0"))

    _walk(op.format)
    return elements


# ============================================================================
# Trait translation
# ============================================================================


def _implicit_terminator_kind(op: Op, ops_by_name: dict[str, Op]) -> str:
    """Returns the C op kind for this op's implicit terminator trait."""
    terminator_traits = [trait for trait in op.traits if trait.name == "ImplicitTerminator"]
    if not terminator_traits:
        return "LOOM_OP_KIND_UNKNOWN"
    if len(terminator_traits) > 1:
        raise ValueError(f"Op '{op.name}': duplicate ImplicitTerminator traits are not supported")
    trait = terminator_traits[0]
    if len(trait.args) != 1:
        raise ValueError(f"Op '{op.name}': ImplicitTerminator requires one op name argument")
    terminator_name = trait.args[0]
    terminator_op = ops_by_name.get(terminator_name)
    if terminator_op is None:
        raise ValueError(f"Op '{op.name}': ImplicitTerminator '{terminator_name}' must name an op in the '{op.namespace}' dialect")
    if not any(trait.name == "Terminator" for trait in terminator_op.traits):
        raise ValueError(f"Op '{op.name}': ImplicitTerminator '{terminator_name}' is not marked with the Terminator trait")
    terminator_layout = compute_layout(terminator_op)
    if terminator_layout.fixed_operand_count != 0 or terminator_layout.fixed_result_count != 0 or terminator_op.attrs or terminator_op.regions:
        raise ValueError(f"Op '{op.name}': ImplicitTerminator '{terminator_name}' must be instantiable with zero operands, results, attrs, and regions")
    return _c_enum_name(terminator_op)


def _region_terminator_kind(op: Op, region: RegionDef, ops_by_name: dict[str, Op]) -> str:
    """Returns the C op kind required for explicit terminators in a region."""
    if region.terminator is None:
        return "LOOM_OP_KIND_UNKNOWN"
    terminator_op = ops_by_name.get(region.terminator)
    if terminator_op is None:
        raise ValueError(f"Op '{op.name}' region '{region.name}': terminator '{region.terminator}' must name a registered op")
    if not any(trait.name == "Terminator" for trait in terminator_op.traits):
        raise ValueError(f"Op '{op.name}' region '{region.name}': terminator '{region.terminator}' is not marked with the Terminator trait")
    return _c_enum_name(terminator_op)


def _trait_op_kinds(
    op: Op,
    ops_by_name: dict[str, Op],
    trait_name: str,
) -> list[str]:
    """Returns op-kind enum names referenced by parameterized placement traits."""
    kinds: list[str] = []
    for trait in op.traits:
        if trait.name != trait_name:
            continue
        if len(trait.args) != 1:
            raise ValueError(f"Op '{op.name}': {trait_name} requires one op name argument")
        ancestor_name = trait.args[0]
        ancestor_op = ops_by_name.get(ancestor_name)
        if ancestor_op is None:
            raise ValueError(f"Op '{op.name}': {trait_name} '{ancestor_name}' must name an op in the '{op.namespace}' dialect")
        kinds.append(_c_enum_name(ancestor_op))
    return kinds


def _trait_flags(op: Op) -> str:
    """Returns the C trait bitfield expression for an op.

    Includes explicitly declared traits and derived summary bits:
    READS_MEMORY and WRITES_MEMORY are derived from per-operand effects.
    PURE is derived when no effects, no allocating results, and no
    NON_DETERMINISTIC or UNKNOWN_EFFECTS traits are present.
    """
    bits = []
    for trait in op.traits:
        c_name = TRAIT_MAP.get(trait.name)
        if c_name:
            bits.append(c_name)

    # Derive summary bits from per-operand effects.
    has_read = False
    has_write = False
    for effect in op.effects:
        if effect.kind in (EffectKind.READ, EffectKind.READWRITE):
            has_read = True
        if effect.kind in (EffectKind.WRITE, EffectKind.READWRITE):
            has_write = True
    if has_read:
        bits.append("LOOM_TRAIT_READS_MEMORY")
    if has_write:
        bits.append("LOOM_TRAIT_WRITES_MEMORY")

    # Derive UNIQUE_IDENTITY when any result allocates fresh storage.
    has_allocating_result = any(r.allocates for r in op.results)
    has_explicit_unique_identity = any(t.name == "UniqueIdentity" for t in op.traits)
    if has_allocating_result and not has_explicit_unique_identity:
        bits.append("LOOM_TRAIT_UNIQUE_IDENTITY")

    # Derive PURE when the op has no effects and no conflicting traits.
    explicit_pure = any(t.name == "Pure" for t in op.traits)
    has_non_deterministic = any(t.name == "NonDeterministic" for t in op.traits)
    has_unknown_effects = any(t.name == "UnknownEffects" for t in op.traits)
    has_hint = any(t.name == "Hint" for t in op.traits)
    if (
        not explicit_pure
        and not op.effects
        and not has_non_deterministic
        and not has_unknown_effects
        and not has_hint
        and not has_allocating_result
        and not has_explicit_unique_identity
        and not has_read
        and not has_write
    ):
        bits.append("LOOM_TRAIT_PURE")

    if not bits:
        return "0"
    return " | ".join(bits)


# ============================================================================
# Pattern detection
# ============================================================================


def _has_flags_attr(op: Op) -> bool:
    """Check if the op has any flags-typed attributes."""
    return any(a.attr_type == ATTR_TYPE_FLAGS for a in op.attrs)


def _non_flags_attrs(op: Op) -> list[AttrDef]:
    """Return attributes that are not flags (regular attrs for the attribute array)."""
    return [a for a in op.attrs if a.attr_type != ATTR_TYPE_FLAGS]


def _detect_builder_pattern(op: Op) -> str | None:
    """Detects if an op matches a standard builder macro pattern.

    Returns the macro name suffix or None for complex ops.
    """
    layout = compute_layout(op)
    non_flags = _non_flags_attrs(op)
    has_flags = _has_flags_attr(op)
    has_template_param = any(isinstance(e, TemplateParam) for e in _flatten_format(op.format))
    operand_names = tuple(operand.name for operand in op.operands)

    # Binary: 2 fixed operands, 1 fixed result, no real attrs/regions.
    # Only use the generic lhs/rhs builder macro when the op uses those exact
    # field names; otherwise the generated C API should preserve the op's
    # semantic operand names.
    if (
        operand_names == ("lhs", "rhs")
        and layout.fixed_operand_count == 2
        and not layout.variadic_operand
        and layout.fixed_result_count == 1
        and not layout.variadic_result
        and len(non_flags) == 0
        and len(op.regions) == 0
    ):
        return "BINARY_WITH_FLAGS" if has_flags else "BINARY"

    # Unary: 1 fixed operand, 1 fixed result, no real attrs/regions.
    if (
        operand_names == ("input",)
        and layout.fixed_operand_count == 1
        and not layout.variadic_operand
        and layout.fixed_result_count == 1
        and not layout.variadic_result
        and len(non_flags) == 0
        and len(op.regions) == 0
    ):
        # Distinguish cast from unary by checking for "to" keyword.
        has_to = any(isinstance(e, Keyword) and e.text == "to" for e in op.format)
        if has_to:
            return "CAST"
        return "UNARY_WITH_FLAGS" if has_flags else "UNARY"

    # Comparison: 2 fixed operands, 1 fixed result, 1 enum attr, no regions.
    if (
        operand_names == ("lhs", "rhs")
        and layout.fixed_operand_count == 2
        and not layout.variadic_operand
        and layout.fixed_result_count == 1
        and not layout.variadic_result
        and len(non_flags) == 1
        and non_flags[0].attr_type == "enum"
        and not has_template_param
        and len(op.regions) == 0
    ):
        return "COMPARISON_WITH_FLAGS" if has_flags else "COMPARISON"

    return None


# ============================================================================
# C builder parameter extraction
# ============================================================================

# Maps format-spec-derived param kinds to C parameter declarations.
# Each entry is (c_type, is_array) where is_array means we also emit a count.
_C_ATTR_TYPE_MAP: dict[str, str] = {
    "i64": "int64_t",
    "f64": "double",
    "string": "loom_string_id_t",
    "bool": "bool",
    "enum": "uint8_t",
    "symbol": "loom_symbol_ref_t",
    "i64_array": "const int64_t*",
    "type": "uint32_t",
    "encoding": "uint16_t",
    "dict": "loom_named_attr_slice_t",
    "any": "loom_attribute_t",
}

# Maps implicit block arg type keywords to C scalar type constants.
_IMPLICIT_ARG_TYPE_MAP: dict[str, str] = {
    "index": "LOOM_SCALAR_TYPE_INDEX",
    "i32": "LOOM_SCALAR_TYPE_I32",
    "i64": "LOOM_SCALAR_TYPE_I64",
    "f32": "LOOM_SCALAR_TYPE_F32",
    "f64": "LOOM_SCALAR_TYPE_F64",
}

_BUILD_FLAG_OPTIONAL_ATTR_TYPES = frozenset(
    {
        "i64",
        "f64",
        "string",
        "bool",
        "enum",
        "symbol",
        "type",
        "encoding",
    }
)


def _flatten_format(
    elements: tuple[FormatElement, ...],
) -> list[FormatElement]:
    """Recursively collects all format elements, flattening groups."""
    result: list[FormatElement] = []
    for element in elements:
        result.append(element)
        if isinstance(element, OptionalGroup | Scope | Clause):
            result.extend(_flatten_format(element.elements))
    return result


def _static_tied_results(op: Op) -> list[tuple[int, int]]:
    """Returns [(result_index, operand_index)] for statically-tied results.

    Statically-tied results are declared with TiedResult in the op definition,
    meaning the tie is part of the op's semantics and the builder emits it
    automatically without caller input.
    """
    ties: list[tuple[int, int]] = []
    operand_names = [o.name for o in op.operands]
    for i, result in enumerate(op.results):
        if isinstance(result, TiedResult):
            operand_index = operand_names.index(result.tied_to)
            ties.append((i, operand_index))
    return ties


def _c_attr_param_type(
    op: Op,
    attr_def: AttrDef,
    shared_enums: dict[int, tuple[str, str, EnumDef]],
) -> str:
    """Returns the C builder parameter type for an attribute."""
    if attr_def.attr_type == "enum" and attr_def.enum_def is not None and not attr_def.optional:
        return _enum_c_type(op, attr_def, shared_enums)
    return _C_ATTR_TYPE_MAP.get(attr_def.attr_type, "loom_attribute_t")


def _extract_c_params(op: Op, shared_enums: dict[int, tuple[str, str, EnumDef]]) -> list[dict[str, Any]]:
    """Extracts builder parameters for a C function from format specs.

    Mirrors the Python _extract_params but produces C-oriented descriptors.
    Each param has: name, kind, c_type, and kind-specific extras.
    """
    layout = compute_layout(op)
    params: list[dict[str, Any]] = []
    implicit_fields = {"iv", "args"}
    covered_attrs: set[str] = set()

    def append_attr_param(name: str) -> None:
        attr_def = op.attr(name)
        if attr_def is None:
            return
        c_type = _c_attr_param_type(op, attr_def, shared_enums)
        params.append(
            {
                "name": name,
                "kind": "attr",
                "c_type": c_type,
                "attr_type": attr_def.attr_type,
                "optional": attr_def.optional,
                "attr_index": _resolve_attr_index(op, name, "builder"),
            }
        )
        covered_attrs.add(name)

    # Collect field names that are anchors of OptionalGroups.
    optional_anchors: set[str] = set()

    def _collect_optional_anchors(elements: tuple[FormatElement, ...]) -> None:
        for element in elements:
            if isinstance(element, OptionalGroup):
                optional_anchors.add(element.anchor)
                _collect_optional_anchors(element.elements)
            elif isinstance(element, Scope | Clause):
                _collect_optional_anchors(element.elements)

    _collect_optional_anchors(op.format)

    # Ops with ResultType or ResultTypeList can have tied results, meaning
    # any operand may be consumed. Track for loom_may_consume annotation.
    has_result_type_list = any(isinstance(e, ResultType | ResultTypeList) for e in _flatten_format(op.format))

    # Static ties are declared in the op definition (TiedResult). Dynamic
    # ties require explicit builder parameters.
    static_ties = _static_tied_results(op)
    has_dynamic_ties = any(isinstance(e, ResultTypeList) for e in _flatten_format(op.format)) and not static_ties

    # Track the most recent BindingList for association with the next Region.
    _pending_binding: dict[str, Any] | None = None
    # Track whether FuncArgs was seen (entry block args come from arg_types).
    _pending_func_args: bool = False

    def _find_region_def(target_op: Op, region_name: str) -> RegionDef | None:
        for r in target_op.regions:
            if r.name == region_name:
                return r
        return None

    def walk(elements: tuple[FormatElement, ...]) -> None:
        nonlocal _pending_binding, _pending_func_args
        for element in elements:
            match element:
                case Ref(field=name):
                    if name in implicit_fields:
                        continue
                    desc = layout.fields.get(name)
                    if desc and desc.kind == FieldKind.OPERAND:
                        params.append(
                            {
                                "name": name,
                                "kind": "operand",
                                "c_type": "loom_value_id_t",
                                "index": desc.index,
                                "may_consume": has_result_type_list,
                            }
                        )

                case Refs(field=name) | TypedRefs(field=name):
                    params.append(
                        {
                            "name": name,
                            "kind": "operand_variadic",
                            "c_type": "const loom_value_id_t*",
                            "may_consume": has_result_type_list,
                        }
                    )

                case BlockRef(field=name):
                    desc = layout.fields.get(name)
                    if desc is None or desc.kind != FieldKind.SUCCESSOR:
                        kind_name = desc.kind.name if desc else "UNKNOWN"
                        raise ValueError(f"Op '{op.name}': BlockRef('{name}') references {kind_name}, expected SUCCESSOR")
                    params.append(
                        {
                            "name": name,
                            "kind": "successor",
                            "c_type": "loom_block_t*",
                            "index": desc.index,
                        }
                    )

                case OperandDict(operands=operand_field, names=names_field):
                    params.append(
                        {
                            "name": operand_field,
                            "kind": "operand_dict",
                            "c_type": "const loom_named_value_t*",
                            "operand_index": layout.fields[operand_field].index,
                            "names_attr_index": _resolve_attr_index(op, names_field, "builder"),
                            "names_field": names_field,
                            "may_consume": has_result_type_list,
                        }
                    )
                    covered_attrs.add(names_field)

                case AttrTable(keys=keys_field, values=values_field):
                    append_attr_param(keys_field)
                    values_desc = layout.fields.get(values_field)
                    if values_desc is None or values_desc.kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' is not an operand field")
                    if not values_desc.variadic:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' must be variadic")
                    params.append(
                        {
                            "name": values_field,
                            "kind": "operand_variadic",
                            "c_type": "const loom_value_id_t*",
                            "may_consume": has_result_type_list,
                            "index": values_desc.index,
                        }
                    )
                    covered_attrs.add(keys_field)

                case RegionTable(keys=keys_field, case_regions=case_regions_field, default_region=default_region_field):
                    append_attr_param(keys_field)
                    case_region_def = _find_region_def(op, case_regions_field)
                    default_region_def = _find_region_def(op, default_region_field)
                    params.append(
                        {
                            "name": "region_table",
                            "kind": "auto_region_table",
                            "case_region_index": layout.fields[case_regions_field].index,
                            "default_region_index": layout.fields[default_region_field].index,
                            "case_region_name": case_regions_field,
                            "default_region_name": default_region_field,
                            "keys_field": keys_field,
                            "case_implicit_args": (case_region_def.implicit_args if case_region_def else ()),
                            "default_implicit_args": (default_region_def.implicit_args if default_region_def else ()),
                        }
                    )
                    covered_attrs.add(keys_field)

                case Attr(field=name):
                    append_attr_param(name)

                case SymbolRef(field=name):
                    attr_def = op.attr(name)
                    params.append(
                        {
                            "name": name,
                            "kind": "symbol",
                            "c_type": "loom_symbol_ref_t",
                            "attr_index": _resolve_attr_index(op, name, "builder"),
                            "optional": (attr_def.optional if attr_def else False),
                        }
                    )
                    covered_attrs.add(name)

                case IndexList(dynamic=dynamic_field, static=static_field):
                    static_desc = layout.fields.get(static_field)
                    params.append(
                        {
                            "name": dynamic_field,
                            "kind": "index_list",
                            "dynamic_field": dynamic_field,
                            "static_field": static_field,
                            "static_attr_index": (_resolve_attr_index(op, static_field, "builder") if static_desc else 0),
                        }
                    )
                    covered_attrs.add(static_field)

                case BindingList(field=name, kind=binding_kind):
                    params.append(
                        {
                            "name": name,
                            "kind": "binding_list",
                            "c_type": "const loom_value_id_t*",
                            "binding_kind": binding_kind,
                            "may_consume": has_result_type_list,
                        }
                    )
                    # Track as pending binding for the next Region.
                    nonlocal _pending_binding
                    _pending_binding = {
                        "name": name,
                        "binding_kind": binding_kind,
                    }

                case BlockArgs():
                    pass

                case ResultType(field=name):
                    # When ResultType references a variadic result, the
                    # builder needs result_types/result_count (same as
                    # ResultTypeList) since the op can have multiple results.
                    field_desc = layout.fields.get(name)
                    is_variadic = field_desc and field_desc.kind == FieldKind.RESULT and layout.variadic_result
                    if is_variadic:
                        params.append(
                            {
                                "name": "result_types",
                                "kind": "result_types",
                            }
                        )
                    else:
                        params.append(
                            {
                                "name": "result_type",
                                "kind": "result_type",
                            }
                        )

                case ResultTypeList(field=name):
                    params.append(
                        {
                            "name": "result_types",
                            "kind": "result_types",
                        }
                    )
                    if has_dynamic_ties:
                        params.append(
                            {
                                "name": "tied_results",
                                "kind": "tied_results",
                            }
                        )

                case RegionFmt(field=name):
                    region_def = _find_region_def(op, name)
                    binding = _pending_binding
                    _pending_binding = None
                    func_args = _pending_func_args
                    _pending_func_args = False
                    params.append(
                        {
                            "name": name,
                            "kind": "auto_region",
                            "region_index": layout.fields[name].index,
                            "optional": (region_def.optional if region_def else False),
                            "binding": binding,
                            "arg_source": (region_def.arg_source if region_def else None),
                            "implicit_args": (region_def.implicit_args if region_def else ()),
                            "func_args": func_args,
                        }
                    )

                case Flags(field=name):
                    params.append(
                        {
                            "name": "instance_flags",
                            "kind": "instance_flags",
                            "c_type": "uint8_t",
                            "attr_name": name,
                        }
                    )
                    covered_attrs.add(name)

                case TemplateParam(field=name):
                    append_attr_param(name)

                case OptionalGroup(elements=inner, anchor=_anchor):
                    walk(inner)

                case Scope(elements=inner):
                    walk(inner)

                case Clause(elements=inner):
                    walk(inner)

                case FuncArgs():
                    params.append(
                        {
                            "name": "arg_types",
                            "kind": "func_args",
                        }
                    )
                    _pending_func_args = True

                case PredicateList(field=name):
                    attr_def = op.attr(name)
                    if attr_def is not None:
                        params.append(
                            {
                                "name": name,
                                "kind": "predicate_list",
                                "optional": attr_def.optional,
                                "attr_index": _resolve_attr_index(op, name, "builder"),
                            }
                        )
                        covered_attrs.add(name)

                case AttrDict(field=name):
                    if name:
                        append_attr_param(name)
                    else:
                        for attr_def in op.attrs:
                            if attr_def.attr_type == ATTR_TYPE_FLAGS:
                                continue
                            if attr_def.name in covered_attrs:
                                continue
                            append_attr_param(attr_def.name)

                case OpRef(field=name):
                    append_attr_param(name)

                case DescriptorRef(key=name, stable_id=stable_id):
                    attr_def = op.attr(name)
                    if attr_def is None:
                        continue
                    params.append(
                        {
                            "name": name,
                            "kind": "descriptor_ref",
                            "c_type": _c_attr_param_type(op, attr_def, shared_enums),
                            "attr_type": attr_def.attr_type,
                            "attr_index": _resolve_attr_index(op, name, "builder"),
                            "stable_id_attr_index": _resolve_attr_index(op, stable_id, "builder"),
                        }
                    )
                    covered_attrs.add(name)
                    covered_attrs.add(stable_id)

                case Keyword() | TypeOf() | TypesOf() | Glue():
                    pass

    walk(op.format)

    # If the op has results but no ResultType/ResultTypeList was encountered
    # in the format spec, we still need a result type parameter so the
    # builder can define result values.
    has_result_param = any(p["kind"] in ("result_type", "result_types") for p in params)
    if not has_result_param and len(op.results) > 0:
        if layout.variadic_result:
            params.append(
                {
                    "name": "result_types",
                    "kind": "result_types",
                }
            )
        else:
            params.append(
                {
                    "name": "result_types",
                    "kind": "result_types",
                }
            )

    return params


def _optional_param_uses_build_flag(param: dict[str, object]) -> bool:
    """Returns true if an optional builder param needs an explicit presence bit."""
    if not param.get("optional"):
        return False
    if param["kind"] == "symbol":
        return True
    if param["kind"] == "auto_region":
        return True
    if param["kind"] != "attr":
        return False
    return param.get("attr_type") in _BUILD_FLAG_OPTIONAL_ATTR_TYPES


def _build_flag_params(params: list[dict[str, object]]) -> list[dict[str, object]]:
    """Returns optional builder parameters controlled by build_flags."""
    return [param for param in params if _optional_param_uses_build_flag(param)]


def _build_flags_type_name(prefix: str) -> str:
    return f"{prefix}_build_flags_t"


def _build_flag_bit_name(prefix: str, param: dict[str, object]) -> str:
    return f"{prefix.upper()}_BUILD_FLAG_HAS_{str(param['name']).upper()}"


def _generate_build_flags_declaration(prefix: str, params: list[dict[str, object]]) -> list[str]:
    flag_params = _build_flag_params(params)
    if not flag_params:
        return []
    if len(flag_params) > 32:
        raise ValueError(f"{prefix}_build has {len(flag_params)} optional fields, which exceeds the 32-bit build flag capacity")

    lines = [f"enum {prefix}_build_flag_bits_e {{"]
    for index, param in enumerate(flag_params):
        lines.append(f"  {_build_flag_bit_name(prefix, param)} = 1u << {index},")
    lines.append("};")
    lines.append(f"typedef uint32_t {_build_flags_type_name(prefix)};")
    return lines


def _build_c_param_list(params: list[dict[str, object]], layout: FieldLayout, prefix: str) -> list[str]:
    """Builds the C parameter string list from extracted params.

    Adds loom_optional annotations for optional attrs and regions.
    """
    c_params = ["loom_builder_t* builder"]
    if _build_flag_params(params):
        c_params.append(f"{_build_flags_type_name(prefix)} build_flags")
    for param in params:
        opt = "loom_optional " if param.get("optional") else ""
        consume = "loom_may_consume " if param.get("may_consume") else ""
        match param["kind"]:
            case "operand":
                c_params.append(f"{consume}loom_value_id_t {param['name']}")
            case "successor":
                c_params.append(f"loom_block_t* {param['name']}")
            case "operand_variadic":
                c_params.append(f"{consume}const loom_value_id_t* {param['name']}")
                c_params.append(f"iree_host_size_t {param['name']}_count")
            case "operand_dict":
                c_params.append(f"{consume}const loom_named_value_t* {param['name']}")
                c_params.append(f"iree_host_size_t {param['name']}_count")
            case "attr":
                c_params.append(f"{opt}{param['c_type']} {param['name']}")
                if param["attr_type"] == "i64_array":
                    c_params.append(f"iree_host_size_t {param['name']}_count")
            case "descriptor_ref":
                c_params.append(f"{param['c_type']} {param['name']}")
            case "symbol":
                c_params.append(f"{opt}loom_symbol_ref_t {param['name']}")
            case "index_list":
                c_params.append(f"const loom_value_id_t* {param['dynamic_field']}")
                c_params.append(f"iree_host_size_t {param['dynamic_field']}_count")
                c_params.append(f"const int64_t* {param['static_field']}")
                c_params.append(f"iree_host_size_t {param['static_field']}_count")
            case "binding_list":
                c_params.append(f"{consume}const loom_value_id_t* {param['name']}")
                c_params.append(f"iree_host_size_t {param['name']}_count")
            case "result_type":
                c_params.append("loom_type_t result_type")
            case "result_types":
                if layout.variadic_result:
                    c_params.append("const loom_type_t* result_types")
                    c_params.append("iree_host_size_t result_count")
                else:
                    c_params.append("loom_type_t result_type")
            case "tied_results":
                c_params.append("const loom_tied_result_t* tied_results")
                c_params.append("iree_host_size_t tied_result_count")
            case "predicate_list":
                c_params.append(f"{opt}const loom_predicate_t* {param['name']}")
                c_params.append(f"iree_host_size_t {param['name']}_count")
            case "instance_flags":
                c_params.append(f"uint8_t {param['name']}")
            case "func_args":
                c_params.append(f"const loom_type_t* {param['name']}")
                c_params.append(f"iree_host_size_t {param['name']}_count")
            case "auto_region":
                pass  # Auto-created by builder, no parameter.
    c_params.append("loom_location_id_t location")
    c_params.append("loom_op_t** out_op")
    return c_params


def _generate_builder_declaration(op: Op, prefix: str, shared_enums: dict[int, tuple[str, str, EnumDef]]) -> list[str]:
    """Generates the C builder function declaration for a complex op."""
    params = _extract_c_params(op, shared_enums)
    layout = compute_layout(op)
    lines: list[str] = []
    c_params = _build_c_param_list(params, layout, prefix)

    # Format as multi-line declaration.
    lines.append(f"iree_status_t {prefix}_build(")
    for i, p in enumerate(c_params):
        comma = "," if i < len(c_params) - 1 else ");"
        lines.append(f"    {p}{comma}")

    return lines


def _emit_builder_count_check(
    lines: list[str],
    *,
    count: str,
    max_value: str,
    message: str,
) -> None:
    """Emits a C range check before narrowing a host-size builder count."""
    lines.append(f"  if ({count} > {max_value}) {{")
    lines.append("    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,")
    lines.append(f'                            "{message}");')
    lines.append("  }")


def _generate_builder_implementation(
    op: Op,
    prefix: str,
    enum_name: str,
    shared_enums: dict[int, tuple[str, str, EnumDef]],
) -> list[str]:
    """Generates the C builder function implementation for a complex op."""
    params = _extract_c_params(op, shared_enums)
    layout = compute_layout(op)
    func_args_are_operands = _func_args_are_operands(op)
    lines: list[str] = []

    # Compute counts for op_alloc.
    # Operands: fixed + variadic. For variadic, the count comes from a parameter.
    has_variadic_result = layout.variadic_result is not None

    # Figure out how to compute operand_count at runtime.
    fixed_operand_count = layout.fixed_operand_count
    variadic_operand_param = None
    for param in params:
        if param["kind"] in ("operand_variadic", "binding_list"):
            variadic_operand_param = param["name"]
            break
        if param["kind"] == "operand_dict":
            variadic_operand_param = param["name"]
            break
        if param["kind"] == "index_list":
            variadic_operand_param = param["dynamic_field"]
            break
        if param["kind"] == "func_args" and func_args_are_operands:
            variadic_operand_param = param["name"]
            break

    c_params = _build_c_param_list(params, layout, prefix)

    lines.append(f"iree_status_t {prefix}_build(")
    for i, p in enumerate(c_params):
        comma = "," if i < len(c_params) - 1 else ") {"
        lines.append(f"    {p}{comma}")

    # Validate all host-size counts before narrowing to the op storage width.
    if variadic_operand_param:
        max_variadic_operand_count = "UINT16_MAX"
        if fixed_operand_count:
            max_variadic_operand_count = f"UINT16_MAX - {fixed_operand_count}"
        _emit_builder_count_check(
            lines,
            count=f"{variadic_operand_param}_count",
            max_value=max_variadic_operand_count,
            message=f"{op.name} operand count exceeds uint16_t range",
        )
    if has_variadic_result:
        _emit_builder_count_check(
            lines,
            count="result_count",
            max_value="UINT16_MAX",
            message=f"{op.name} result count exceeds uint16_t range",
        )
    for param in params:
        if param["kind"] == "auto_region_table":
            _emit_builder_count_check(
                lines,
                count=f"{param['keys_field']}_count",
                max_value=f"UINT8_MAX - {layout.fixed_region_count}",
                message=f"{op.name} region count exceeds uint8_t range",
            )
    if any(param["kind"] == "tied_results" for param in params):
        _emit_builder_count_check(
            lines,
            count="tied_result_count",
            max_value="UINT16_MAX",
            message=f"{op.name} tied result count exceeds uint16_t range",
        )
    for param in params:
        if param["kind"] == "index_list":
            _emit_builder_count_check(
                lines,
                count=f"{param['static_field']}_count",
                max_value="UINT16_MAX",
                message=f"{op.name} static index count exceeds uint16_t range",
            )
        elif param["kind"] == "predicate_list":
            _emit_builder_count_check(
                lines,
                count=f"{param['name']}_count",
                max_value="UINT16_MAX",
                message=f"{op.name} predicate count exceeds uint16_t range",
            )
        elif param["kind"] == "attr" and param["attr_type"] == "i64_array":
            _emit_builder_count_check(
                lines,
                count=f"{param['name']}_count",
                max_value="UINT16_MAX",
                message=f"{op.name} i64 array attribute count exceeds uint16_t range",
            )

    # Compute operand count expression.
    if variadic_operand_param:
        operand_count_expr = f"{fixed_operand_count} + (uint16_t){variadic_operand_param}_count"
    else:
        operand_count_expr = str(fixed_operand_count)

    # Compute result count expression.
    if has_variadic_result:
        result_count_expr = "(uint16_t)result_count"
    else:
        result_count_expr = str(layout.fixed_result_count)

    successor_count_expr = str(len(op.successors))

    region_count_expr = str(len(op.regions))
    for param in params:
        if param["kind"] == "auto_region_table":
            region_count_expr = f"{layout.fixed_region_count} + (uint8_t){param['keys_field']}_count"
            break
    optional_auto_regions = [param for param in params if param["kind"] == "auto_region" and param.get("optional")]
    if optional_auto_regions:
        optional_auto_regions.sort(key=lambda param: param["region_index"])
        lines.append(f"  uint8_t region_count = {layout.required_region_count};")
        previous_optional_flag = ""
        for param in optional_auto_regions:
            optional_flag = _build_flag_bit_name(prefix, param)
            if previous_optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag}) &&")
                lines.append(f"      !iree_any_bit_set(build_flags, {previous_optional_flag})) {{")
                lines.append("    return iree_make_status(")
                lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
                lines.append(f'        "{op.name} optional region {param["name"]} requires preceding optional region");')
                lines.append("  }")
            lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
            lines.append(f"    region_count = {param['region_index'] + 1};")
            lines.append("  }")
            previous_optional_flag = optional_flag
        region_count_expr = "region_count"
    attr_count = len(_non_flags_attrs(op))

    # Compute tied result count expression.
    static_ties = _static_tied_results(op)
    has_tied_param = any(p["kind"] == "tied_results" for p in params)
    if static_ties:
        tied_count_expr = str(len(static_ties))
    elif has_tied_param:
        tied_count_expr = "(uint16_t)tied_result_count"
    else:
        tied_count_expr = "0"

    allocate_fn = "loom_builder_allocate_op_with_successors" if op.successors else "loom_builder_allocate_op"
    lines.append(f"  IREE_RETURN_IF_ERROR({allocate_fn}(")
    lines.append(f"      builder, {enum_name}, {operand_count_expr},")
    if op.successors:
        lines.append(f"      {result_count_expr}, {successor_count_expr}, {region_count_expr}, {tied_count_expr},")
    else:
        lines.append(f"      {result_count_expr}, {region_count_expr}, {tied_count_expr},")
    lines.append(f"      {attr_count}, location, out_op));")
    lines.extend(f"  (*out_op)->instance_flags = {param['name']};" for param in params if param["kind"] == "instance_flags")

    # Fill in fixed operands.
    lines.extend(f"  loom_op_operands(*out_op)[{param['index']}] = {param['name']};" for param in params if param["kind"] == "operand")

    # Fill in fixed successors.
    lines.extend(f"  loom_op_successors(*out_op)[{param['index']}] = {param['name']};" for param in params if param["kind"] == "successor")

    # Fill in variadic operands (memcpy from the array parameter).
    for param in params:
        if param["kind"] in ("operand_variadic", "binding_list"):
            name = param["name"]
            lines.append(f"  if ({name}_count > 0) {{")
            lines.append(f"    memcpy(loom_op_operands(*out_op) + {fixed_operand_count},")
            lines.append(f"           {name}, {name}_count * sizeof(loom_value_id_t));")
            lines.append("  }")
        elif param["kind"] == "operand_dict":
            name = param["name"]
            operand_index = param["operand_index"]
            attr_index = param["names_attr_index"]
            lines.append("  IREE_RETURN_IF_ERROR(loom_builder_set_operand_dict(")
            lines.append(f"      builder, loom_make_named_value_slice({name}, {name}_count),")
            lines.append(f"      loom_op_operands(*out_op) + {operand_index},")
            lines.append(f"      &loom_op_attrs(*out_op)[{attr_index}]));")
        elif param["kind"] == "index_list":
            dyn = param["dynamic_field"]
            lines.append(f"  if ({dyn}_count > 0) {{")
            lines.append(f"    memcpy(loom_op_operands(*out_op) + {fixed_operand_count},")
            lines.append(f"           {dyn}, {dyn}_count * sizeof(loom_value_id_t));")
            lines.append("  }")
        elif param["kind"] == "func_args" and func_args_are_operands:
            lines.append("  for (iree_host_size_t _i = 0; _i < arg_types_count; ++_i) {")
            lines.append("    loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append("    IREE_RETURN_IF_ERROR(")
            lines.append("        loom_builder_define_value(builder, arg_types[_i], &_arg_id));")
            lines.append(f"    loom_op_operands(*out_op)[{fixed_operand_count} + _i] = _arg_id;")
            lines.append("  }")

    def emit_auto_region(slot_expr: str, name: str, implicit_args: tuple[tuple[str, str], ...]) -> None:
        has_block_args = bool(implicit_args)
        lines.append(f"  // Auto-create {name} region with entry block.")
        lines.append("  {")
        lines.append("    loom_region_t* _region = NULL;")
        lines.append("    IREE_RETURN_IF_ERROR(")
        lines.append("        loom_module_allocate_region(builder->module, 1, &_region));")
        if has_block_args:
            lines.append("    loom_block_t* _block = loom_region_entry_block(_region);")
        for _arg_name, arg_type_kw in implicit_args:
            scalar_type = _IMPLICIT_ARG_TYPE_MAP.get(arg_type_kw)
            if scalar_type is None:
                raise ValueError(f"Unknown implicit arg type: {arg_type_kw}")
            lines.append("    {")
            lines.append("      loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append("      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
            lines.append("          builder, _block,")
            lines.append(f"          loom_type_scalar({scalar_type}), &_arg_id));")
            lines.append("    }")
        lines.append(f"    loom_op_regions(*out_op)[{slot_expr}] = _region;")
        lines.append("  }")

    # Auto-create regions with typed entry block args.
    for param in params:
        if param["kind"] != "auto_region":
            continue
        idx = param["region_index"]
        name = param["name"]
        optional_region = bool(param.get("optional"))
        region_indent = "    " if optional_region else "  "
        inner_indent = region_indent + "  "
        if optional_region:
            optional_flag = _build_flag_bit_name(prefix, param)
            lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
        has_block_args = bool(param.get("implicit_args")) or bool(param.get("binding")) or bool(param.get("arg_source")) or bool(param.get("func_args"))
        lines.append(f"{region_indent}// Auto-create {name} region with entry block.")
        lines.append(f"{region_indent}{{")
        lines.append(f"{inner_indent}loom_region_t* _region = NULL;")
        lines.append(f"{inner_indent}IREE_RETURN_IF_ERROR(")
        lines.append(f"{inner_indent}    loom_module_allocate_region(builder->module, 1, &_region));")
        if has_block_args:
            lines.append(f"{inner_indent}loom_block_t* _block = loom_region_entry_block(_region);")

        # Implicit args (e.g., loop IV).
        for _arg_name, arg_type_kw in param.get("implicit_args", ()):
            scalar_type = _IMPLICIT_ARG_TYPE_MAP.get(arg_type_kw)
            if scalar_type is None:
                raise ValueError(f"Unknown implicit arg type: {arg_type_kw}")
            lines.append(f"{inner_indent}{{")
            lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
            lines.append(f"{inner_indent}      builder, _block,")
            lines.append(f"{inner_indent}      loom_type_scalar({scalar_type}), &_arg_id));")
            lines.append(f"{inner_indent}}}")

        # Binding list args (capture or element).
        binding = param.get("binding")
        if binding:
            binding_name = binding["name"]
            binding_kind = binding["binding_kind"]
            if binding_kind == "capture":
                lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < {binding_name}_count; ++_i) {{")
                lines.append(f"{inner_indent}  loom_type_t _arg_type =")
                lines.append(f"{inner_indent}      loom_module_value_type(builder->module, {binding_name}[_i]);")
                lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
                lines.append(f"{inner_indent}      builder, _block, _arg_type, &_arg_id));")
                lines.append(f"{inner_indent}}}")
            elif binding_kind == "element":
                lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < {binding_name}_count; ++_i) {{")
                lines.append(f"{inner_indent}  loom_type_t _operand_type =")
                lines.append(f"{inner_indent}      loom_module_value_type(builder->module, {binding_name}[_i]);")
                lines.append(f"{inner_indent}  loom_type_t _arg_type =")
                lines.append(f"{inner_indent}      loom_type_scalar(loom_type_element_type(_operand_type));")
                lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
                lines.append(f"{inner_indent}      builder, _block, _arg_type, &_arg_id));")
                lines.append(f"{inner_indent}}}")

        # Region args sourced from an existing value field. This is used for
        # regions whose entry args are semantically linked to a terminator or
        # loop edge, while the builder can still seed them from the op's input
        # values before the body is populated.
        arg_source = param.get("arg_source")
        if arg_source and not binding:
            lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < {arg_source}_count; ++_i) {{")
            lines.append(f"{inner_indent}  loom_type_t _arg_type =")
            lines.append(f"{inner_indent}      loom_module_value_type(builder->module, {arg_source}[_i]);")
            lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
            lines.append(f"{inner_indent}      builder, _block, _arg_type, &_arg_id));")
            lines.append(f"{inner_indent}}}")

        # FuncArgs: entry block args typed from the arg_types parameter.
        if param.get("func_args") and not binding:
            lines.append(f"{inner_indent}for (iree_host_size_t _i = 0; _i < arg_types_count; ++_i) {{")
            lines.append(f"{inner_indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
            lines.append(f"{inner_indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
            lines.append(f"{inner_indent}      builder, _block, arg_types[_i], &_arg_id));")
            lines.append(f"{inner_indent}}}")

        lines.append(f"{inner_indent}loom_op_regions(*out_op)[{idx}] = _region;")
        lines.append(f"{region_indent}}}")
        if optional_region:
            lines.append("  }")

    for param in params:
        if param["kind"] != "auto_region_table":
            continue
        emit_auto_region(
            str(param["default_region_index"]),
            param["default_region_name"],
            param["default_implicit_args"],
        )
        lines.append(f"  for (iree_host_size_t _case = 0; _case < {param['keys_field']}_count; ++_case) {{")
        lines.append("    loom_region_t* _region = NULL;")
        lines.append("    IREE_RETURN_IF_ERROR(")
        lines.append("        loom_module_allocate_region(builder->module, 1, &_region));")
        if param["case_implicit_args"]:
            lines.append("    loom_block_t* _block = loom_region_entry_block(_region);")
            for _arg_name, arg_type_kw in param["case_implicit_args"]:
                scalar_type = _IMPLICIT_ARG_TYPE_MAP.get(arg_type_kw)
                if scalar_type is None:
                    raise ValueError(f"Unknown implicit arg type: {arg_type_kw}")
                lines.append("    {")
                lines.append("      loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                lines.append("      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
                lines.append("          builder, _block,")
                lines.append(f"          loom_type_scalar({scalar_type}), &_arg_id));")
                lines.append("    }")
        lines.append(f"    loom_op_regions(*out_op)[{param['case_region_index']} + _case] = _region;")
        lines.append("  }")

    # Fill in attributes.
    for param in params:
        if param["kind"] == "predicate_list":
            idx = param["attr_index"]
            name = param["name"]
            if param.get("optional"):
                lines.append(f"  if ({name}_count > 0) {{")
                lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_predicate_list((loom_predicate_t*){name}, (uint16_t){name}_count);")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_predicate_list((loom_predicate_t*){name}, (uint16_t){name}_count);")
        elif param["kind"] == "index_list":
            static_field = param["static_field"]
            idx = param["static_attr_index"]
            lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array((int64_t*){static_field}, (uint16_t){static_field}_count);")
        elif param["kind"] == "symbol":
            idx = param["attr_index"]
            optional_flag = _build_flag_bit_name(prefix, param) if _optional_param_uses_build_flag(param) else ""
            if optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_symbol({param['name']});")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_symbol({param['name']});")
        elif param["kind"] == "attr":
            idx = param["attr_index"]
            attr_type = param["attr_type"]
            name = param["name"]
            is_optional = param.get("optional", False)
            constructor_map = {
                "i64": f"loom_attr_i64({name})",
                "f64": f"loom_attr_f64({name})",
                "string": f"loom_attr_string({name})",
                "bool": f"loom_attr_bool({name})",
                "enum": f"loom_attr_enum({name})",
                "symbol": f"loom_attr_symbol({name})",
                "type": f"loom_attr_type({name})",
                "encoding": f"loom_attr_encoding({name})",
                "any": name,
            }
            constructor = constructor_map.get(attr_type, name)
            optional_flag = _build_flag_bit_name(prefix, param) if _optional_param_uses_build_flag(param) else ""
            if attr_type == "i64_array":
                if is_optional:
                    lines.append(f"  if ({name}_count > 0) {{")
                    lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array((int64_t*){name}, (uint16_t){name}_count);")
                    lines.append("  }")
                else:
                    lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array((int64_t*){name}, (uint16_t){name}_count);")
            elif attr_type == "dict":
                if is_optional:
                    lines.append(f"  if ({name}.count > 0) {{")
                    lines.append("    IREE_RETURN_IF_ERROR(")
                    lines.append("        loom_module_make_canonical_attr_dict(")
                    lines.append(f"            builder->module, {name},")
                    lines.append(f"            &loom_op_attrs(*out_op)[{idx}]));")
                    lines.append("  }")
                else:
                    lines.append("  IREE_RETURN_IF_ERROR(")
                    lines.append("      loom_module_make_canonical_attr_dict(")
                    lines.append(f"          builder->module, {name},")
                    lines.append(f"          &loom_op_attrs(*out_op)[{idx}]));")
            elif optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_attrs(*out_op)[{idx}] = {constructor};")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_attrs(*out_op)[{idx}] = {constructor};")
        elif param["kind"] == "descriptor_ref":
            idx = param["attr_index"]
            stable_id_idx = param["stable_id_attr_index"]
            name = param["name"]
            lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_string({name});")
            lines.append(f"  loom_op_attrs(*out_op)[{stable_id_idx}] = loom_attr_i64((int64_t)")
            lines.append("      loom_stable_id_from_string(")
            lines.append(f"          builder->module->strings.entries[{name}]));")

    # Define result values in the module's value table.
    for param in params:
        if param["kind"] == "result_type":
            lines.append("  {")
            lines.append("    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;")
            lines.append("    IREE_RETURN_IF_ERROR(")
            lines.append("        loom_builder_define_value(builder, result_type, &_result_id));")
            lines.append("    loom_op_results(*out_op)[0] = _result_id;")
            lines.append("  }")
        elif param["kind"] == "result_types":
            if has_variadic_result:
                lines.append("  for (iree_host_size_t _r = 0; _r < result_count; ++_r) {")
                lines.append("    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;")
                lines.append("    IREE_RETURN_IF_ERROR(")
                lines.append("        loom_builder_define_value(builder, result_types[_r], &_result_id));")
                lines.append("    loom_op_results(*out_op)[_r] = _result_id;")
                lines.append("  }")
            else:
                lines.append("  {")
                lines.append("    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;")
                lines.append("    IREE_RETURN_IF_ERROR(")
                lines.append("        loom_builder_define_value(builder, result_type, &_result_id));")
                lines.append("    loom_op_results(*out_op)[0] = _result_id;")
                lines.append("  }")

    # Populate tied result metadata.
    if static_ties:
        for tie_index, (result_idx, operand_idx) in enumerate(static_ties):
            lines.append(f"  loom_op_tied_results(*out_op)[{tie_index}] = (loom_tied_result_t){{.result_index = {result_idx}, .operand_index = {operand_idx}, .has_type_change = true}};")
    elif has_tied_param:
        lines.append("  if (tied_result_count > 0) {")
        lines.append("    memcpy(loom_op_tied_results(*out_op), tied_results,")
        lines.append("           tied_result_count * sizeof(loom_tied_result_t));")
        lines.append("  }")

    lines.append("  return loom_builder_finalize_op(builder, *out_op);")
    lines.append("}")

    return lines


# ============================================================================
# B-string encoding
# ============================================================================


def _c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _bstring_expr(value: str) -> str:
    if len(value.encode()) > 255:
        raise ValueError(f"B-string '{value}' exceeds 255 bytes")
    return f'_BSTRING({len(value.encode())}, "{_c_string_literal(value)}")'


def _op_name_expr(value: str) -> str:
    value_length = len(value.encode())
    namespace_length = len(value.rsplit(".", 1)[0].encode()) if "." in value else 0
    if value_length > 255:
        raise ValueError(f"op name '{value}' exceeds 255 bytes")
    if namespace_length > 255:
        raise ValueError(f"op namespace '{value}' exceeds 255 bytes")
    return f'_OP_NAME({value_length}, {namespace_length}, "{_c_string_literal(value)}")'


def _emit_table_string_macros(lines: list[str], _dialect_name: str) -> None:
    lines.append("#define _BSTRING(length, value) LOOM_BSTRING_REF(length, value)")
    lines.append("#define _OP_NAME(length, namespace_length, value) \\")
    lines.append("  LOOM_OP_NAME_REF(length, namespace_length, value)")
    lines.append("")


# ============================================================================
# ops.h generation
# ============================================================================


def generate_ops_h(dialect_name: str, dialect_id: int, ops: Sequence[Op]) -> str:
    """Generates the ops.h header for a dialect."""
    lines: list[str] = []
    guard = _guard_name(dialect_name)
    dialect_enum = _c_dialect_enum(dialect_name)
    shared_enums = _collect_shared_enums(dialect_name, ops)

    lines.append(COPYRIGHT)
    lines.extend(
        line_comment_header(
            "//",
            generator="loom.gen.c_tables",
            regenerate="python3 loom/py/loom/gen/run.py c_tables",
        )
    )
    lines.append("// clang-format off")
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append('#include "loom/ops/op_defs.h"')
    enum_includes = sorted(
        {attr_def.enum_def.c_include for op in ops for attr_def in op.attrs if attr_def.attr_type == "enum" and attr_def.enum_def is not None and attr_def.enum_def.c_include is not None}
    )
    lines.extend(f'#include "{include}"' for include in enum_includes)
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")

    # Op kind enum.
    lines.append("enum {")
    for i, op in enumerate(ops):
        enum_name = _c_enum_name(op)
        lines.append(f"  {enum_name} = LOOM_OP_KIND({dialect_enum}, {i}),")
    lines.append(f"  LOOM_OP_{dialect_name.upper()}_COUNT_ = {len(ops)},")
    lines.append("};")

    lines.append("")

    # Flag bit defines — emitted once per unique flags enum.
    emitted_flag_enums: set[str] = set()
    for op in ops:
        for attr_def in op.attrs:
            if attr_def.attr_type != ATTR_TYPE_FLAGS or attr_def.enum_def is None:
                continue
            if attr_def.enum_def.name in emitted_flag_enums:
                continue
            emitted_flag_enums.add(attr_def.enum_def.name)
            enum_prefix = "LOOM_" + op.namespace.upper() + "_" + attr_def.enum_def.name.upper()
            if attr_def.enum_def.doc:
                lines.append(f"// {attr_def.enum_def.doc}")
            lines.extend(f"#define {enum_prefix}_{_enum_case_c_ident(case.keyword)} ((uint8_t){case.value})" for case in attr_def.enum_def.cases)
            lines.append("")

    # Enum attr C enums. When the same EnumDef object is shared by
    # multiple ops (e.g., CallingConv used by func.def, func.decl,
    # func.template, func.ukernel), emit it once with a dialect-level
    # name (loom_func_cc_t) instead of duplicating per-op.
    # Emit shared enums first.
    for c_prefix, const_prefix, enum_def in shared_enums.values():
        if enum_def.doc:
            lines.append(f"// {enum_def.doc}")
        lines.append(f"typedef enum {c_prefix}_e {{")
        lines.extend(f"  {const_prefix}_{_enum_case_c_ident(case.keyword)} = {case.value}," for case in enum_def.cases)
        max_value = max(c.value for c in enum_def.cases)
        lines.append(f"  {const_prefix}_COUNT_ = {max_value + 1},")
        lines.append(f"}} {c_prefix}_t;")
        lines.append("")

    # Emit per-op enums (only for EnumDefs not already emitted as shared).
    emitted_enum_defs: set[str] = set()
    for op in ops:
        for attr_def in op.attrs:
            if attr_def.attr_type != "enum" or attr_def.enum_def is None:
                continue
            if attr_def.enum_def.c_type is not None:
                continue
            if id(attr_def.enum_def) in shared_enums:
                continue
            key = f"{op.name}:{attr_def.name}"
            if key in emitted_enum_defs:
                continue
            emitted_enum_defs.add(key)
            c_prefix = _c_prefix(op) + "_" + attr_def.name
            enum_tag = c_prefix + "_e"
            const_prefix = c_prefix.upper()
            if attr_def.enum_def.doc:
                lines.append(f"// {attr_def.enum_def.doc}")
            lines.append(f"typedef enum {enum_tag} {{")
            lines.extend(f"  {const_prefix}_{_enum_case_c_ident(case.keyword)} = {case.value}," for case in attr_def.enum_def.cases)
            max_value = max(c.value for c in attr_def.enum_def.cases)
            lines.append(f"  {const_prefix}_COUNT_ = {max_value + 1},")
            lines.append(f"}} {c_prefix}_t;")
            lines.append("")

    # Per-op sections.
    emitted_canonicalize_declarations: set[str] = set()
    emitted_type_transfer_declarations: set[str] = set()
    for op in ops:
        prefix = _c_prefix(op)
        enum_name = _c_enum_name(op)
        layout = compute_layout(op)

        # Assembly format comment from first example, or synthesized.
        if op.examples:
            asm_fmt_lines = op.examples[0].split("\n")
        else:
            asm_fmt_lines = [op.name]
        doc = op.doc or f"{op.name} operation."

        lines.append(f"// {enum_name}: {doc}")
        lines.extend(f"// {line}" for line in asm_fmt_lines)

        # ISA check.
        lines.append(f"LOOM_DEFINE_ISA({prefix}_isa, {enum_name})")

        # Accessors.
        for operand in op.operands:
            desc = layout.fields[operand.name]
            if operand.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_OPERANDS({prefix}_{operand.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_OPERAND({prefix}_{operand.name}, {desc.index})")

        for result in op.results:
            desc = layout.fields[result.name]
            if result.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_RESULTS({prefix}_{result.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_RESULT({prefix}_{result.name}, {desc.index})")

        for successor in op.successors:
            desc = layout.fields[successor.name]
            if successor.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_SUCCESSORS({prefix}_{successor.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_SUCCESSOR({prefix}_{successor.name}, {desc.index})")

        # Regular attribute accessors (excludes flags attrs).
        non_flags_index = 0
        for attr_def in op.attrs:
            if attr_def.attr_type == ATTR_TYPE_FLAGS:
                lines.append(f"LOOM_DEFINE_INSTANCE_FLAGS({prefix}_{attr_def.name})")
                continue
            desc_index = non_flags_index
            non_flags_index += 1
            macro_map = {
                "i64": "LOOM_DEFINE_ATTR_I64",
                "f64": "LOOM_DEFINE_ATTR_F64",
                "string": "LOOM_DEFINE_ATTR_STRING",
                "bool": "LOOM_DEFINE_ATTR_BOOL",
                "i64_array": "LOOM_DEFINE_ATTR_I64_ARRAY",
                "dict": "LOOM_DEFINE_ATTR_DICT",
                "encoding": "LOOM_DEFINE_ATTR_ENCODING",
                "enum": "LOOM_DEFINE_ATTR_ENUM",
                "symbol": "LOOM_DEFINE_ATTR_SYMBOL",
                "type": "LOOM_DEFINE_ATTR_TYPE",
                "any": "LOOM_DEFINE_ATTR_ANY",
            }
            macro = macro_map.get(attr_def.attr_type)
            if attr_def.attr_type == "enum" and attr_def.enum_def:
                enum_type = _enum_c_type(op, attr_def, shared_enums)
                lines.append(f"LOOM_DEFINE_ATTR_ENUM_TYPED({prefix}_{attr_def.name}, {desc_index}, {enum_type})")
            elif macro:
                lines.append(f"{macro}({prefix}_{attr_def.name}, {desc_index})")

        for region_def in op.regions:
            desc = layout.fields[region_def.name]
            if region_def.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_REGIONS({prefix}_{region_def.name}, {desc.index})")
            elif region_def.optional:
                lines.append(f"LOOM_DEFINE_OPTIONAL_REGION({prefix}_{region_def.name}, {desc.index})")
            else:
                lines.append(f"LOOM_DEFINE_REGION({prefix}_{region_def.name}, {desc.index})")

        # Builder declaration.
        build_params = _extract_c_params(op, shared_enums)
        build_flags_declaration = _generate_build_flags_declaration(prefix, build_params)
        if build_flags_declaration:
            lines.extend(build_flags_declaration)
        pattern = _detect_builder_pattern(op)
        if pattern == "BINARY":
            lines.append(f"iree_status_t {prefix}_build(")
            lines.append("    loom_builder_t* builder, loom_value_id_t lhs,")
            lines.append("    loom_value_id_t rhs, loom_type_t result_type,")
            lines.append("    loom_location_id_t location, loom_op_t** out_op);")
        elif pattern == "BINARY_WITH_FLAGS":
            lines.append(f"iree_status_t {prefix}_build(")
            lines.append("    loom_builder_t* builder, uint8_t instance_flags,")
            lines.append("    loom_value_id_t lhs, loom_value_id_t rhs,")
            lines.append("    loom_type_t result_type, loom_location_id_t location,")
            lines.append("    loom_op_t** out_op);")
        elif pattern == "UNARY":
            lines.append(f"iree_status_t {prefix}_build(")
            lines.append("    loom_builder_t* builder, loom_value_id_t input,")
            lines.append("    loom_type_t result_type, loom_location_id_t location,")
            lines.append("    loom_op_t** out_op);")
        elif pattern == "UNARY_WITH_FLAGS":
            lines.append(f"iree_status_t {prefix}_build(")
            lines.append("    loom_builder_t* builder, uint8_t instance_flags,")
            lines.append("    loom_value_id_t input, loom_type_t result_type,")
            lines.append("    loom_location_id_t location, loom_op_t** out_op);")
        elif pattern == "CAST":
            lines.append(f"iree_status_t {prefix}_build(")
            lines.append("    loom_builder_t* builder, loom_value_id_t input,")
            lines.append("    loom_type_t input_type, loom_type_t result_type,")
            lines.append("    loom_location_id_t location, loom_op_t** out_op);")
        elif pattern == "COMPARISON":
            lines.append(f"iree_status_t {prefix}_build(")
            lines.append("    loom_builder_t* builder, uint8_t predicate,")
            lines.append("    loom_value_id_t lhs, loom_value_id_t rhs,")
            lines.append("    loom_type_t operand_type, loom_type_t result_type,")
            lines.append("    loom_location_id_t location, loom_op_t** out_op);")
        elif pattern == "COMPARISON_WITH_FLAGS":
            lines.append(f"iree_status_t {prefix}_build(")
            lines.append("    loom_builder_t* builder, uint8_t instance_flags,")
            lines.append("    uint8_t predicate, loom_value_id_t lhs,")
            lines.append("    loom_value_id_t rhs, loom_type_t operand_type,")
            lines.append("    loom_type_t result_type, loom_location_id_t location,")
            lines.append("    loom_op_t** out_op);")
        else:
            lines.extend(_generate_builder_declaration(op, prefix, shared_enums))

        # Canonicalize function declaration (hand-written, linked in).
        if op.canonicalize and op.canonicalize not in emitted_canonicalize_declarations:
            lines.append(f"iree_status_t {op.canonicalize}(loom_op_t* op, loom_rewriter_t* rewriter);")
            emitted_canonicalize_declarations.add(op.canonicalize)

        # Effective traits function declaration (hand-written, linked in).
        if op.effective_traits:
            lines.append(f"loom_trait_flags_t {op.effective_traits}(const loom_op_t* op);")

        # Fact inference function declaration (hand-written, linked in).
        if op.facts:
            lines.append(f"iree_status_t {op.facts}(")
            lines.append("    loom_fact_context_t* context,")
            lines.append("    const loom_module_t* module, const loom_op_t* op,")
            lines.append("    const loom_value_facts_t* operand_facts,")
            lines.append("    loom_value_facts_t* result_facts);")

        # Semantic type-transfer function declaration (hand-written, linked in).
        if op.type_transfer and op.type_transfer not in emitted_type_transfer_declarations:
            lines.append(f"iree_status_t {op.type_transfer}(")
            lines.append("    loom_type_transfer_context_t* context,")
            lines.append("    const loom_module_t* module, loom_op_t* op);")
            emitted_type_transfer_declarations.add(op.type_transfer)

        # Verify function declaration (hand-written, linked in).
        if op.verify:
            lines.append(f"iree_status_t {op.verify}(")
            lines.append("    const loom_module_t* module, const loom_op_t* op,")
            lines.append("    iree_diagnostic_emitter_t emitter);")

        lines.append("")

    # Registration function.
    lines.append(f"// Returns the vtable array for the {dialect_name} dialect.")
    lines.append(f"const loom_op_vtable_t* const* loom_{dialect_name}_dialect_vtables(")
    lines.append("    iree_host_size_t* out_count);")
    lines.append("")

    lines.append(f"// Returns the dense semantic metadata array for the {dialect_name} dialect.")
    lines.append(f"const loom_op_semantics_t* loom_{dialect_name}_dialect_op_semantics(")
    lines.append("    iree_host_size_t* out_count);")
    lines.append("")
    lines.append(f"// Returns semantic metadata for a {dialect_name} op kind, or empty metadata.")
    lines.append(f"loom_op_semantics_t loom_{dialect_name}_op_semantics(")
    lines.append("    loom_op_kind_t kind);")
    lines.append("")

    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# tables.c generation
# ============================================================================


def generate_tables_c(
    dialect_name: str,
    dialect_id: int,
    ops: Sequence[Op],
    *,
    include_path: str | None = None,
    emit_registration: bool = True,
    export_vtables: bool = False,
    private_header: bool = False,
) -> str:
    """Generates the tables.c file for a dialect (.rodata)."""
    lines: list[str] = []
    ops_by_name = {op.name: op for op in ops}

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    if private_header:
        lines.append(f'#include "{include_path}/tables.h"')
    else:
        lines.append(f'#include "{include_path}/ops.h"')
    if _target_like_bundle_table_symbols(ops):
        lines.append("")
        lines.append("#include <stddef.h>")
        lines.append("")
        lines.append('#include "loom/target/types.h"')
    lines.append('#include "loom/error/error_defs.h"')
    lines.append("")
    if not private_header:
        _emit_table_string_macros(lines, dialect_name)

    # Canonicalize functions are declared in ops.h (not here) so there
    # are no extern declarations in .c files.
    shared_enums = _collect_shared_enums(dialect_name, ops)

    def _emit_enum_case_names(lines: list[str], array_name: str, enum_def: EnumDef) -> None:
        cases_by_value = sorted(enum_def.cases, key=lambda c: c.value)
        max_value = max(c.value for c in cases_by_value)
        value_to_name: dict[int, str] = {c.value: c.keyword for c in cases_by_value}
        lines.append(f"static const loom_bstring_t {array_name}[] = {{")
        for v in range(max_value + 1):
            name = value_to_name.get(v)
            if name is not None:
                lines.append(f"    {_bstring_expr(name)},")
            else:
                lines.append("    NULL,")
        lines.append("};")

    # Symbol definition descriptors may refer to fact domains outside this
    # generated translation unit.
    symbol_fact_domain_symbols = sorted({fact_domain for op in ops if op.symbol_def is not None if (fact_domain := _symbol_fact_domain_symbol(op)) is not None})
    if symbol_fact_domain_symbols:
        lines.extend(f"extern const loom_symbol_fact_domain_t {fact_domain};" for fact_domain in symbol_fact_domain_symbols)
        lines.append("")
    interface_c_ptr_symbols = _interface_c_ptr_symbols(ops)
    if interface_c_ptr_symbols:
        lines.extend(f"extern const {c_type} {symbol};" for c_type, symbol in interface_c_ptr_symbols)
        lines.append("")
    target_like_bundle_table_symbols = _target_like_bundle_table_symbols(ops)
    if target_like_bundle_table_symbols:
        lines.extend(f"extern const loom_target_bundle_table_t {symbol};" for symbol in target_like_bundle_table_symbols)
        lines.append("")

    emitted_enum_case_name_arrays: set[str] = set()

    # Op metadata blocks.
    for op in ops:
        prefix = _c_prefix(op)
        layout = compute_layout(op)
        elements = _translate_format_elements(op)
        non_flags = _non_flags_attrs(op)
        has_flags = _has_flags_attr(op)

        # Format element array.
        if elements:
            lines.append(f"static const loom_format_element_t {prefix}_format[] = {{")
            for kind, field_index, data in elements:
                lines.append(f"    {{{kind}, {field_index}, {data}}},")
            lines.append("};")

        # Operand descriptors.
        func_args_are_operands = _func_args_are_operands(op)
        explicit_func_args_operand = _explicit_func_args_operand(op)
        synthesize_func_args_operand = func_args_are_operands and explicit_func_args_operand is None
        if op.operands or synthesize_func_args_operand:
            func_args_name = _func_args_field_name(op) if synthesize_func_args_operand else ""
            effect_map = {effect.operand: effect.kind for effect in op.effects}
            lines.append(f"static const loom_operand_descriptor_t {prefix}_operand_desc[] = {{")
            for operand in op.operands:
                type_constraint = TYPE_CONSTRAINT_MAP[operand.type_constraint]
                flags_parts = []
                if operand.variadic:
                    flags_parts.append("LOOM_OPERAND_VARIADIC")
                if operand.optional:
                    flags_parts.append("LOOM_OPERAND_OPTIONAL")
                effect_kind = effect_map.get(operand.name)
                if effect_kind in (EffectKind.READ, EffectKind.READWRITE):
                    flags_parts.append("LOOM_OPERAND_READS")
                if effect_kind in (EffectKind.WRITE, EffectKind.READWRITE):
                    flags_parts.append("LOOM_OPERAND_WRITES")
                flags = " | ".join(flags_parts) if flags_parts else "0"
                lines.append(f"    {{{_bstring_expr(operand.name)}, {type_constraint}, {flags}}},")
            if synthesize_func_args_operand:
                lines.append(f"    {{{_bstring_expr(func_args_name)}, LOOM_TYPE_CONSTRAINT_ANY, LOOM_OPERAND_VARIADIC}},")
            lines.append("};")

        # Result descriptors.
        if op.results:
            lines.append(f"static const loom_result_descriptor_t {prefix}_result_desc[] = {{")
            for result in op.results:
                type_constraint = TYPE_CONSTRAINT_MAP[result.type_constraint]
                flags_parts = []
                if result.variadic:
                    flags_parts.append("LOOM_RESULT_VARIADIC")
                if result.allocates:
                    flags_parts.append("LOOM_RESULT_ALLOCATES")
                flags = " | ".join(flags_parts) if flags_parts else "0"
                lines.append(f"    {{{_bstring_expr(result.name)}, {type_constraint}, {flags}}},")
            lines.append("};")

        # Enum case name arrays. Generated C may expose an external enum alias,
        # a dialect-level shared enum, or a per-op enum typedef, but all three
        # still need one parser/printer keyword table per C symbol name.
        for attr_def in op.attrs:
            if attr_def.attr_type == "enum" and attr_def.enum_def:
                array_name = _enum_names_array_name(op, attr_def, shared_enums)
                if array_name in emitted_enum_case_name_arrays:
                    continue
                _emit_enum_case_names(lines, array_name, attr_def.enum_def)
                emitted_enum_case_name_arrays.add(array_name)

        # Instance flags case name array.
        if has_flags:
            flags_attr = next(a for a in op.attrs if a.attr_type == ATTR_TYPE_FLAGS)
            assert flags_attr.enum_def is not None, f"flags attr on {op.name} has no enum_def"
            individual_cases = [c for c in flags_attr.enum_def.cases if c.value != 0 and (c.value & (c.value - 1)) == 0]
            individual_cases.sort(key=lambda c: c.value)
            array_name = f"{prefix}_instance_flags_names"
            lines.append(f"static const loom_bstring_t {array_name}[] = {{")
            lines.extend(f"    {_bstring_expr(case.keyword)}," for case in individual_cases)
            lines.append("};")

        # Attribute symbol-reference descriptors.
        for attr_def in non_flags:
            if attr_def.symbol_ref is None:
                continue
            flags = _symbol_interface_flags(attr_def.symbol_ref.interfaces)
            descriptor_name = f"{prefix}_{attr_def.name}_symbol_ref"
            lines.append(f"static const loom_symbol_reference_descriptor_t {descriptor_name} = {{{_bstring_expr(attr_def.symbol_ref.name)}, {flags}}};")

        # Attribute descriptors.
        if non_flags:
            lines.append(f"static const loom_attr_descriptor_t {prefix}_attr_desc[] = {{")
            for attr_def in non_flags:
                if attr_def.attr_type not in ATTR_KIND_MAP:
                    raise ValueError(f"attr {attr_def.name!r} on {op.name!r} has unknown attr_type {attr_def.attr_type!r} with no C mapping")
                attr_kind = ATTR_KIND_MAP[attr_def.attr_type]
                flag_names = []
                if attr_def.optional:
                    flag_names.append("LOOM_ATTR_OPTIONAL")
                if attr_def.elide_default:
                    flag_names.append("LOOM_ATTR_ELIDE_DEFAULT")
                if attr_def.open_enum:
                    flag_names.append("LOOM_ATTR_OPEN_ENUM")
                flags = " | ".join(flag_names) if flag_names else "0"
                if attr_def.attr_type == "enum" and attr_def.enum_def:
                    enum_names = _enum_names_array_name(op, attr_def, shared_enums)
                    enum_case_count = f"IREE_ARRAYSIZE({enum_names})"
                else:
                    enum_names = "NULL"
                    enum_case_count = "0"
                symbol_ref = f"&{prefix}_{attr_def.name}_symbol_ref" if attr_def.symbol_ref is not None else "NULL"
                lines.append(f"    {{{_bstring_expr(attr_def.name)}, {attr_kind}, {flags}, {enum_case_count}, {enum_names}, {symbol_ref}}},")
            lines.append("};")

        # Region descriptors.
        if op.regions:
            implicit_terminator = _implicit_terminator_kind(op, ops_by_name)
            lines.append(f"static const loom_region_descriptor_t {prefix}_region_desc[] = {{")
            for region_def in op.regions:
                region_flags = []
                if region_def.single_block:
                    region_flags.append("LOOM_REGION_SINGLE_BLOCK")
                if region_def.optional:
                    region_flags.append("LOOM_REGION_OPTIONAL")
                flags = " | ".join(region_flags) if region_flags else "0"
                terminator = _region_terminator_kind(op, region_def, ops_by_name)
                lines.append(f"    {{{terminator}, {implicit_terminator}, {flags}}},")
            lines.append("};")

        # Constraint table.
        if op.constraints:
            lines.append(f"static const loom_constraint_t {prefix}_constraints[] = {{")
            for constraint in op.constraints:
                constraint_entry = CONSTRAINT_MAP.get(constraint.name)
                if constraint_entry is None:
                    raise ValueError(f"Op '{op.name}': unknown constraint '{constraint.name}'")
                relation_name, property_name = constraint_entry
                if property_name == "$data":
                    if constraint.data is None:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: missing data payload")
                    if not isinstance(constraint.data, int):
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: data payload must be an integer")
                    if constraint.data < 0 or constraint.data > 255:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: data payload out of uint8_t range")
                    property_name = str(constraint.data)
                elif property_name == "$type_constraint_data":
                    if not isinstance(constraint.data, TypeConstraint):
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: data payload must be a TypeConstraint")
                    property_name = TYPE_CONSTRAINT_MAP[constraint.data]
                arg_refs: list[str] = []
                for arg_name in constraint.args:
                    field = layout.fields.get(arg_name)
                    if field is None:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: unknown field '{arg_name}'")
                    if field.kind == FieldKind.SUCCESSOR:
                        raise ValueError(f"Op '{op.name}' constraint {constraint.name}: successor field '{arg_name}' cannot be encoded as a value/type constraint argument")
                    category = FIELD_CATEGORY_MAP[field.kind]
                    arg_refs.append(_constraint_arg_ref(op, constraint.name, arg_name, category, field.index))
                while len(arg_refs) < 4:
                    arg_refs.append("0")
                args_str = ", ".join(arg_refs)
                error_ref = _error_ref_literal(constraint.error) if constraint.error is not None else "LOOM_ERROR_REF_NONE"
                lines.append(f"    {{{relation_name}, {property_name}, {len(constraint.args)}, 0, {{{args_str}}}, {error_ref}}},")
            lines.append("};")

        target_like_iface = _find_interface(op, TargetLikeInterface)
        if target_like_iface is not None:
            _emit_target_like_descriptor(op, target_like_iface, lines)

        # Interface vtables.
        for spec in _INTERFACES:
            _emit_interface_vtable(op, spec, lines)

        # Symbol definition descriptor.
        if op.symbol_def is not None:
            attr_index = _resolve_attr_index(op, op.symbol_def.field, "symbol_def")
            flags = _symbol_interface_flags(op.symbol_def.interfaces)
            fact_domain = _symbol_fact_domain_symbol(op)
            lines.append(f"static const loom_symbol_definition_descriptor_t {prefix}_symbol_def = {{")
            lines.append(f"    .name = {_bstring_expr(op.symbol_def.name)},")
            if attr_index != 0:
                lines.append(f"    .name_attr_index = {attr_index},")
            if flags != "0":
                lines.append(f"    .interfaces = {flags},")
            if op.symbol_def.bytecode_kind != "LOOM_SYMBOL_NONE":
                lines.append(f"    .bytecode_kind = {op.symbol_def.bytecode_kind},")
            if fact_domain:
                lines.append(f"    .fact_domain = &{fact_domain},")
            lines.append("};")

        # Structural placement descriptor.
        required_parent_kinds = _trait_op_kinds(op, ops_by_name, "HasParent")
        required_ancestor_kinds = _trait_op_kinds(op, ops_by_name, "HasAncestor")
        forbidden_ancestor_kinds = _trait_op_kinds(op, ops_by_name, "NoAncestor")
        if required_parent_kinds or required_ancestor_kinds or forbidden_ancestor_kinds:
            required_parent_ptr = "NULL"
            required_ptr = "NULL"
            forbidden_ptr = "NULL"
            if required_parent_kinds:
                required_parent_ptr = f"{prefix}_required_parents"
                lines.append(f"static const loom_op_kind_t {required_parent_ptr}[] = {{")
                lines.extend(f"    {kind}," for kind in required_parent_kinds)
                lines.append("};")
            if required_ancestor_kinds:
                required_ptr = f"{prefix}_required_ancestors"
                lines.append(f"static const loom_op_kind_t {required_ptr}[] = {{")
                lines.extend(f"    {kind}," for kind in required_ancestor_kinds)
                lines.append("};")
            if forbidden_ancestor_kinds:
                forbidden_ptr = f"{prefix}_forbidden_ancestors"
                lines.append(f"static const loom_op_kind_t {forbidden_ptr}[] = {{")
                lines.extend(f"    {kind}," for kind in forbidden_ancestor_kinds)
                lines.append("};")
            lines.append(f"static const loom_op_placement_descriptor_t {prefix}_placement = {{")
            if required_parent_ptr != "NULL":
                lines.append(f"    .required_parents = {required_parent_ptr},")
                lines.append(f"    .required_parent_count = IREE_ARRAYSIZE({required_parent_ptr}),")
            if required_ptr != "NULL":
                lines.append(f"    .required_ancestors = {required_ptr},")
                lines.append(f"    .required_ancestor_count = IREE_ARRAYSIZE({required_ptr}),")
            if forbidden_ptr != "NULL":
                lines.append(f"    .forbidden_ancestors = {forbidden_ptr},")
                lines.append(f"    .forbidden_ancestor_count = IREE_ARRAYSIZE({forbidden_ptr}),")
            lines.append("};")

        # Vtable.
        traits = _trait_flags(op)
        vtable_flag_bits: list[str] = []
        if layout.variadic_operand or _func_args_are_operands(op):
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_OPERANDS")
        if layout.variadic_result:
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_RESULTS")
        if layout.variadic_region:
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_REGIONS")
        if has_flags:
            vtable_flag_bits.append("LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS")
        vtable_flags_str = " | ".join(vtable_flag_bits) if vtable_flag_bits else "0"

        sym_kind = _symbol_kind(op)
        canon = op.canonicalize or "NULL"
        infer_facts_fn = op.facts or "NULL"
        type_transfer_fn = op.type_transfer or "NULL"
        verify_fn = op.verify or "NULL"
        eff_traits = op.effective_traits or "NULL"
        interface_ptrs = {spec.vtable_field: _interface_vtable_ptr(op, spec) for spec in _INTERFACES}
        symbol_def_ptr = f"&{prefix}_symbol_def" if op.symbol_def is not None else "NULL"
        has_placement = any(trait.name in ("HasParent", "HasAncestor", "NoAncestor") for trait in op.traits)
        placement_ptr = f"&{prefix}_placement" if has_placement else "NULL"
        attr_desc_ptr = f"{prefix}_attr_desc" if non_flags else "NULL"
        operand_desc_ptr = f"{prefix}_operand_desc" if op.operands or _func_args_are_operands(op) else "NULL"
        result_desc_ptr = f"{prefix}_result_desc" if op.results else "NULL"
        region_desc_ptr = f"{prefix}_region_desc" if op.regions else "NULL"
        constraint_ptr = f"{prefix}_constraints" if op.constraints else "NULL"
        fmt_ptr = f"{prefix}_format" if elements else "NULL"

        vtable_storage = "const" if export_vtables else "static const"
        lines.append(f"{vtable_storage} loom_op_vtable_t {prefix}_vtable = {{")

        def append_nonzero(field_name: str, value: int | str) -> None:
            if value != 0 and value != "0":
                lines.append(f"    .{field_name} = {value},")

        def append_nonnull(field_name: str, value: str) -> None:
            if value != "NULL":
                lines.append(f"    .{field_name} = {value},")

        append_nonzero("traits", traits)
        append_nonzero("fixed_operand_count", layout.fixed_operand_count)
        append_nonzero("fixed_result_count", layout.fixed_result_count)
        append_nonzero("vtable_flags", vtable_flags_str)
        if sym_kind != "LOOM_SYMBOL_NONE":
            lines.append(f"    .symbol_kind = {sym_kind},")
        append_nonnull("canonicalize", canon)
        append_nonnull("infer_facts", infer_facts_fn)
        append_nonnull("effective_traits", eff_traits)
        append_nonnull("type_transfer", type_transfer_fn)
        append_nonnull("verify", verify_fn)
        lines.append(f"    .name = {_op_name_expr(op.name)},")
        if attr_desc_ptr != "NULL":
            lines.append(f"    .attr_descriptors = {attr_desc_ptr},")
            lines.append(f"    .attribute_count = IREE_ARRAYSIZE({attr_desc_ptr}),")
        append_nonnull("operand_descriptors", operand_desc_ptr)
        append_nonnull("result_descriptors", result_desc_ptr)
        if region_desc_ptr != "NULL":
            lines.append(f"    .region_descriptors = {region_desc_ptr},")
            lines.append(f"    .region_count = IREE_ARRAYSIZE({region_desc_ptr}),")
        if constraint_ptr != "NULL":
            lines.append(f"    .constraints = {constraint_ptr},")
            lines.append(f"    .constraint_count = IREE_ARRAYSIZE({constraint_ptr}),")
        append_nonnull("format_elements", fmt_ptr)
        if elements:
            lines.append(f"    .format_element_count = IREE_ARRAYSIZE({fmt_ptr}),")
        if has_flags:
            lines.append(f"    .instance_flags_case_names = {prefix}_instance_flags_names,")
            lines.append(f"    .instance_flags_case_count = IREE_ARRAYSIZE({prefix}_instance_flags_names),")
        for spec in _INTERFACES:
            interface_ptr = interface_ptrs[spec.vtable_field]
            if interface_ptr != "NULL":
                lines.append(f"    .{spec.vtable_field} = {interface_ptr},")
        if symbol_def_ptr != "NULL":
            lines.append(f"    .symbol_def = {symbol_def_ptr},")
        if placement_ptr != "NULL":
            lines.append(f"    .placement = {placement_ptr},")

        lines.append("};")
        lines.append("")

    lines.append("#undef _OP_NAME")
    lines.append("#undef _BSTRING")
    lines.append("")

    if not emit_registration:
        return "\n".join(lines)

    # Registration function.
    lines.append(f"static const loom_op_vtable_t* const loom_{dialect_name}_vtable_array[] = {{")
    for op in ops:
        prefix = _c_prefix(op)
        lines.append(f"    &{prefix}_vtable,")
    lines.append("};")
    lines.append("")
    lines.append(f"const loom_op_vtable_t* const* loom_{dialect_name}_dialect_vtables(")
    lines.append("    iree_host_size_t* out_count) {")
    lines.append(f"  *out_count = IREE_ARRAYSIZE(loom_{dialect_name}_vtable_array);")
    lines.append(f"  return loom_{dialect_name}_vtable_array;")
    lines.append("}")
    lines.append("")

    lines.append(f"static const loom_op_semantics_t loom_{dialect_name}_semantics_array[] = {{")
    for op in ops:
        _emit_op_semantics(lines, op)
    lines.append("};")
    lines.append("")
    lines.append(f"const loom_op_semantics_t* loom_{dialect_name}_dialect_op_semantics(")
    lines.append("    iree_host_size_t* out_count) {")
    lines.append(f"  *out_count = IREE_ARRAYSIZE(loom_{dialect_name}_semantics_array);")
    lines.append(f"  return loom_{dialect_name}_semantics_array;")
    lines.append("}")
    lines.append("")
    lines.append(f"loom_op_semantics_t loom_{dialect_name}_op_semantics(")
    lines.append("    loom_op_kind_t kind) {")
    lines.append(f"  if (loom_op_dialect_id(kind) != {_c_dialect_enum(dialect_name)}) {{")
    lines.append("    return loom_op_semantics_empty();")
    lines.append("  }")
    lines.append("  uint8_t op_index = loom_op_dialect_index(kind);")
    lines.append(f"  if (op_index >= IREE_ARRAYSIZE(loom_{dialect_name}_semantics_array)) {{")
    lines.append("    return loom_op_semantics_empty();")
    lines.append("  }")
    lines.append(f"  return loom_{dialect_name}_semantics_array[op_index];")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_tables_h(dialect_name: str, ops: Sequence[Op], *, include_path: str | None = None) -> str:
    """Generates private declarations shared by a sharded dialect table."""
    guard = f"LOOM_OPS_{dialect_name.upper()}_TABLES_H_"
    lines: list[str] = []

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.c_tables"))
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    lines.append(f'#include "{include_path}/ops.h"')
    lines.append("")
    _emit_table_string_macros(lines, dialect_name)
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")
    lines.extend(f"extern const loom_op_vtable_t {_c_prefix(op)}_vtable;" for op in ops)
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('}  // extern "C"')
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    lines.append("")
    return "\n".join(lines)


def generate_tables_aggregator_c(
    dialect_name: str,
    dialect_id: int,
    ops: Sequence[Op],
    *,
    include_path: str | None = None,
) -> str:
    """Generates a dialect table aggregator for sharded per-op vtable files."""
    lines: list[str] = []

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    lines.append(f'#include "{include_path}/tables.h"')
    lines.append("")

    lines.append(f"static const loom_op_vtable_t* const loom_{dialect_name}_vtable_array[] = {{")
    lines.extend(f"    &{_c_prefix(op)}_vtable," for op in ops)
    lines.append("};")
    lines.append("")
    lines.append(f"const loom_op_vtable_t* const* loom_{dialect_name}_dialect_vtables(")
    lines.append("    iree_host_size_t* out_count) {")
    lines.append(f"  *out_count = IREE_ARRAYSIZE(loom_{dialect_name}_vtable_array);")
    lines.append(f"  return loom_{dialect_name}_vtable_array;")
    lines.append("}")
    lines.append("")

    lines.append(f"static const loom_op_semantics_t loom_{dialect_name}_semantics_array[] = {{")
    for op in ops:
        _emit_op_semantics(lines, op)
    lines.append("};")
    lines.append("")
    lines.append(f"const loom_op_semantics_t* loom_{dialect_name}_dialect_op_semantics(")
    lines.append("    iree_host_size_t* out_count) {")
    lines.append(f"  *out_count = IREE_ARRAYSIZE(loom_{dialect_name}_semantics_array);")
    lines.append(f"  return loom_{dialect_name}_semantics_array;")
    lines.append("}")
    lines.append("")
    lines.append(f"loom_op_semantics_t loom_{dialect_name}_op_semantics(")
    lines.append("    loom_op_kind_t kind) {")
    lines.append(f"  if (loom_op_dialect_id(kind) != {_c_dialect_enum(dialect_name)}) {{")
    lines.append("    return loom_op_semantics_empty();")
    lines.append("  }")
    lines.append("  uint8_t op_index = loom_op_dialect_index(kind);")
    lines.append(f"  if (op_index >= IREE_ARRAYSIZE(loom_{dialect_name}_semantics_array)) {{")
    lines.append("    return loom_op_semantics_empty();")
    lines.append("  }")
    lines.append(f"  return loom_{dialect_name}_semantics_array[op_index];")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_sharded_tables_c(
    dialect_name: str,
    dialect_id: int,
    category_groups: Sequence[tuple[Any, Sequence[Op]]],
    *,
    include_path: str | None = None,
) -> dict[str, str]:
    """Generates an aggregator plus category shards for one dialect."""
    all_ops: list[Op] = []
    table_files: dict[str, str] = {}
    for category, category_ops in category_groups:
        shard_ops = list(category_ops)
        all_ops.extend(shard_ops)
        if not shard_ops:
            continue
        category_key = category.key
        filename = f"tables/{_c_identifier(category_key)}.c"
        table_files[filename] = generate_tables_c(
            dialect_name,
            dialect_id,
            shard_ops,
            include_path=include_path,
            emit_registration=False,
            export_vtables=True,
            private_header=True,
        )
    table_files["tables.c"] = generate_tables_aggregator_c(dialect_name, dialect_id, all_ops, include_path=include_path)
    table_files["tables.h"] = generate_tables_h(dialect_name, all_ops, include_path=include_path)
    return table_files


# ============================================================================
# builders.c generation
# ============================================================================


def generate_builders_c(dialect_name: str, ops: Sequence[Op], *, include_path: str | None = None) -> str:
    """Generates the builders.c file for a dialect."""
    lines: list[str] = []
    shared_enums = _collect_shared_enums(dialect_name, ops)

    lines.append(COPYRIGHT)
    lines.extend(line_comment_header("//", generator="loom.gen.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    lines.append(f'#include "{include_path}/ops.h"')
    lines.append("")
    lines.append("#include <string.h>")
    lines.append("")
    lines.append('#include "loom/ir/module.h"')
    lines.append('#include "loom/ops/builder_macros.h"')
    if any(isinstance(element, DescriptorRef) for op in ops for element in _flatten_format(op.format)):
        lines.append('#include "loom/util/stable_id.h"')
    lines.append("")

    for op in ops:
        prefix = _c_prefix(op)
        enum_name = _c_enum_name(op)
        pattern = _detect_builder_pattern(op)

        if pattern == "BINARY":
            lines.append(f"LOOM_DEFINE_BINARY_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "BINARY_WITH_FLAGS":
            lines.append(f"LOOM_DEFINE_BINARY_OP_WITH_FLAGS_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "UNARY":
            lines.append(f"LOOM_DEFINE_UNARY_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "UNARY_WITH_FLAGS":
            lines.append(f"LOOM_DEFINE_UNARY_OP_WITH_FLAGS_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "CAST":
            lines.append(f"LOOM_DEFINE_CAST_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "COMPARISON":
            lines.append(f"LOOM_DEFINE_COMPARISON_OP_BUILDER({prefix}_build, {enum_name})")
        elif pattern == "COMPARISON_WITH_FLAGS":
            lines.append(f"LOOM_DEFINE_COMPARISON_OP_WITH_FLAGS_BUILDER({prefix}_build, {enum_name})")
        else:
            lines.extend(_generate_builder_implementation(op, prefix, enum_name, shared_enums))

        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Op registry (production dialect registration)
# ============================================================================


def generate_op_registry(
    dialects: list[tuple[Any, list[Op]]],
) -> tuple[str, str]:
    """Generate op_registry.h and op_registry.c.

    Returns (header_content, source_content).
    """
    # Header.
    header = [GENERATED_HEADER]
    header.append("#ifndef LOOM_OPS_OP_REGISTRY_H_")
    header.append("#define LOOM_OPS_OP_REGISTRY_H_")
    header.append("")
    header.append('#include "iree/base/api.h"')
    header.append('#include "loom/ir/context.h"')
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append('extern "C" {')
    header.append("#endif")
    header.append("")
    header.append("// Registers production dialect vtables and built-in encoding families.")
    header.append("//")
    header.append("// The context must have been initialized and must not have been")
    header.append("// finalized yet. The test dialect is intentionally not registered here;")
    header.append("// developer tools and tests that need it must opt in explicitly.")
    header.append("iree_status_t loom_op_registry_register_all_dialects(")
    header.append("    loom_context_t* context);")
    header.append("")
    header.append("// Initializes |out_context| with production dialects and encodings.")
    header.append("//")
    header.append("// On failure the partially initialized context is deinitialized before")
    header.append("// returning.")
    header.append("iree_status_t loom_op_registry_initialize_context(")
    header.append("    iree_allocator_t allocator, loom_context_t* out_context);")
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append("}")
    header.append("#endif")
    header.append("")
    header.append("#endif  // LOOM_OPS_OP_REGISTRY_H_")
    header.append("")

    # Source.
    source = [GENERATED_HEADER]
    source.append('#include "loom/ops/op_registry.h"')
    source.append("")
    source.append("#include <stdint.h>")
    source.append("")
    source.append('#include "loom/ops/encoding/families.h"')
    source.extend(f'#include "{_c_dialect_include_path(dialect)}/ops.h"' for dialect, _ops in sorted(dialects, key=lambda item: item[0].name))
    source.append("")
    source.append("typedef const loom_op_vtable_t* const* (*loom_op_registry_dialect_vtables_fn_t)(")
    source.append("    iree_host_size_t* out_count);")
    source.append("")
    source.append("typedef const loom_op_semantics_t* (*loom_op_registry_dialect_semantics_fn_t)(")
    source.append("    iree_host_size_t* out_count);")
    source.append("")
    source.append("typedef struct loom_op_registry_dialect_registration_t {")
    source.append("  loom_dialect_id_t dialect_id;")
    source.append("  loom_op_registry_dialect_vtables_fn_t vtables_fn;")
    source.append("  loom_op_registry_dialect_semantics_fn_t semantics_fn;")
    source.append("} loom_op_registry_dialect_registration_t;")
    source.append("")
    source.append("static const loom_op_registry_dialect_registration_t loom_op_registry_dialects[] = {")
    for dialect, _ops in sorted(dialects, key=lambda item: item[0].dialect_id):
        source.append(f"    {{{_c_dialect_enum(dialect.name)}, loom_{dialect.name}_dialect_vtables, loom_{dialect.name}_dialect_op_semantics}},")
    source.append("};")
    source.append("")
    source.append("static iree_status_t loom_op_registry_register_dialect(")
    source.append("    loom_context_t* context,")
    source.append("    const loom_op_registry_dialect_registration_t* registration) {")
    source.append("  iree_host_size_t count = 0;")
    source.append("  const loom_op_vtable_t* const* vtables = registration->vtables_fn(&count);")
    source.append("  iree_host_size_t semantics_count = 0;")
    source.append("  const loom_op_semantics_t* semantics =")
    source.append("      registration->semantics_fn(&semantics_count);")
    source.append("  if (count > UINT16_MAX) {")
    source.append("    return iree_make_status(")
    source.append("        IREE_STATUS_RESOURCE_EXHAUSTED,")
    source.append('        "dialect %u has %" PRIhsz')
    source.append('        " ops, exceeding the uint16_t registry cap",')
    source.append("        (unsigned)registration->dialect_id, count);")
    source.append("  }")
    source.append("  if (semantics_count != count) {")
    source.append("    return iree_make_status(")
    source.append("        IREE_STATUS_FAILED_PRECONDITION,")
    source.append('        "dialect %u semantics count %" PRIhsz')
    source.append('        " does not match vtable count %" PRIhsz,')
    source.append("        (unsigned)registration->dialect_id, semantics_count, count);")
    source.append("  }")
    source.append("  IREE_RETURN_IF_ERROR(loom_context_register_dialect(")
    source.append("      context, registration->dialect_id, vtables, (uint16_t)count));")
    source.append("  return loom_context_register_dialect_semantics(")
    source.append("      context, registration->dialect_id, semantics, (uint16_t)count);")
    source.append("}")
    source.append("")
    source.append("iree_status_t loom_op_registry_register_all_dialects(loom_context_t* context) {")
    source.append("  for (iree_host_size_t i = 0;")
    source.append("       i < IREE_ARRAYSIZE(loom_op_registry_dialects); ++i) {")
    source.append("    IREE_RETURN_IF_ERROR(loom_op_registry_register_dialect(")
    source.append("        context, &loom_op_registry_dialects[i]));")
    source.append("  }")
    source.append("  return loom_context_register_builtin_encoding_vtables(context);")
    source.append("}")
    source.append("")
    source.append("iree_status_t loom_op_registry_initialize_context(")
    source.append("    iree_allocator_t allocator, loom_context_t* out_context) {")
    source.append("  loom_context_initialize(allocator, out_context);")
    source.append("  iree_status_t status =")
    source.append("      loom_op_registry_register_all_dialects(out_context);")
    source.append("  if (iree_status_is_ok(status)) {")
    source.append("    status = loom_context_finalize(out_context);")
    source.append("  }")
    source.append("  if (!iree_status_is_ok(status)) {")
    source.append("    loom_context_deinitialize(out_context);")
    source.append("  }")
    source.append("  return status;")
    source.append("}")
    source.append("")

    return "\n".join(header), "\n".join(source)


# ============================================================================
# Type registry (type descriptors + sorted lookup table)
# ============================================================================

# Maps ir_kind string to C enum name.
_IR_KIND_MAP: dict[str, str] = {
    "tile": "LOOM_TYPE_TILE",
    "tensor": "LOOM_TYPE_TENSOR",
    "vector": "LOOM_TYPE_VECTOR",
    "view": "LOOM_TYPE_VIEW",
    "buffer": "LOOM_TYPE_BUFFER",
    "storage": "LOOM_TYPE_STORAGE",
    "pool": "LOOM_TYPE_POOL",
    "group": "LOOM_TYPE_GROUP",
    "dialect": "LOOM_TYPE_DIALECT",
}


def _type_c_ident(name: str) -> str:
    """Convert a type name to a C identifier fragment: 'hal.buffer' -> 'hal_buffer'."""
    return name.replace(".", "_")


_C_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def _type_fact_domain_symbol(type_def: Any) -> str | None:
    """Returns the validated C fact-domain symbol for a TypeDef, if any."""
    fact_domain = getattr(type_def, "fact_domain", None)
    if fact_domain is None:
        return None
    if not isinstance(fact_domain, str) or not _C_SYMBOL_RE.fullmatch(fact_domain):
        raise ValueError(f"TypeDef {type_def.name!r}: fact_domain must be a C symbol name, got {fact_domain!r}")
    return fact_domain


def _symbol_fact_domain_symbol(op: Any) -> str | None:
    """Returns the validated C symbol fact-domain symbol for an Op, if any."""
    if op.symbol_def is None:
        return None
    fact_domain = getattr(op.symbol_def, "fact_domain", None)
    if fact_domain is None:
        return None
    if not isinstance(fact_domain, str) or not _C_SYMBOL_RE.fullmatch(fact_domain):
        raise ValueError(f"Op {op.name!r}: symbol_def.fact_domain must be a C symbol name, got {fact_domain!r}")
    return fact_domain


def _normalize_c_symbol_reference(symbol: str) -> str:
    """Returns a bare C symbol name from a c_ptr field value."""
    symbol = symbol[1:] if symbol.startswith("&") else symbol
    if not _C_SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"C pointer interface field must be a C symbol name, got {symbol!r}")
    return symbol


def _interface_c_ptr_symbols(ops: Sequence[Op]) -> list[tuple[str, str]]:
    """Returns sorted (type, symbol) extern declarations for interface C pointers."""
    declarations: set[tuple[str, str]] = set()
    for op in ops:
        for spec in _INTERFACES:
            iface = _find_interface(op, spec.python_class)
            if iface is None:
                continue
            for field_spec in spec.fields:
                if field_spec.kind != "c_ptr":
                    continue
                if isinstance(iface, TargetLikeInterface) and field_spec.py_field == "descriptor" and iface.bundle_table is not None:
                    continue
                if not field_spec.c_pointee_type:
                    raise ValueError(f"{spec.name} field {field_spec.py_field!r}: kind='c_ptr' requires c_pointee_type")
                py_value = getattr(iface, field_spec.py_field)
                if py_value is None:
                    continue
                if not isinstance(py_value, str) or not py_value:
                    raise ValueError(f"{spec.name} field {field_spec.py_field!r}: expected C symbol name or None, got {py_value!r}")
                declarations.add((field_spec.c_pointee_type, _normalize_c_symbol_reference(py_value)))
    return sorted(declarations)


def _target_like_bundle_table_symbols(ops: Sequence[Op]) -> list[str]:
    symbols: set[str] = set()
    for op in ops:
        iface = _find_interface(op, TargetLikeInterface)
        if iface is None or iface.bundle_table is None:
            continue
        symbols.add(_normalize_c_symbol_reference(iface.bundle_table))
    return sorted(symbols)


def _translate_type_format_elements(
    type_def: Any,
) -> list[str]:
    """Translate a TypeDef's format elements to C initializer strings."""
    from loom.assembly import (
        Attr,
        Clause,
        EncodingOf,
        Glue,
        Keyword,
        OptionalGroup,
        ScalarOf,
        ShapeOf,
        TypeOf,
    )

    param_names = [p.name for p in type_def.params]

    def param_index(field: str) -> int:
        return param_names.index(field)

    def translate(element: Any) -> list[str]:
        match element:
            case ShapeOf(field=field):
                return [f"{{LOOM_TYPE_FMT_SHAPE, {param_index(field)}, 0}}"]
            case ScalarOf(field=field):
                return [f"{{LOOM_TYPE_FMT_SCALAR, {param_index(field)}, 0}}"]
            case EncodingOf(field=field):
                return [f"{{LOOM_TYPE_FMT_ENCODING, {param_index(field)}, 0}}"]
            case TypeOf(field=field):
                return [f"{{LOOM_TYPE_FMT_TYPE, {param_index(field)}, 0}}"]
            case Attr(field=field):
                return [f"{{LOOM_TYPE_FMT_ATTR, {param_index(field)}, 0}}"]
            case Glue():
                return ["{LOOM_TYPE_FMT_GLUE, 0, 0}"]
            case Keyword(text=text):
                kw_name = KEYWORD_MAP.get(text)
                if kw_name is None:
                    raise ValueError(f"unknown keyword in type format: {text!r}")
                return [f"{{LOOM_TYPE_FMT_KEYWORD, 0, {kw_name}}}"]
            case Clause(name=name, elements=elements):
                kw_name = KEYWORD_MAP.get(name)
                if kw_name is None:
                    raise ValueError(f"unknown clause name in type format: {name!r}")
                result = [
                    f"{{LOOM_TYPE_FMT_KEYWORD, 0, {kw_name}}}",
                    "{LOOM_TYPE_FMT_GLUE, 0, 0}",
                    f"{{LOOM_TYPE_FMT_KEYWORD, 0, {KEYWORD_MAP['(']}}}",
                ]
                for e in elements:
                    result.extend(translate(e))
                result.append(f"{{LOOM_TYPE_FMT_KEYWORD, 0, {KEYWORD_MAP[')']}}}")
                return result
            case OptionalGroup(elements=elements, anchor=anchor):
                anchor_idx = param_index(anchor)
                inner = []
                for e in elements:
                    inner.extend(translate(e))
                skip_count = len(inner)
                # data: high byte = skip count, low byte = anchor field index.
                data = f"({skip_count} << 8) | {anchor_idx}"
                result = [f"{{LOOM_TYPE_FMT_OPTIONAL, {anchor_idx}, {data}}}"]
                result.extend(inner)
                return result
            case _:
                raise ValueError(f"unsupported type format element: {element!r}")

    elements: list[str] = []
    for element in type_def.format:
        elements.extend(translate(element))
    return elements


def generate_type_registry(
    all_types: list[Any],
) -> tuple[str, str]:
    """Generate type_registry.h and type_registry.c.

    Returns (header_content, source_content).
    """
    # Header.
    header = [GENERATED_HEADER]
    header.append("#ifndef LOOM_OPS_TYPE_REGISTRY_H_")
    header.append("#define LOOM_OPS_TYPE_REGISTRY_H_")
    header.append("")
    header.append('#include "iree/base/api.h"')
    header.append('#include "loom/ir/types.h"')
    header.append('#include "loom/ops/op_defs.h"')
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append('extern "C" {')
    header.append("#endif")
    header.append("")
    header.append("typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;")
    header.append("")
    header.append("// Format element kinds for type interiors (inside <...>).")
    header.append("// These are separate from op format elements because type")
    header.append("// interiors have different semantics (shape dims, element")
    header.append("// types, encodings) than op bodies (operand refs, attr values).")
    header.append("typedef enum loom_type_format_kind_e {")
    header.append("  LOOM_TYPE_FMT_SHAPE = 0,      // Dimension list: 4x[%M]x...")
    header.append("  LOOM_TYPE_FMT_SCALAR = 1,      // Element type keyword: f32, i8.")
    header.append("  LOOM_TYPE_FMT_ENCODING = 2,    // Encoding ref: #q8_0 or %enc.")
    header.append("  LOOM_TYPE_FMT_TYPE = 3,         // Recursive type: vm.ref<T>.")
    header.append("  LOOM_TYPE_FMT_ATTR = 4,         // Bare identifier: group<workgroup>.")
    header.append("  LOOM_TYPE_FMT_KEYWORD = 5,      // Literal punctuation/word.")
    header.append("  LOOM_TYPE_FMT_OPTIONAL = 6,     // Conditional elements.")
    header.append("  LOOM_TYPE_FMT_GLUE = 7,         // Suppress space.")
    header.append("} loom_type_format_kind_t;")
    header.append("")
    header.append("// A 4-byte format element for type interiors. Same layout")
    header.append("// as loom_format_element_t for consistent handling.")
    header.append("typedef struct loom_type_format_element_t {")
    header.append("  // Format opcode, encoded as loom_type_format_kind_t.")
    header.append("  uint8_t kind;")
    header.append("")
    header.append("  // Parameter index consumed by this element.")
    header.append("  uint8_t field_index;")
    header.append("")
    header.append("  // Kind-specific payload such as a keyword ID or skip count.")
    header.append("  uint16_t data;")
    header.append("} loom_type_format_element_t;")
    header.append("")
    header.append("// Descriptor for a registered type. Contains the name,")
    header.append("// the IR type kind to construct, parameter count, and")
    header.append("// format elements describing the type interior syntax.")
    header.append("typedef struct loom_type_descriptor_t {")
    header.append('  // B-string name: [length]"tile", [length]"hal.buffer".')
    header.append("  const uint8_t* name;")
    header.append("")
    header.append("  // What IR type kind to construct when parsing.")
    header.append("  loom_type_kind_t ir_kind;")
    header.append("")
    header.append("  // Number of declared parameters.")
    header.append("  uint8_t param_count;")
    header.append("")
    header.append("  // Optional type-owned value fact domain. NULL means the type only has generic")
    header.append("  // scalar facts or uses the domain-free extension behavior.")
    header.append("  const loom_value_fact_domain_t* fact_domain;")
    header.append("")
    header.append("  // Semantic role and target-contract families for this type.")
    header.append("  loom_type_semantics_t semantics;")
    header.append("")
    header.append("  // Format element array for the type interior (inside <...>).")
    header.append("  // NULL for opaque types (no angle brackets).")
    header.append("  const loom_type_format_element_t* format_elements;")
    header.append("")
    header.append("  // Number of entries in |format_elements|.")
    header.append("  uint8_t format_element_count;")
    header.append("} loom_type_descriptor_t;")
    header.append("")
    header.append("// Entry in the sorted type registry.")
    header.append("typedef struct loom_type_registry_entry_t {")
    header.append("  iree_string_view_t name;")
    header.append("  const loom_type_descriptor_t* descriptor;")
    header.append("} loom_type_registry_entry_t;")
    header.append("")
    header.append("// Returns the number of entries in the global type registry.")
    header.append("iree_host_size_t loom_type_registry_count(void);")
    header.append("")
    header.append("// Returns the sorted registry array (for iteration/testing).")
    header.append("const loom_type_registry_entry_t* loom_type_registry_entries(void);")
    header.append("")
    header.append('// Looks up a type descriptor by name (e.g., "tile", "hal.buffer").')
    header.append("// Returns the descriptor on success, NULL if not found.")
    header.append("const loom_type_descriptor_t* loom_type_registry_lookup(")
    header.append("    iree_string_view_t name);")
    header.append("")
    header.append("// Resolves the type-owned value fact domain for |type|, or NULL if the")
    header.append("// registered type has no extension fact domain.")
    header.append("const loom_value_fact_domain_t* loom_type_registry_resolve_fact_domain(")
    header.append("    void* user_data, const loom_fact_context_t* context,")
    header.append("    const loom_module_t* module, loom_type_t type);")
    header.append("")
    header.append("// Installs the generated type-registry fact-domain resolver on |context|.")
    header.append("void loom_type_registry_configure_fact_context(")
    header.append("    loom_fact_context_t* context);")
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append("}")
    header.append("#endif")
    header.append("")
    header.append("#endif  // LOOM_OPS_TYPE_REGISTRY_H_")
    header.append("")

    # Source.
    source = [GENERATED_HEADER]
    source.append('#include "loom/ops/type_registry.h"')
    source.append("")
    source.append('#include "loom/util/fact_table.h"')
    source.append("")

    # Emit B-string names, format arrays, and descriptors for each type.
    for td in all_types:
        ident = _type_c_ident(td.name)
        name_len = len(td.name)
        source.append(f'static const uint8_t loom_type_{ident}_name[] = "\\x{name_len:02x}" "{td.name}";')

    source.append("")

    fact_domain_symbols = sorted({fact_domain for td in all_types if (fact_domain := _type_fact_domain_symbol(td)) is not None})
    if fact_domain_symbols:
        source.extend(f"extern const loom_value_fact_domain_t {fact_domain};" for fact_domain in fact_domain_symbols)
        source.append("")

    # Format element arrays for parameterized types.
    for td in all_types:
        if not td.format:
            continue
        ident = _type_c_ident(td.name)
        elements = _translate_type_format_elements(td)
        source.append(f"static const loom_type_format_element_t loom_type_{ident}_format[] = {{")
        source.extend(f"    {elem_str}," for elem_str in elements)
        source.append("};")

    source.append("")

    # Descriptors.
    for td in all_types:
        ident = _type_c_ident(td.name)
        ir_kind = _IR_KIND_MAP[td.ir_kind]
        param_count = len(td.params)
        fact_domain = _type_fact_domain_symbol(td)
        if td.format:
            elements = _translate_type_format_elements(td)
            fmt_ref = f"loom_type_{ident}_format"
            fmt_count = len(elements)
        else:
            fmt_ref = "NULL"
            fmt_count = 0
        source.append(f"static const loom_type_descriptor_t loom_type_{ident}_descriptor = {{")
        source.append(f"    .name = loom_type_{ident}_name,")
        source.append(f"    .ir_kind = {ir_kind},")
        if param_count != 0:
            source.append(f"    .param_count = {param_count},")
        if fact_domain:
            source.append(f"    .fact_domain = &{fact_domain},")
        _emit_type_semantics(source, td)
        if fmt_ref != "NULL":
            source.append(f"    .format_elements = {fmt_ref},")
            source.append(f"    .format_element_count = {fmt_count},")
        source.append("};")
        source.append("")

    # Sorted registry array.
    sorted_types = sorted(all_types, key=lambda td: td.name)
    source.append("static const loom_type_registry_entry_t loom_type_registry[] = {")
    for td in sorted_types:
        ident = _type_c_ident(td.name)
        source.append(f'    {{IREE_SVL("{td.name}"), &loom_type_{ident}_descriptor}},')
    source.append("};")
    source.append("")
    count = len(sorted_types)
    source.append(f"#define LOOM_TYPE_REGISTRY_COUNT {count}")
    source.append("")
    source.append("iree_host_size_t loom_type_registry_count(void) {")
    source.append("  return LOOM_TYPE_REGISTRY_COUNT;")
    source.append("}")
    source.append("")
    source.append("const loom_type_registry_entry_t* loom_type_registry_entries(void) {")
    source.append("  return loom_type_registry;")
    source.append("}")
    source.append("")
    source.append("static iree_string_view_t loom_type_registry_builtin_name(")
    source.append("    loom_type_kind_t kind) {")
    source.append("  switch (kind) {")
    source.append("    case LOOM_TYPE_TILE:")
    source.append('      return IREE_SV("tile");')
    source.append("    case LOOM_TYPE_TENSOR:")
    source.append('      return IREE_SV("tensor");')
    source.append("    case LOOM_TYPE_VECTOR:")
    source.append('      return IREE_SV("vector");')
    source.append("    case LOOM_TYPE_VIEW:")
    source.append('      return IREE_SV("view");')
    source.append("    case LOOM_TYPE_BUFFER:")
    source.append('      return IREE_SV("buffer");')
    source.append("    case LOOM_TYPE_STORAGE:")
    source.append('      return IREE_SV("low.storage");')
    source.append("    case LOOM_TYPE_GROUP:")
    source.append('      return IREE_SV("group");')
    source.append("    case LOOM_TYPE_POOL:")
    source.append('      return IREE_SV("pool");')
    source.append("    default:")
    source.append("      return iree_string_view_empty();")
    source.append("  }")
    source.append("}")
    source.append("")
    source.append("const loom_type_descriptor_t* loom_type_registry_lookup(")
    source.append("    iree_string_view_t name) {")
    source.append("  iree_host_size_t low = 0;")
    source.append("  iree_host_size_t high = LOOM_TYPE_REGISTRY_COUNT;")
    source.append("  while (low < high) {")
    source.append("    iree_host_size_t mid = low + (high - low) / 2;")
    source.append("    int cmp = iree_string_view_compare(loom_type_registry[mid].name, name);")
    source.append("    if (cmp < 0) {")
    source.append("      low = mid + 1;")
    source.append("    } else if (cmp > 0) {")
    source.append("      high = mid;")
    source.append("    } else {")
    source.append("      return loom_type_registry[mid].descriptor;")
    source.append("    }")
    source.append("  }")
    source.append("  return NULL;")
    source.append("}")
    source.append("")
    source.append("const loom_value_fact_domain_t* loom_type_registry_resolve_fact_domain(")
    source.append("    void* user_data, const loom_fact_context_t* context,")
    source.append("    const loom_module_t* module, loom_type_t type) {")
    source.append("  (void)user_data;")
    source.append("  (void)context;")
    source.append("  iree_string_view_t name = iree_string_view_empty();")
    source.append("  if (loom_type_is_dialect(type)) {")
    source.append("    loom_string_id_t name_id = loom_type_dialect_name_id(type);")
    source.append("    if (!module || name_id == LOOM_STRING_ID_INVALID ||")
    source.append("        (iree_host_size_t)name_id >= module->strings.count) {")
    source.append("      return NULL;")
    source.append("    }")
    source.append("    name = module->strings.entries[name_id];")
    source.append("  } else {")
    source.append("    name = loom_type_registry_builtin_name(loom_type_kind(type));")
    source.append("  }")
    source.append("  if (iree_string_view_is_empty(name)) {")
    source.append("    return NULL;")
    source.append("  }")
    source.append("  const loom_type_descriptor_t* descriptor =")
    source.append("      loom_type_registry_lookup(name);")
    source.append("  return descriptor ? descriptor->fact_domain : NULL;")
    source.append("}")
    source.append("")
    source.append("void loom_type_registry_configure_fact_context(")
    source.append("    loom_fact_context_t* context) {")
    source.append("  if (!context) {")
    source.append("    return;")
    source.append("  }")
    source.append("  context->resolve_type_domain =")
    source.append("      loom_value_fact_type_domain_resolver_callback_make(")
    source.append("          loom_type_registry_resolve_fact_domain, NULL);")
    source.append("}")
    source.append("")

    return "\n".join(header), "\n".join(source)


# ============================================================================
# Keyword definitions
# ============================================================================


def generate_keyword_enum_inc() -> str:
    """Generate keyword_enum.inc — the enum body for loom_keyword_id_e.

    Ordinals are assigned by position in KEYWORD_MAP. The generated
    file is #included from op_defs.h inside the enum typedef.
    """
    lines = [COPYRIGHT, "// clang-format off", ""]
    for ordinal, (text, c_name) in enumerate(KEYWORD_MAP.items()):
        # Escape backslashes and quotes for the comment.
        display = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  {c_name} = {ordinal},  // "{display}"')
    lines.append("")
    return "\n".join(lines)


def generate_keyword_table_inc() -> str:
    """Generate keyword_table.inc — bstring initializers for keyword_bstrings.

    The generated file is #included from op_defs.c inside the array
    initializer.
    """
    lines = [COPYRIGHT, "// clang-format off", ""]
    for text, c_name in KEYWORD_MAP.items():
        text_length = len(text)
        # Escape the text for C string literals.
        c_text = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    [{c_name}] = (const uint8_t*)"\\x{text_length:02x}" "{c_text}",')
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """Generate C tables for all registered dialects."""
    from loom.builtin_types import ALL_BUILTIN_TYPES
    from loom.dialect.buffer import ALL_BUFFER_OPS, buffer_ops
    from loom.dialect.cfg import ALL_CFG_OPS, cfg_ops
    from loom.dialect.check import ALL_CHECK_OPS, check_ops
    from loom.dialect.encoding import ALL_ENCODING_OPS, encoding_ops
    from loom.dialect.func import ALL_FUNC_OPS, func_ops
    from loom.dialect.globals import ALL_GLOBAL_OPS, global_ops
    from loom.dialect.hal import ALL_HAL_TYPES
    from loom.dialect.index import ALL_INDEX_OPS, index_ops
    from loom.dialect.kernel import ALL_KERNEL_OPS, ALL_KERNEL_TYPES, kernel_ops
    from loom.dialect.llvmir import ALL_LLVMIR_OPS, llvmir_ops
    from loom.dialect.low import ALL_LOW_OPS, low_ops
    from loom.dialect.pass_ import ALL_PASS_OPS, pass_ops
    from loom.dialect.pool import ALL_POOL_OPS, pool_ops
    from loom.dialect.scalar import ALL_SCALAR_OPS, SCALAR_OP_CATEGORY_GROUPS, scalar_ops
    from loom.dialect.scf import ALL_SCF_OPS, scf_ops
    from loom.dialect.target import ALL_TARGET_OPS, target_ops
    from loom.dialect.test import ALL_TEST_OPS, test_ops
    from loom.dialect.vector import ALL_VECTOR_OPS, VECTOR_OP_CATEGORY_GROUPS, vector_ops
    from loom.dialect.view import ALL_VIEW_OPS, view_ops
    from loom.target.arch.amdgpu.dialect import ALL_AMDGPU_OPS, amdgpu_ops
    from loom.target.arch.wasm.dialect import ALL_WASM_OPS, wasm_ops
    from loom.target.arch.x86.dialect import ALL_X86_OPS, x86_ops
    from loom.target.emit.ireevm.dialect import ALL_IREEVM_OPS, ireevm_ops

    dialects = [
        (test_ops, list(ALL_TEST_OPS), None),
        (scalar_ops, list(ALL_SCALAR_OPS), SCALAR_OP_CATEGORY_GROUPS),
        (func_ops, list(ALL_FUNC_OPS), None),
        (encoding_ops, list(ALL_ENCODING_OPS), None),
        (pool_ops, list(ALL_POOL_OPS), None),
        (global_ops, list(ALL_GLOBAL_OPS), None),
        (scf_ops, list(ALL_SCF_OPS), None),
        (cfg_ops, list(ALL_CFG_OPS), None),
        (check_ops, list(ALL_CHECK_OPS), None),
        (buffer_ops, list(ALL_BUFFER_OPS), None),
        (view_ops, list(ALL_VIEW_OPS), None),
        (vector_ops, list(ALL_VECTOR_OPS), VECTOR_OP_CATEGORY_GROUPS),
        (index_ops, list(ALL_INDEX_OPS), None),
        (kernel_ops, list(ALL_KERNEL_OPS), None),
        (llvmir_ops, list(ALL_LLVMIR_OPS), None),
        (target_ops, list(ALL_TARGET_OPS), None),
        (low_ops, list(ALL_LOW_OPS), None),
        (pass_ops, list(ALL_PASS_OPS), None),
        (amdgpu_ops, list(ALL_AMDGPU_OPS), None),
        (x86_ops, list(ALL_X86_OPS), None),
        (wasm_ops, list(ALL_WASM_OPS), None),
        (ireevm_ops, list(ALL_IREEVM_OPS), None),
    ]
    production_dialects = [(dialect, ops) for dialect, ops, _ in dialects if dialect.register_by_default]

    output_root = _bootstrap.REPO_ROOT / "loom" / "src" / "loom"

    for dialect, ops, table_shards in dialects:
        dialect_dir = output_root / _c_dialect_path(dialect)
        dialect_dir.mkdir(parents=True, exist_ok=True)
        include_path = _c_dialect_include_path(dialect)

        ops_h = generate_ops_h(dialect.name, dialect.dialect_id, ops)
        table_files = (
            generate_sharded_tables_c(
                dialect.name,
                dialect.dialect_id,
                table_shards,
                include_path=include_path,
            )
            if table_shards is not None
            else {
                "tables.c": generate_tables_c(
                    dialect.name,
                    dialect.dialect_id,
                    ops,
                    include_path=include_path,
                )
            }
        )
        builders_c = generate_builders_c(dialect.name, ops, include_path=include_path)

        ops_h_path = dialect_dir / "ops.h"
        builders_c_path = dialect_dir / "builders.c"
        generated_paths = {dialect_dir / filename for filename in table_files}
        if table_shards is not None:
            for stale_path in dialect_dir.glob("tables_*.c"):
                if stale_path not in generated_paths:
                    stale_path.unlink()
            shard_dir = dialect_dir / "tables"
            if shard_dir.exists():
                for stale_path in shard_dir.glob("*.c"):
                    if stale_path not in generated_paths:
                        stale_path.unlink()

        for path, content in [
            (ops_h_path, ops_h),
            (builders_c_path, builders_c),
            *((dialect_dir / filename, content) for filename, content in table_files.items()),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        rel = dialect_dir.relative_to(_bootstrap.REPO_ROOT / "loom" / "src" / "loom")
        print(f"  {dialect.name}: {len(ops)} ops in {rel}/")

    # Generate cross-dialect registries.
    op_reg_h, op_reg_c = generate_op_registry(production_dialects)
    all_types = [
        *ALL_BUILTIN_TYPES,
        *ALL_HAL_TYPES,
        *ALL_KERNEL_TYPES,
    ]
    type_reg_h, type_reg_c = generate_type_registry(all_types)

    # Generate keyword definitions.
    keyword_enum = generate_keyword_enum_inc()
    keyword_table = generate_keyword_table_inc()

    for filename, content in [
        ("op_registry.h", op_reg_h),
        ("op_registry.c", op_reg_c),
        ("type_registry.h", type_reg_h),
        ("type_registry.c", type_reg_c),
        ("keyword_enum.inc", keyword_enum),
        ("keyword_table.inc", keyword_table),
    ]:
        path = output_root / "ops" / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    total_ops = sum(len(ops) for _, ops, _ in dialects)
    total_types = len(all_types)
    print(f"  op tables: {total_ops} ops")
    print(f"  op_registry: {len(production_dialects)} production dialects")
    print(f"  type_registry: {total_types} types")
    print(f"  keywords: {len(KEYWORD_MAP)} keywords")


if __name__ == "__main__":
    main()
