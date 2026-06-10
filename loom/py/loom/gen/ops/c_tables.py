# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Op declarations -> C op tables, accessors, and builders.

Reads Op declarations from the Python DSL and emits C op metadata per dialect:

  ops.h      — enum + ISA macros + accessor macros + builder declarations
  builders.c — builder implementations (macros for common, explicit for complex)
  tables.c   — .rodata: B-string names, format arrays, descriptors, vtables

Public generated headers are checked into the repository for code archaeology
and editor/search ergonomics. Bulky generated C table sources are build outputs.

Usage:
    python3 loom/py/loom/gen/run.py c_tables --check
    python3 loom/py/loom/gen/run.py c_tables --in-place
    bazel run //loom/py/loom/gen/ops:c_tables_generator -- --dialect=check --builders=/tmp/builders.c --tables=/tmp/tables.c
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    StableKeyRef,
    SymbolRef,
    TemplateParam,
    TemplateParamFlags,
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
    ContractFamily,
    EffectKind,
    EnumDef,
    FuncLikeInterface,
    Op,
    OperandOwnershipEffect,
    RegionDef,
    ResultOwnershipEffect,
    TargetLikeInterface,
    TiedResult,
    TypeConstraint,
)
from loom.fields import FieldKind, FieldLayout, compute_layout
from loom.gen import bootstrap as _bootstrap
from loom.gen.ops import c_format, c_interfaces, c_queries, c_symbols
from loom.gen.ops.c_enum_attrs import SharedEnumMap
from loom.gen.ops.c_enum_attrs import (
    collect_shared_enums as _collect_shared_enums,
)
from loom.gen.ops.c_enum_attrs import (
    enum_c_type as _enum_c_type,
)
from loom.gen.ops.c_enum_attrs import (
    enum_case_c_ident as _enum_case_c_ident,
)
from loom.gen.ops.c_enum_attrs import (
    enum_names_array_name as _enum_names_array_name,
)
from loom.gen.ops.c_enums import (
    ATTR_KIND_MAP,
    CONSTRAINT_MAP,
    FIELD_CATEGORY_MAP,
    LOOM_FIELD_REF_MAX_INDEX,
    OPERAND_OWNERSHIP_EFFECT_MAP,
    OWNERSHIP_CARRIER_MAP,
    RESULT_OWNERSHIP_EFFECT_MAP,
    TRAIT_MAP,
    TYPE_CONSTRAINT_MAP,
)
from loom.gen.ops.c_enums import (
    error_ref_literal as _error_ref_literal,
)
from loom.gen.ops.c_names import (
    COPYRIGHT,
    GENERATED_HEADER,
)
from loom.gen.ops.c_names import (
    c_dialect_enum as _c_dialect_enum,
)
from loom.gen.ops.c_names import (
    c_dialect_include_path as _c_dialect_include_path,
)
from loom.gen.ops.c_names import (
    c_dialect_path as _c_dialect_path,
)
from loom.gen.ops.c_names import (
    c_enum_name as _c_enum_name,
)
from loom.gen.ops.c_names import (
    c_prefix as _c_prefix,
)
from loom.gen.ops.c_names import (
    guard_name as _guard_name,
)
from loom.gen.ops.keywords import generate_keyword_enum_inc, generate_keyword_table_inc
from loom.gen.ops.model import (
    DialectGeneration,
    GenerationModel,
    load_dialect_generation,
    load_generation_model,
)
from loom.gen.ops.type_registry import generate_type_registry
from loom.gen.support.c import c_identifier as _c_identifier
from loom.gen.support.c import c_string_literal as _c_string_literal
from loom.gen.support.files import write_text_file as _write_file
from loom.gen.support.generated_file import line_comment_header


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


def _emit_op_semantics(lines: list[str], op: Op) -> None:
    """Appends a sparse initializer for one op semantic metadata row."""

    contract_families = _contract_family_mask(op.contracts)
    lines.append("    {")
    lines.append(f"        .phase = {_op_phase_c_name(op)},")
    if contract_families != "0":
        lines.append(f"        .contract_families = {contract_families},")
    lines.append("    },")


# Maps Python symbol interface names to C interface flag constants.
SYMBOL_INTERFACE_MAP: dict[str, str] = {
    "func_like": "LOOM_SYMBOL_INTERFACE_FUNC_LIKE",
    "global": "LOOM_SYMBOL_INTERFACE_GLOBAL",
    "executable": "LOOM_SYMBOL_INTERFACE_EXECUTABLE",
    "record": "LOOM_SYMBOL_INTERFACE_RECORD",
    "target": "LOOM_SYMBOL_INTERFACE_TARGET",
    "config": "LOOM_SYMBOL_INTERFACE_CONFIG",
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
        and not op.ownership_effects
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


def _detect_builder_pattern(op: Op) -> str | None:
    """Detects if an op matches a standard builder macro pattern.

    Returns the macro name suffix or None for complex ops.
    """
    layout = compute_layout(op)
    non_flags = c_queries.non_flags_attrs(op)
    has_flags = c_queries.has_flags_attr(op)
    has_template_param = any(isinstance(e, TemplateParam | TemplateParamFlags) for e in _flatten_format(op.format))
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
    shared_enums: SharedEnumMap,
) -> str:
    """Returns the C builder parameter type for an attribute."""
    if attr_def.attr_type == "enum" and attr_def.enum_def is not None and not attr_def.optional:
        return _enum_c_type(op, attr_def, shared_enums)
    return _C_ATTR_TYPE_MAP.get(attr_def.attr_type, "loom_attribute_t")


def _extract_c_params(op: Op, shared_enums: SharedEnumMap) -> list[dict[str, Any]]:
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
                "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
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
    func_args_field_name = c_queries.func_args_field_name(op)
    func_like_body_region_name: str | None = None
    for interface in op.interfaces:
        if isinstance(interface, FuncLikeInterface):
            func_like_body_region_name = interface.body
            break

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
                                "optional": desc.optional,
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
                            "names_attr_index": c_queries.resolve_attr_index(op, names_field, "builder"),
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
                            "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
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
                            "static_attr_index": (c_queries.resolve_attr_index(op, static_field, "builder") if static_desc else 0),
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
                    arg_source = region_def.arg_source if region_def else None
                    func_args = _pending_func_args or arg_source == func_args_field_name or name == func_like_body_region_name
                    _pending_func_args = False
                    if arg_source == func_args_field_name:
                        arg_source = None
                    params.append(
                        {
                            "name": name,
                            "kind": "auto_region",
                            "region_index": layout.fields[name].index,
                            "optional": (region_def.optional if region_def else False),
                            "binding": binding,
                            "arg_source": arg_source,
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

                case TemplateParamFlags(param=param_name, flags=flags_name):
                    append_attr_param(param_name)
                    params.append(
                        {
                            "name": "instance_flags",
                            "kind": "instance_flags",
                            "c_type": "uint8_t",
                            "attr_name": flags_name,
                        }
                    )
                    covered_attrs.add(flags_name)

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
                                "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
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

                case DescriptorRef(key=name, ordinal=ordinal):
                    attr_def = op.attr(name)
                    if attr_def is None:
                        continue
                    params.append(
                        {
                            "name": name,
                            "kind": "descriptor_ref",
                            "c_type": _c_attr_param_type(op, attr_def, shared_enums),
                            "attr_type": attr_def.attr_type,
                            "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
                            "ordinal_attr_index": c_queries.resolve_attr_index(op, ordinal, "builder"),
                        }
                    )
                    covered_attrs.add(name)
                    covered_attrs.add(ordinal)

                case StableKeyRef(key=name, stable_id=stable_id):
                    attr_def = op.attr(name)
                    if attr_def is None:
                        continue
                    params.append(
                        {
                            "name": name,
                            "kind": "stable_key_ref",
                            "c_type": _c_attr_param_type(op, attr_def, shared_enums),
                            "attr_type": attr_def.attr_type,
                            "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
                            "stable_id_attr_index": c_queries.resolve_attr_index(op, stable_id, "builder"),
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
    if param["kind"] == "operand":
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


def _build_flags_storage_type(flag_count: int) -> str:
    """Returns the smallest public integer type that can hold all build flags."""
    if flag_count <= 32:
        return "uint32_t"
    if flag_count <= 64:
        return "uint64_t"
    raise ValueError(f"build has {flag_count} optional fields, which exceeds the 64-bit build flag capacity")


def _build_flag_bit_literal(flag_count: int) -> str:
    """Returns the C integer literal used for build flag enumerators."""
    return "UINT64_C(1)" if flag_count > 32 else "1u"


def _build_flag_bit_name(prefix: str, param: dict[str, object]) -> str:
    return f"{prefix.upper()}_BUILD_FLAG_HAS_{str(param['name']).upper()}"


def _generate_build_flags_declaration(prefix: str, params: list[dict[str, object]]) -> list[str]:
    flag_params = _build_flag_params(params)
    if not flag_params:
        return []
    flag_count = len(flag_params)
    storage_type = _build_flags_storage_type(flag_count)
    bit_literal = _build_flag_bit_literal(flag_count)

    lines = [f"enum {prefix}_build_flag_bits_e {{"]
    for index, param in enumerate(flag_params):
        lines.append(f"  {_build_flag_bit_name(prefix, param)} = {bit_literal} << {index},")
    lines.append("};")
    lines.append(f"typedef {storage_type} {_build_flags_type_name(prefix)};")
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
                c_params.append(f"{opt}{consume}loom_value_id_t {param['name']}")
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
            case "descriptor_ref" | "stable_key_ref":
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


def _generate_builder_declaration(op: Op, prefix: str, shared_enums: SharedEnumMap) -> list[str]:
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


def _emit_builder_i64_array_storage(
    lines: list[str],
    *,
    source: str,
    count: str,
    storage: str,
    op_name: str,
    field_name: str,
) -> None:
    """Emits C code that copies an i64-array attr payload into the builder."""
    lines.append(f"  int64_t* {storage} = NULL;")
    lines.append(f"  if ({count} > 0) {{")
    lines.append(f"    if (!{source}) {{")
    lines.append("      return iree_make_status(")
    lines.append("          IREE_STATUS_INVALID_ARGUMENT,")
    lines.append(f'          "{op_name} {field_name} storage is NULL for non-zero count");')
    lines.append("    }")
    lines.append("    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(")
    lines.append(f"        builder->arena, {count}, sizeof(*{storage}), (void**)&{storage}));")
    lines.append(f"    memcpy({storage}, {source}, {count} * sizeof(*{storage}));")
    lines.append("  }")


def _emit_builder_predicate_list_storage(
    lines: list[str],
    *,
    source: str,
    count: str,
    storage: str,
    op_name: str,
    field_name: str,
) -> None:
    """Emits C code that copies a predicate-list attr payload into the builder."""
    lines.append(f"  loom_predicate_t* {storage} = NULL;")
    lines.append(f"  if ({count} > 0) {{")
    lines.append(f"    if (!{source}) {{")
    lines.append("      return iree_make_status(")
    lines.append("          IREE_STATUS_INVALID_ARGUMENT,")
    lines.append(f'          "{op_name} {field_name} storage is NULL for non-zero count");')
    lines.append("    }")
    lines.append("    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(")
    lines.append(f"        builder->arena, {count}, sizeof(*{storage}), (void**)&{storage}));")
    lines.append(f"    memcpy({storage}, {source}, {count} * sizeof(*{storage}));")
    lines.append("  }")


def _generate_builder_implementation(
    op: Op,
    prefix: str,
    enum_name: str,
    shared_enums: SharedEnumMap,
) -> list[str]:
    """Generates the C builder function implementation for a complex op."""
    params = _extract_c_params(op, shared_enums)
    layout = compute_layout(op)
    func_args_are_operands = c_queries.func_args_are_operands(op)
    params_by_name = {str(param["name"]): param for param in params if "name" in param}
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
    if layout.segmented_operands:
        for param in params:
            if param["kind"] in (
                "operand_variadic",
                "binding_list",
                "operand_dict",
                "index_list",
            ) or (param["kind"] == "func_args" and func_args_are_operands):
                count_name = param["dynamic_field"] if param["kind"] == "index_list" else param["name"]
                _emit_builder_count_check(
                    lines,
                    count=f"{count_name}_count",
                    max_value="UINT16_MAX",
                    message=f"{op.name} operand segment '{count_name}' exceeds uint16_t range",
                )
    elif variadic_operand_param:
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

    optional_operand_params = [param for param in params if param["kind"] == "operand" and param.get("optional")]
    if layout.segmented_operands:
        lines.append(f"  uint16_t operand_segment_counts[{len(op.operands)}] = {{0}};")
        lines.append("  uint32_t operand_count_32 = 0;")
        for operand in op.operands:
            desc = layout.fields[operand.name]
            operand_param = params_by_name.get(operand.name)
            if operand.variadic:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': variadic operand '{operand.name}' has no builder parameter")
                count_name = operand_param["dynamic_field"] if operand_param["kind"] == "index_list" else operand_param["name"]
                lines.append(f"  operand_segment_counts[{desc.index}] = (uint16_t){count_name}_count;")
                lines.append(f"  operand_count_32 += (uint32_t){count_name}_count;")
            elif operand.optional:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': optional operand '{operand.name}' has no builder parameter")
                optional_flag = _build_flag_bit_name(prefix, operand_param)
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    operand_segment_counts[{desc.index}] = 1;")
                lines.append("    operand_count_32 += 1;")
                lines.append("  }")
            else:
                lines.append(f"  operand_segment_counts[{desc.index}] = 1;")
                lines.append("  operand_count_32 += 1;")
        lines.append("  if (operand_count_32 > UINT16_MAX) {")
        lines.append("    return iree_make_status(")
        lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
        lines.append(f'        "{op.name} operand count exceeds uint16_t range");')
        lines.append("  }")
        lines.append("  uint16_t operand_count = (uint16_t)operand_count_32;")
        operand_count_expr = "operand_count"
    elif optional_operand_params:
        optional_operand_params.sort(key=lambda param: param["index"])
        lines.append(f"  uint16_t operand_count = {fixed_operand_count};")
        previous_optional_flag = ""
        for param in optional_operand_params:
            optional_flag = _build_flag_bit_name(prefix, param)
            if previous_optional_flag:
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag}) &&")
                lines.append(f"      !iree_any_bit_set(build_flags, {previous_optional_flag})) {{")
                lines.append("    return iree_make_status(")
                lines.append("        IREE_STATUS_INVALID_ARGUMENT,")
                lines.append(f'        "{op.name} optional operand {param["name"]} requires preceding optional operand");')
                lines.append("  }")
            lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
            lines.append(f"    operand_count = {param['index'] + 1};")
            lines.append("  }")
            previous_optional_flag = optional_flag
        operand_count_expr = "operand_count"
    elif variadic_operand_param:
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
    attr_count = len(c_queries.non_flags_attrs(op))

    # Compute tied result count expression.
    static_ties = _static_tied_results(op)
    has_tied_param = any(p["kind"] == "tied_results" for p in params)
    if static_ties:
        tied_count_expr = str(len(static_ties))
    elif has_tied_param:
        tied_count_expr = "(uint16_t)tied_result_count"
    else:
        tied_count_expr = "0"

    if layout.segmented_operands:
        allocate_fn = "loom_builder_allocate_segmented_op_with_successors" if op.successors else "loom_builder_allocate_segmented_op"
    else:
        allocate_fn = "loom_builder_allocate_op_with_successors" if op.successors else "loom_builder_allocate_op"
    lines.append(f"  IREE_RETURN_IF_ERROR({allocate_fn}(")
    lines.append(f"      builder, {enum_name}, {operand_count_expr},")
    if layout.segmented_operands:
        lines.append("      operand_segment_counts, IREE_ARRAYSIZE(operand_segment_counts),")
    if op.successors:
        lines.append(f"      {result_count_expr}, {successor_count_expr}, {region_count_expr}, {tied_count_expr},")
    else:
        lines.append(f"      {result_count_expr}, {region_count_expr}, {tied_count_expr},")
    lines.append(f"      {attr_count}, location, out_op));")
    lines.extend(f"  (*out_op)->instance_flags = {param['name']};" for param in params if param["kind"] == "instance_flags")

    if layout.segmented_operands:
        lines.append("  uint16_t operand_offset = 0;")
        for operand in op.operands:
            operand_param = params_by_name.get(operand.name)
            if operand.variadic:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': variadic operand '{operand.name}' has no builder parameter")
                if operand_param["kind"] in ("operand_variadic", "binding_list"):
                    name = operand_param["name"]
                    lines.append(f"  if ({name}_count > 0) {{")
                    lines.append("    memcpy(loom_op_operands(*out_op) + operand_offset,")
                    lines.append(f"           {name}, {name}_count * sizeof(loom_value_id_t));")
                    lines.append("  }")
                    lines.append(f"  operand_offset += (uint16_t){name}_count;")
                elif operand_param["kind"] == "operand_dict":
                    name = operand_param["name"]
                    attr_index = operand_param["names_attr_index"]
                    lines.append("  IREE_RETURN_IF_ERROR(loom_builder_set_operand_dict(")
                    lines.append(f"      builder, loom_make_named_value_slice({name}, {name}_count),")
                    lines.append("      loom_op_operands(*out_op) + operand_offset,")
                    lines.append(f"      &loom_op_attrs(*out_op)[{attr_index}]));")
                    lines.append(f"  operand_offset += (uint16_t){name}_count;")
                elif operand_param["kind"] == "index_list":
                    dyn = operand_param["dynamic_field"]
                    lines.append(f"  if ({dyn}_count > 0) {{")
                    lines.append("    memcpy(loom_op_operands(*out_op) + operand_offset,")
                    lines.append(f"           {dyn}, {dyn}_count * sizeof(loom_value_id_t));")
                    lines.append("  }")
                    lines.append(f"  operand_offset += (uint16_t){dyn}_count;")
                elif operand_param["kind"] == "func_args" and func_args_are_operands:
                    lines.append("  for (iree_host_size_t _i = 0; _i < arg_types_count; ++_i) {")
                    lines.append("    loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
                    lines.append("    IREE_RETURN_IF_ERROR(")
                    lines.append("        loom_builder_define_value(builder, arg_types[_i], &_arg_id));")
                    lines.append("    loom_op_operands(*out_op)[operand_offset + _i] = _arg_id;")
                    lines.append("  }")
                    lines.append("  operand_offset += (uint16_t)arg_types_count;")
            elif operand.optional:
                if operand_param is None:
                    raise ValueError(f"Op '{op.name}': optional operand '{operand.name}' has no builder parameter")
                optional_flag = _build_flag_bit_name(prefix, operand_param)
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_operands(*out_op)[operand_offset++] = {operand.name};")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_operands(*out_op)[operand_offset++] = {operand.name};")
    else:
        # Fill in fixed operands.
        for param in params:
            if param["kind"] != "operand":
                continue
            if param.get("optional"):
                optional_flag = _build_flag_bit_name(prefix, param)
                lines.append(f"  if (iree_any_bit_set(build_flags, {optional_flag})) {{")
                lines.append(f"    loom_op_operands(*out_op)[{param['index']}] = {param['name']};")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_operands(*out_op)[{param['index']}] = {param['name']};")

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

    # Fill in fixed successors.
    lines.extend(f"  loom_op_successors(*out_op)[{param['index']}] = {param['name']};" for param in params if param["kind"] == "successor")

    def implicit_block_arg_type_expr(arg_type_kw: str) -> str:
        scalar_type = _IMPLICIT_ARG_TYPE_MAP.get(arg_type_kw)
        if scalar_type is not None:
            return f"loom_type_scalar({scalar_type})"
        if arg_type_kw.startswith("type_of:"):
            source_field = arg_type_kw.removeprefix("type_of:")
            source_desc = layout.fields.get(source_field)
            if source_desc is None or source_desc.kind != FieldKind.OPERAND:
                raise ValueError(f"Op '{op.name}': implicit arg type source '{source_field}' must be an operand field")
            if source_desc.variadic or source_desc.optional:
                raise ValueError(f"Op '{op.name}': implicit arg type source '{source_field}' must be a required fixed operand")
            source_param = params_by_name.get(source_field)
            if source_param is None or source_param.get("kind") != "operand":
                raise ValueError(f"Op '{op.name}': implicit arg type source '{source_field}' must be a fixed operand")
            return f"loom_module_value_type(builder->module, {source_field})"
        raise ValueError(f"Unknown implicit arg type: {arg_type_kw}")

    def emit_define_implicit_block_arg(indent: str, arg_type_kw: str) -> None:
        scalar_type = _IMPLICIT_ARG_TYPE_MAP.get(arg_type_kw)
        lines.append(f"{indent}{{")
        lines.append(f"{indent}  loom_value_id_t _arg_id = LOOM_VALUE_ID_INVALID;")
        lines.append(f"{indent}  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(")
        lines.append(f"{indent}      builder, _block,")
        if scalar_type is not None:
            lines.append(f"{indent}      loom_type_scalar({scalar_type}), &_arg_id));")
        else:
            lines.append(f"{indent}      {implicit_block_arg_type_expr(arg_type_kw)}, &_arg_id));")
        lines.append(f"{indent}}}")

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
            emit_define_implicit_block_arg("    ", arg_type_kw)
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
            emit_define_implicit_block_arg(inner_indent, arg_type_kw)

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
                emit_define_implicit_block_arg("    ", arg_type_kw)
        lines.append(f"    loom_op_regions(*out_op)[{param['case_region_index']} + _case] = _region;")
        lines.append("  }")

    # Fill in attributes.
    for param in params:
        if param["kind"] == "predicate_list":
            idx = param["attr_index"]
            name = param["name"]
            storage = f"_{name}_storage"
            _emit_builder_predicate_list_storage(
                lines,
                source=name,
                count=f"{name}_count",
                storage=storage,
                op_name=op.name,
                field_name=name,
            )
            if param.get("optional"):
                lines.append(f"  if ({name}_count > 0) {{")
                lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_predicate_list({storage}, (uint16_t){name}_count);")
                lines.append("  }")
            else:
                lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_predicate_list({storage}, (uint16_t){name}_count);")
        elif param["kind"] == "index_list":
            static_field = param["static_field"]
            idx = param["static_attr_index"]
            storage = f"_{static_field}_storage"
            _emit_builder_i64_array_storage(
                lines,
                source=static_field,
                count=f"{static_field}_count",
                storage=storage,
                op_name=op.name,
                field_name=static_field,
            )
            lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array({storage}, (uint16_t){static_field}_count);")
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
                storage = f"_{name}_storage"
                _emit_builder_i64_array_storage(
                    lines,
                    source=name,
                    count=f"{name}_count",
                    storage=storage,
                    op_name=op.name,
                    field_name=name,
                )
                if is_optional:
                    lines.append(f"  if ({name}_count > 0) {{")
                    lines.append(f"    loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array({storage}, (uint16_t){name}_count);")
                    lines.append("  }")
                else:
                    lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_i64_array({storage}, (uint16_t){name}_count);")
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
            ordinal_idx = param["ordinal_attr_index"]
            name = param["name"]
            lines.append(f"  loom_op_attrs(*out_op)[{idx}] = loom_attr_string({name});")
            lines.append(f"  loom_op_attrs(*out_op)[{ordinal_idx}] = loom_attr_i64(-1);")
        elif param["kind"] == "stable_key_ref":
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
            generator="loom.gen.ops.c_tables",
            regenerate="python3 loom/py/loom/gen/run.py c_tables --in-place",
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
            if layout.segmented_operands:
                if operand.variadic:
                    lines.append(f"LOOM_DEFINE_SEGMENTED_OPERANDS({prefix}_{operand.name}, {desc.index})")
                elif operand.optional:
                    lines.append(f"LOOM_DEFINE_SEGMENTED_OPTIONAL_OPERAND({prefix}_{operand.name}, {desc.index})")
                else:
                    lines.append(f"LOOM_DEFINE_SEGMENTED_OPERAND({prefix}_{operand.name}, {desc.index})")
            elif operand.variadic:
                lines.append(f"LOOM_DEFINE_VARIADIC_OPERANDS({prefix}_{operand.name}, {desc.index})")
            elif operand.optional:
                lines.append(f"LOOM_DEFINE_OPTIONAL_OPERAND({prefix}_{operand.name}, {desc.index})")
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
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    if private_header:
        lines.append(f'#include "{include_path}/tables.h"')
    else:
        lines.append(f'#include "{include_path}/ops.h"')
    if c_interfaces.target_like_bundle_table_symbols(ops):
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
    symbol_fact_domain_symbols = sorted({fact_domain for op in ops if op.symbol_def is not None if (fact_domain := c_symbols.symbol_fact_domain_symbol(op)) is not None})
    if symbol_fact_domain_symbols:
        lines.extend(f"extern const loom_symbol_fact_domain_t {fact_domain};" for fact_domain in symbol_fact_domain_symbols)
        lines.append("")
    interface_c_ptr_symbols = c_interfaces.interface_c_ptr_symbols(ops)
    if interface_c_ptr_symbols:
        lines.extend(f"extern const {c_type} {symbol};" for c_type, symbol in interface_c_ptr_symbols)
        lines.append("")
    target_like_bundle_table_symbols = c_interfaces.target_like_bundle_table_symbols(ops)
    if target_like_bundle_table_symbols:
        lines.extend(f"extern const loom_target_bundle_table_t {symbol};" for symbol in target_like_bundle_table_symbols)
        lines.append("")

    emitted_enum_case_name_arrays: set[str] = set()

    # Op metadata blocks.
    for op in ops:
        prefix = _c_prefix(op)
        layout = compute_layout(op)
        elements = c_format.translate_format_elements(op)
        non_flags = c_queries.non_flags_attrs(op)
        has_flags = c_queries.has_flags_attr(op)

        # Format element array.
        if elements:
            lines.append(f"static const loom_format_element_t {prefix}_format[] = {{")
            for kind, field_index, data in elements:
                lines.append(f"    {{{kind}, {field_index}, {data}}},")
            lines.append("};")

        # Operand descriptors.
        func_args_are_operands = c_queries.func_args_are_operands(op)
        explicit_func_args_operand = c_queries.explicit_func_args_operand(op)
        synthesize_func_args_operand = func_args_are_operands and explicit_func_args_operand is None
        if op.operands or synthesize_func_args_operand:
            func_args_name = c_queries.func_args_field_name(op) if synthesize_func_args_operand else ""
            effect_map = {effect.operand: effect.kind for effect in op.effects}
            ownership_operand_map = {effect.operand: effect for effect in op.ownership_effects if isinstance(effect, OperandOwnershipEffect)}
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
                ownership_effect = ownership_operand_map.get(operand.name)
                if ownership_effect is None:
                    ownership_effect_name = "LOOM_OPERAND_OWNERSHIP_NONE"
                    ownership_carrier_name = "LOOM_OWNERSHIP_CARRIER_NONE"
                else:
                    ownership_effect_name = OPERAND_OWNERSHIP_EFFECT_MAP[ownership_effect.kind]
                    ownership_carrier_name = OWNERSHIP_CARRIER_MAP[ownership_effect.carrier]
                if ownership_effect is None:
                    lines.append(f"    {{{_bstring_expr(operand.name)}, {type_constraint}, {flags}}},")
                else:
                    lines.append(f"    {{{_bstring_expr(operand.name)}, {type_constraint}, {flags}, {ownership_effect_name}, {ownership_carrier_name}}},")
            if synthesize_func_args_operand:
                lines.append(f"    {{{_bstring_expr(func_args_name)}, LOOM_TYPE_CONSTRAINT_ANY, LOOM_OPERAND_VARIADIC}},")
            lines.append("};")

        # Result descriptors.
        if op.results:
            ownership_result_map = {effect.result: effect for effect in op.ownership_effects if isinstance(effect, ResultOwnershipEffect)}
            lines.append(f"static const loom_result_descriptor_t {prefix}_result_desc[] = {{")
            for result in op.results:
                type_constraint = TYPE_CONSTRAINT_MAP[result.type_constraint]
                flags_parts = []
                if result.variadic:
                    flags_parts.append("LOOM_RESULT_VARIADIC")
                if result.allocates:
                    flags_parts.append("LOOM_RESULT_ALLOCATES")
                flags = " | ".join(flags_parts) if flags_parts else "0"
                result_ownership_effect = ownership_result_map.get(result.name)
                source_operand_index = "LOOM_RESULT_OWNERSHIP_SOURCE_FIELD_NONE"
                if result_ownership_effect is not None:
                    ownership_effect_name = RESULT_OWNERSHIP_EFFECT_MAP[result_ownership_effect.kind]
                    if result_ownership_effect.source is not None:
                        source_operand_index = str(c_queries.resolve_ownership_source_operand_index(op, result_ownership_effect.source))
                else:
                    ownership_effect_name = "LOOM_RESULT_OWNERSHIP_NONE"
                if result_ownership_effect is None:
                    lines.append(f"    {{{_bstring_expr(result.name)}, {type_constraint}, {flags}}},")
                else:
                    lines.append(f"    {{{_bstring_expr(result.name)}, {type_constraint}, {flags}, {ownership_effect_name}, {source_operand_index}}},")
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
            func_args_fields = c_queries.func_args_field_names(op)
            for region_def in op.regions:
                region_flags = []
                if region_def.single_block:
                    region_flags.append("LOOM_REGION_SINGLE_BLOCK")
                if region_def.optional:
                    region_flags.append("LOOM_REGION_OPTIONAL")
                if region_def.arg_source in func_args_fields:
                    region_flags.append("LOOM_REGION_PROJECT_FUNC_ARGS")
                buffer_arg_memory_space = region_def.buffer_arg_memory_space
                if buffer_arg_memory_space is not None:
                    if buffer_arg_memory_space != "global":
                        raise ValueError(f"Op '{op.name}' region '{region_def.name}' has unsupported buffer_arg_memory_space '{buffer_arg_memory_space}'")
                    region_flags.append("LOOM_REGION_GLOBAL_BUFFER_ARGS")
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

        target_like_iface = c_queries.find_interface(op, TargetLikeInterface)
        if target_like_iface is not None:
            c_interfaces.emit_target_like_descriptor(op, target_like_iface, lines)

        # Interface vtables.
        for spec in c_interfaces.INTERFACES:
            c_interfaces.emit_interface_vtable(op, spec, lines)

        # Symbol definition descriptor.
        if op.symbol_def is not None:
            attr_index = c_queries.resolve_attr_index(op, op.symbol_def.field, "symbol_def")
            flags = _symbol_interface_flags(op.symbol_def.interfaces)
            fact_domain = c_symbols.symbol_fact_domain_symbol(op)
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
        if layout.segmented_operands:
            vtable_flag_bits.append("LOOM_OP_VTABLE_SEGMENTED_OPERANDS")
        elif layout.variadic_operand or c_queries.func_args_are_operands(op):
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_OPERANDS")
        if layout.variadic_result:
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_RESULTS")
        if layout.variadic_region:
            vtable_flag_bits.append("LOOM_OP_VTABLE_VARIADIC_REGIONS")
        if has_flags:
            vtable_flag_bits.append("LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS")
        if c_queries.op_has_type_propagation_candidate(op, layout):
            vtable_flag_bits.append("LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE")
        vtable_flags_str = " | ".join(vtable_flag_bits) if vtable_flag_bits else "0"

        sym_kind = _symbol_kind(op)
        canon = op.canonicalize or "NULL"
        infer_facts_fn = op.facts or "NULL"
        type_transfer_fn = op.type_transfer or "NULL"
        verify_fn = op.verify or "NULL"
        eff_traits = op.effective_traits or "NULL"
        interface_ptrs = {spec.vtable_field: c_interfaces.interface_vtable_ptr(op, spec) for spec in c_interfaces.INTERFACES}
        symbol_def_ptr = f"&{prefix}_symbol_def" if op.symbol_def is not None else "NULL"
        has_placement = any(trait.name in ("HasParent", "HasAncestor", "NoAncestor") for trait in op.traits)
        placement_ptr = f"&{prefix}_placement" if has_placement else "NULL"
        attr_desc_ptr = f"{prefix}_attr_desc" if non_flags else "NULL"
        operand_desc_ptr = f"{prefix}_operand_desc" if op.operands or c_queries.func_args_are_operands(op) else "NULL"
        operand_descriptor_count = len(op.operands)
        if c_queries.func_args_are_operands(op) and c_queries.explicit_func_args_operand(op) is None:
            operand_descriptor_count += 1
        successor_selector_operand_index = c_queries.resolve_successor_selector_operand_index(op)
        implied_operand_descriptor_count = layout.fixed_operand_count
        if layout.segmented_operands:
            implied_operand_descriptor_count = -1
        elif layout.variadic_operand or c_queries.func_args_are_operands(op):
            implied_operand_descriptor_count += 1
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
        if operand_desc_ptr != "NULL" and operand_descriptor_count != implied_operand_descriptor_count:
            lines.append(f"    .operand_descriptor_count = IREE_ARRAYSIZE({operand_desc_ptr}),")
        append_nonzero("fixed_result_count", layout.fixed_result_count)
        append_nonzero("vtable_flags", vtable_flags_str)
        if successor_selector_operand_index is not None:
            lines.append("    .control_flow_flags = LOOM_OP_CONTROL_FLOW_HAS_SUCCESSOR_SELECTOR,")
            lines.append(f"    .successor_selector_operand_index = {successor_selector_operand_index},")
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
        for spec in c_interfaces.INTERFACES:
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
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
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
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
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
    lines.extend(line_comment_header("//", generator="loom.gen.ops.c_tables"))
    lines.append("// clang-format off")
    lines.append("")
    include_path = include_path or f"loom/ops/{dialect_name}"
    lines.append(f'#include "{include_path}/ops.h"')
    lines.append("")
    lines.append("#include <string.h>")
    lines.append("")
    lines.append('#include "loom/ir/module.h"')
    lines.append('#include "loom/ops/builder_macros.h"')
    if any(isinstance(element, StableKeyRef) for op in ops for element in _flatten_format(op.format)):
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
# CLI
# ============================================================================


@dataclass(frozen=True)
class NamedOutput:
    """One named generated output path from the CLI."""

    name: str
    path: Path


def _parse_named_output(value: str) -> NamedOutput:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return NamedOutput(name=name, path=Path(path))


def _generate_dialect_contents(generation: DialectGeneration) -> dict[str, str]:
    """Returns generated file contents keyed relative to the dialect C directory."""
    dialect = generation.dialect
    include_path = _c_dialect_include_path(dialect)
    table_files = (
        generate_sharded_tables_c(
            dialect.name,
            dialect.dialect_id,
            generation.table_shards,
            include_path=include_path,
        )
        if generation.table_shards is not None
        else {
            "tables.c": generate_tables_c(
                dialect.name,
                dialect.dialect_id,
                generation.ops,
                include_path=include_path,
            )
        }
    )
    return {
        "ops.h": generate_ops_h(dialect.name, dialect.dialect_id, generation.ops),
        "builders.c": generate_builders_c(dialect.name, generation.ops, include_path=include_path),
        **table_files,
    }


def _production_dialects(model: GenerationModel) -> list[tuple[Any, list[Op]]]:
    return [(generation.dialect, generation.ops) for generation in model.dialects if generation.dialect.register_by_default]


def _generate_registry_contents(model: GenerationModel) -> dict[str, str]:
    op_reg_h, op_reg_c = generate_op_registry(_production_dialects(model))
    type_reg_h, type_reg_c = generate_type_registry(model.types)
    return {
        "op_registry.h": op_reg_h,
        "op_registry.c": op_reg_c,
        "type_registry.h": type_reg_h,
        "type_registry.c": type_reg_c,
        "keyword_enum.inc": generate_keyword_enum_inc(),
        "keyword_table.inc": generate_keyword_table_inc(),
    }


def _checked_in_output_contents(model: GenerationModel) -> dict[Path, str]:
    output_root = _bootstrap.REPO_ROOT / "loom" / "src" / "loom"
    outputs: dict[Path, str] = {}
    for generation in model.dialects:
        dialect_dir = output_root / _c_dialect_path(generation.dialect)
        contents = _generate_dialect_contents(generation)
        outputs[dialect_dir / "ops.h"] = contents["ops.h"]
        if "tables.h" in contents:
            outputs[dialect_dir / "tables.h"] = contents["tables.h"]

    registry_contents = _generate_registry_contents(model)
    registry_dir = output_root / "ops"
    for filename in (
        "op_registry.h",
        "type_registry.h",
        "keyword_enum.inc",
        "keyword_table.inc",
    ):
        outputs[registry_dir / filename] = registry_contents[filename]
    return outputs


def _build_generated_source_paths(model: GenerationModel) -> list[Path]:
    output_root = _bootstrap.REPO_ROOT / "loom" / "src" / "loom"
    paths: list[Path] = []
    for generation in model.dialects:
        dialect_dir = output_root / _c_dialect_path(generation.dialect)
        contents = _generate_dialect_contents(generation)
        paths.extend(dialect_dir / filename for filename in contents if filename.endswith(".c"))
    registry_dir = output_root / "ops"
    paths.extend(
        [
            registry_dir / "op_registry.c",
            registry_dir / "type_registry.c",
        ]
    )
    return sorted(paths)


def _check_checked_in_outputs(model: GenerationModel) -> int:
    failures: list[str] = []
    for path, expected in sorted(_checked_in_output_contents(model).items()):
        if not path.exists():
            failures.append(f"{path.relative_to(_bootstrap.REPO_ROOT)}: missing generated file")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            failures.append(f"{path.relative_to(_bootstrap.REPO_ROOT)}: stale generated file")

    failures.extend(f"{path.relative_to(_bootstrap.REPO_ROOT)}: generated C source must be a build output" for path in _build_generated_source_paths(model) if path.exists())

    if failures:
        print("c table generation check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print("regenerate with python3 loom/py/loom/gen/run.py c_tables --in-place", file=sys.stderr)
        return 1

    print(f"checked {len(_checked_in_output_contents(model))} generated C table headers")
    return 0


def _update_checked_in_outputs(model: GenerationModel) -> int:
    for path, content in _checked_in_output_contents(model).items():
        _write_file(path, content)
    print(f"updated {len(_checked_in_output_contents(model))} generated C table headers")
    return 0


def _set_output(parser: argparse.ArgumentParser, outputs: dict[str, Path], name: str, path: Path | None) -> None:
    if path is None:
        return
    if name in outputs:
        parser.error(f"duplicate output for {name}")
    outputs[name] = path


def _generate_selected_outputs(
    parser: argparse.ArgumentParser,
    contents: Mapping[str, str],
    outputs: Mapping[str, Path],
) -> int:
    if not outputs:
        parser.error("at least one output path is required")
    unknown_outputs = sorted(name for name in outputs if name not in contents)
    if unknown_outputs:
        parser.error(f"unknown generated output(s): {', '.join(unknown_outputs)}")
    for name, path in outputs.items():
        _write_file(path, contents[name])
    return 0


def _main_build_output_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.dialect:
        try:
            generation = load_dialect_generation(args.dialect)
        except ValueError as exc:
            parser.error(str(exc))

        outputs: dict[str, Path] = {}
        _set_output(parser, outputs, "ops.h", args.ops_header)
        _set_output(parser, outputs, "builders.c", args.builders)
        _set_output(parser, outputs, "tables.c", args.tables)
        _set_output(parser, outputs, "tables.h", args.table_header)
        for output in args.table_shard:
            _set_output(parser, outputs, f"tables/{output.name}.c", output.path)
        return _generate_selected_outputs(parser, _generate_dialect_contents(generation), outputs)

    outputs = {}
    _set_output(parser, outputs, "op_registry.h", args.op_registry_header)
    _set_output(parser, outputs, "op_registry.c", args.op_registry_source)
    _set_output(parser, outputs, "type_registry.h", args.type_registry_header)
    _set_output(parser, outputs, "type_registry.c", args.type_registry_source)
    _set_output(parser, outputs, "keyword_enum.inc", args.keyword_enum)
    _set_output(parser, outputs, "keyword_table.inc", args.keyword_table)
    model = load_generation_model()
    return _generate_selected_outputs(parser, _generate_registry_contents(model), outputs)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate C tables for Loom dialects and registries."""
    parser = argparse.ArgumentParser(description="Generate Loom C op tables from Python definitions.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify checked-in generated headers are current and generated C sources are absent.",
    )
    mode.add_argument(
        "--in-place",
        action="store_true",
        help="Regenerate checked-in generated headers.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--dialect", help="Generate selected outputs for one dialect.")
    target.add_argument(
        "--registry",
        action="store_true",
        help="Generate selected cross-dialect registry outputs.",
    )
    parser.add_argument("--ops-header", type=Path, help="Generated dialect ops.h path.")
    parser.add_argument("--builders", type=Path, help="Generated dialect builders.c path.")
    parser.add_argument("--tables", type=Path, help="Generated dialect tables.c path.")
    parser.add_argument("--table-header", type=Path, help="Generated sharded-dialect tables.h path.")
    parser.add_argument(
        "--table-shard",
        action="append",
        default=[],
        metavar="NAME=PATH",
        type=_parse_named_output,
        help="Generated sharded-dialect tables/NAME.c path.",
    )
    parser.add_argument("--op-registry-header", type=Path, help="Generated op_registry.h path.")
    parser.add_argument("--op-registry-source", type=Path, help="Generated op_registry.c path.")
    parser.add_argument("--type-registry-header", type=Path, help="Generated type_registry.h path.")
    parser.add_argument("--type-registry-source", type=Path, help="Generated type_registry.c path.")
    parser.add_argument("--keyword-enum", type=Path, help="Generated keyword_enum.inc path.")
    parser.add_argument("--keyword-table", type=Path, help="Generated keyword_table.inc path.")
    args = parser.parse_args(argv)

    build_output_selected = args.dialect is not None or args.registry
    header_mode_selected = args.check or args.in_place
    if build_output_selected and header_mode_selected:
        parser.error("build-output generation cannot be combined with --check or --in-place")
    if not build_output_selected and not header_mode_selected:
        parser.error("select --check, --in-place, --dialect, or --registry")

    if args.check:
        return _check_checked_in_outputs(load_generation_model())
    if args.in_place:
        return _update_checked_in_outputs(load_generation_model())
    return _main_build_output_mode(parser, args)


if __name__ == "__main__":
    sys.exit(main())
