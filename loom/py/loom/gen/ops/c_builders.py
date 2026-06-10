# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C builder declaration and source generation for Loom ops."""

from __future__ import annotations

from collections.abc import Sequence
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
    FuncLikeInterface,
    Op,
    RegionDef,
    TiedResult,
)
from loom.fields import FieldKind, FieldLayout, compute_layout
from loom.gen.ops import c_queries
from loom.gen.ops.c_enum_attrs import SharedEnumMap
from loom.gen.ops.c_enum_attrs import (
    collect_shared_enums as _collect_shared_enums,
)
from loom.gen.ops.c_enum_attrs import (
    enum_c_type as _enum_c_type,
)
from loom.gen.ops.c_names import COPYRIGHT
from loom.gen.ops.c_names import (
    c_enum_name as _c_enum_name,
)
from loom.gen.ops.c_names import (
    c_prefix as _c_prefix,
)
from loom.gen.support.generated_file import line_comment_header

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


def generate_builder_header_lines(op: Op, shared_enums: SharedEnumMap) -> list[str]:
    """Generates build flag and builder function declarations for ops.h."""
    prefix = _c_prefix(op)
    params = _extract_c_params(op, shared_enums)
    lines = _generate_build_flags_declaration(prefix, params)
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
