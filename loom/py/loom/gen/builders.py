# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Op declarations → typed Python builder stubs.

Reads Op declarations and emits builders.py files with typed methods.
Parameter order follows the format spec. Each method has full type
annotations, docstrings, and handles tied results automatically
for ops with fixed tying patterns.

Usage:
    python3 loom/py/loom/gen/run.py builders

Generates: dialect/*/builders.py
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loom.assembly import (
    Attr,
    AttrDict,
    BindingList,
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
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TemplateParam,
    TypeOf,
    TypesOf,
)
from loom.assembly import (
    Region as RegionFmt,
)
from loom.dsl import AttrDef, Op
from loom.fields import FieldKind, compute_layout
from loom.gen import bootstrap as _bootstrap

_PYTHON_KEYWORDS = frozenset(
    {
        "yield",
        "return",
        "for",
        "if",
        "else",
        "while",
        "import",
        "from",
        "class",
        "def",
        "pass",
        "break",
        "continue",
        "and",
        "or",
        "not",
        "in",
        "is",
    }
)


# ============================================================================
# Parameter extraction from format specs
# ============================================================================


def _extract_params(op: Op) -> list[dict[str, Any]]:
    """Extract builder parameters from an op's format spec.

    Walks the format elements in order and produces a parameter
    descriptor for each element that maps to a builder parameter.
    The order matches the textual assembly.
    """
    layout = compute_layout(op)
    params: list[dict[str, Any]] = []
    covered_attrs: set[str] = set()

    def append_attr_param(name: str) -> None:
        attr_def = op.attr(name)
        type_hint = _attr_type_hint(attr_def)
        is_optional = attr_def.optional if attr_def else False
        params.append(
            {
                "name": name,
                "kind": "attr",
                "type_hint": type_hint,
                "optional": is_optional,
                "attr_def": attr_def,
                "doc": attr_def.doc if attr_def else "",
            }
        )
        covered_attrs.add(name)

    def walk(elements: tuple[FormatElement, ...]) -> None:
        for element in elements:
            match element:
                case Ref(field=name):
                    if name in ("iv",):
                        # Implicit field — not a builder parameter.
                        continue
                    desc = layout.fields.get(name)
                    if desc and desc.kind == FieldKind.OPERAND:
                        params.append(
                            {
                                "name": name,
                                "kind": "operand",
                                "type_hint": "ValueRef",
                                "doc": f"Operand: {name}",
                            }
                        )

                case Refs(field=name):
                    params.append(
                        {
                            "name": name,
                            "kind": "operand_variadic",
                            "type_hint": "list[ValueRef]",
                            "doc": f"Variadic operands: {name}",
                        }
                    )

                case Attr(field=name):
                    append_attr_param(name)

                case SymbolRef(field=name):
                    params.append(
                        {
                            "name": name,
                            "kind": "attr",
                            "type_hint": "str",
                            "doc": f"Symbol reference: {name}",
                        }
                    )
                    covered_attrs.add(name)

                case IndexList(dynamic=dynamic_field, static=static_field):
                    params.append(
                        {
                            "name": dynamic_field,
                            "kind": "index_list",
                            "type_hint": "list[int | ValueRef]",
                            "static_field": static_field,
                            "doc": f"Index list: {dynamic_field}",
                        }
                    )
                    covered_attrs.add(static_field)

                case BindingList(field=name, kind=binding_kind):
                    params.append(
                        {
                            "name": name,
                            "kind": "binding_list",
                            "type_hint": "list[ValueRef]",
                            "binding_kind": binding_kind,
                            "doc": f"Binding list: {name}",
                        }
                    )

                case OperandDict(operands=operand_field, names=names_field):
                    params.append(
                        {
                            "name": operand_field,
                            "kind": "operand_dict",
                            "type_hint": "dict[str, ValueRef]",
                            "names_field": names_field,
                            "doc": f"Operand dictionary: {operand_field}",
                        }
                    )
                    covered_attrs.add(names_field)

                case ResultType(field=name):
                    params.append(
                        {
                            "name": "results",
                            "kind": "result_types",
                            "type_hint": "list[Type | TiedResultSpec]",
                            "doc": "Result type",
                        }
                    )

                case ResultTypeList(field=name):
                    params.append(
                        {
                            "name": "results",
                            "kind": "result_types",
                            "type_hint": "list[Type | TiedResultSpec]",
                            "doc": "Result types (use tied() for in-place results)",
                        }
                    )

                case RegionFmt(field=name):
                    params.append(
                        {
                            "name": name,
                            "kind": "region",
                            "type_hint": "Region | None",
                            "doc": f"Region: {name}",
                        }
                    )

                case OptionalGroup(elements=inner, anchor=_anchor):
                    walk(inner)

                case Scope(elements=inner):
                    walk(inner)

                case FuncArgs(field=name):
                    params.append(
                        {
                            "name": name,
                            "kind": "func_args",
                            "type_hint": "list[ValueRef]",
                            "optional": True,
                            "doc": f"Function signature args: {name}",
                        }
                    )

                case Flags(field=name):
                    attr_def = op.attr(name)
                    params.append(
                        {
                            "name": name,
                            "kind": "flags",
                            "type_hint": "str",
                            "optional": True,
                            "attr_def": attr_def,
                            "doc": attr_def.doc if attr_def else "Instance flags.",
                        }
                    )
                    covered_attrs.add(name)

                case OpRef(field=name):
                    append_attr_param(name)

                case TemplateParam(field=name):
                    append_attr_param(name)

                case PredicateList(field=name):
                    attr_def = op.attr(name)
                    is_optional = attr_def.optional if attr_def else False
                    params.append(
                        {
                            "name": name,
                            "kind": "predicate_list",
                            "type_hint": "list[Predicate]",
                            "optional": is_optional,
                            "doc": f"Predicate list: {name}",
                        }
                    )
                    covered_attrs.add(name)

                case AttrDict(field=name):
                    if name:
                        append_attr_param(name)
                    else:
                        for attr_def in op.attrs:
                            if attr_def.attr_type == "flags":
                                continue
                            if attr_def.name in covered_attrs:
                                continue
                            append_attr_param(attr_def.name)

                case Keyword() | TypeOf() | TypesOf() | Glue():
                    pass  # Not parameters.

    walk(op.format)
    return params


def _attr_type_hint(attr_def: AttrDef | None) -> str:
    """Get Python type hint for an attribute type."""
    if attr_def is None:
        return "Any"
    match attr_def.attr_type:
        case "i64":
            return "int"
        case "f64":
            return "float"
        case "string":
            return "str"
        case "bool":
            return "bool"
        case "enum":
            return "str"
        case "symbol":
            return "str"
        case "i64_array":
            return "list[int]"
        case "any":
            return "Any"
        case "dict":
            return "Mapping[str, Any]"
        case _:
            return "Any"


# ============================================================================
# Code generation
# ============================================================================


def generate_builders(ops: Sequence[Op], class_name: str) -> str:
    """Generate a builder class with typed methods for a set of ops.

    Returns the Python source code as a string.
    """
    # Generate method bodies first so we can scan for used imports.
    method_names = _builder_method_names(ops)
    method_lines: list[str] = []
    for op, method_name in zip(ops, method_names, strict=True):
        method_lines.append("")
        method_lines.extend(_generate_method(op, method_name))

    # Scan generated code to determine which imports are actually needed.
    body_text = "\n".join(method_lines)
    builder_imports = ["IRBuilder", "ValueRef"]
    if "IndexedValue" in body_text:
        builder_imports.append("IndexedValue")
    if "TiedResultSpec" in body_text:
        builder_imports.append("TiedResultSpec")
    if "tied(" in body_text:
        builder_imports.append("tied")

    # Scan for Predicate usage.
    needs_predicate = "list[Predicate]" in body_text
    needs_mapping = "Mapping[str, Any]" in body_text

    lines: list[str] = []
    lines.append("# GENERATED by loom.gen.builders — do not edit.")
    lines.append("#")
    lines.append("# Regenerate with: python3 loom/py/loom/gen/run.py builders")
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import builtins")
    lines.append("")
    if needs_mapping:
        lines.append("from collections.abc import Mapping")
        lines.append("")
    lines.append("from typing import Any, cast")
    lines.append("")
    lines.append(f"from loom.builder import {', '.join(sorted(builder_imports))}")
    ir_imports = ["Region", "Type"]
    if needs_predicate:
        ir_imports.append("Predicate")
    lines.append(f"from loom.ir import {', '.join(sorted(ir_imports))}")
    lines.append("")
    lines.append("")
    lines.append(f"class {class_name}:")
    lines.append(f'    """Typed builder methods for {class_name.replace("Builders", "").lower()} ops."""')
    lines.append("")
    lines.append("    __test__ = False")
    lines.append("")
    lines.append("    def __init__(self, builder: IRBuilder) -> None:")
    lines.append("        self._b = builder")
    lines.extend(method_lines)

    return "\n".join(lines) + "\n"


def _builder_method_names(ops: Sequence[Op]) -> list[str]:
    """Return unique Python method names for builder ops.

    Most ops use the final op-name segment (`scalar.addi` -> `addi`). When
    that segment collides within a dialect, use the full dialect-local suffix
    (`vector.load.mask` -> `load_mask`) so dotted op families remain distinct.
    """
    short_name_counts: dict[str, int] = {}
    for op in ops:
        short_name_counts[op.short_name] = short_name_counts.get(op.short_name, 0) + 1

    method_names: list[str] = []
    used_by: dict[str, str] = {}
    for op in ops:
        if short_name_counts[op.short_name] == 1:
            method_name = op.short_name
        else:
            dialect_separator = op.name.find(".")
            local_name = op.name[dialect_separator + 1 :] if dialect_separator >= 0 else op.name
            method_name = local_name.replace(".", "_")
        if method_name in _PYTHON_KEYWORDS:
            method_name = method_name + "_"
        prior_op_name = used_by.get(method_name)
        if prior_op_name is not None:
            raise ValueError(f"Python builder method name collision: {prior_op_name!r} and {op.name!r} both map to {method_name!r}")
        used_by[method_name] = op.name
        method_names.append(method_name)
    return method_names


def _generate_method(op: Op, method_name: str) -> list[str]:
    """Generate one builder method for an op."""
    params = _extract_params(op)

    # Build parameter list.
    keyword_param_strs: list[str] = []
    for param in params:
        if param["kind"] == "result_types":
            keyword_param_strs.append(f"{param['name']}: {param['type_hint']}")
        elif param["kind"] == "region":
            keyword_param_strs.append(f"{param['name']}: {param['type_hint']} = None")
        elif param.get("optional"):
            keyword_param_strs.append(f"{param['name']}: {param['type_hint']} | None = None")
        else:
            keyword_param_strs.append(f"{param['name']}: {param['type_hint']}")

    # Determine return type.
    result_count = len(op.results)
    if result_count == 0:
        return_type = "None"
    elif result_count == 1:
        has_variadic_result = any(r.variadic for r in op.results)
        return_type = "list[ValueRef]" if has_variadic_result else "ValueRef"
    else:
        return_type = "list[ValueRef]"

    # Check if results are always from ResultTypeList (needs explicit results param)
    has_result_type_list = any(p["kind"] == "result_types" for p in params)
    has_func_args = any(p["kind"] == "func_args" for p in params)
    # If no ResultTypeList but op has results, add result_types param.
    if not has_result_type_list and result_count > 0:
        keyword_param_strs.append("result_types: list[Type]")

    param_strs = ["self"]
    if keyword_param_strs:
        param_strs.append("*")
        param_strs.extend(keyword_param_strs)

    lines: list[str] = []

    # Method signature.
    sig = ", ".join(param_strs)
    lines.append(f"    def {method_name}({sig}) -> {return_type}:")

    # Docstring.
    doc = op.doc or f"{op.name} operation."
    lines.append(f'        """{doc}')
    if op.examples:
        lines.append("")
        lines.append("        Example::")
        lines.extend(f"            {example_line}" for example_line in op.examples[0].split("\n"))
    lines.append('        """')

    # Body: collect operands, attributes, regions.
    lines.append("        _operands: list[ValueRef | int] = []")
    if has_func_args:
        lines.append("        _func_args: list[ValueRef | int] = []")
    lines.append("        _attributes: builtins.dict[str, Any] = {}")
    lines.append("        _regions: list[Region] = []")

    params_by_name = {param["name"]: param for param in params}
    operand_param_kinds = frozenset(
        {
            "operand",
            "operand_variadic",
            "index_list",
            "binding_list",
            "operand_dict",
        }
    )

    for param in params:
        if param["kind"] in operand_param_kinds:
            continue
        match param["kind"]:
            case "attr":
                if param.get("optional"):
                    lines.append(f"        if {param['name']} is not None:")
                    lines.append(f'            _attributes["{param["name"]}"] = {param["name"]}')
                else:
                    lines.append(f'        _attributes["{param["name"]}"] = {param["name"]}')
            case "binding_list" | "index_list" | "operand" | "operand_variadic":
                pass
            case "func_args":
                lines.append(f"        if {param['name']} is not None:")
                lines.append(f"            _func_args.extend({param['name']})")
            case "predicate_list":
                if param.get("optional"):
                    lines.append(f"        if {param['name']}:")
                    lines.append(f'            _attributes["{param["name"]}"] = {param["name"]}')
                else:
                    lines.append(f'        _attributes["{param["name"]}"] = {param["name"]}')
            case "flags":
                lines.append(f"        if {param['name']} is not None:")
                lines.append(f'            _attributes["{param["name"]}"] = {param["name"]}')
            case "result_types":
                pass  # Handled below.
            case "region":
                lines.append(f"        if {param['name']} is not None:")
                lines.append(f"            _regions.append({param['name']})")

    for operand in op.operands:
        operand_param = params_by_name.get(operand.name)
        if operand_param is None:
            continue
        match operand_param["kind"]:
            case "operand":
                lines.append(f"        _operands.append({operand_param['name']})")
            case "operand_variadic":
                lines.append(f"        _operands.extend({operand_param['name']})")
            case "index_list":
                lines.append("        _sentinel = -(2**63)")
                lines.append("        _static = []")
                lines.append(f"        for _idx in {operand_param['name']}:")
                lines.append("            if isinstance(_idx, ValueRef):")
                lines.append("                _static.append(_sentinel)")
                lines.append("                _operands.append(_idx)")
                lines.append("            else:")
                lines.append("                _static.append(_idx)")
                lines.append(f'        _attributes["{operand_param["static_field"]}"] = _static')
            case "binding_list":
                lines.append(f"        _operands.extend({operand_param['name']})")
            case "operand_dict":
                lines.append(f"        if {operand_param['name']}:")
                lines.append("            _operand_dict_names: builtins.dict[str, int] = {}")
                lines.append(f"            for _name in sorted({operand_param['name']}):")
                lines.append("                _operand_dict_names[_name] = len(_operand_dict_names)")
                lines.append(f"                _operands.append({operand_param['name']}[_name])")
                lines.append(f'            _attributes["{operand_param["names_field"]}"] = _operand_dict_names')

    # Build call and return with appropriate type narrowing.
    if has_result_type_list:
        if return_type == "None":
            lines.append(f'        self._b.build("{op.name}", _operands,')
            if has_func_args:
                lines.append("            func_args=_func_args, results=results,")
            else:
                lines.append("            results=results,")
            lines.append("            attributes=_attributes, regions=_regions)")
        else:
            lines.append(f'        return cast({return_type}, self._b.build("{op.name}", _operands,')
            if has_func_args:
                lines.append("            func_args=_func_args, results=results,")
            else:
                lines.append("            results=results,")
            lines.append("            attributes=_attributes, regions=_regions))")
    elif result_count > 0:
        lines.append(f'        return cast({return_type}, self._b.build("{op.name}", _operands,')
        if has_func_args:
            lines.append("            func_args=_func_args, results=result_types,")
        else:
            lines.append("            results=result_types,")
        lines.append("            attributes=_attributes, regions=_regions))")
    else:
        lines.append(f'        self._b.build("{op.name}", _operands,')
        if has_func_args:
            lines.append("            func_args=_func_args, attributes=_attributes,")
        else:
            lines.append("            attributes=_attributes,")
        lines.append("            regions=_regions)")

    return lines


def _has_typeof_result(op: Op) -> bool:
    """Check if the format has TypeOf referencing a result field."""
    layout = compute_layout(op)
    for element in op.format:
        if isinstance(element, TypeOf):
            desc = layout.fields.get(element.field)
            if desc and desc.kind == FieldKind.RESULT:
                return True
    return False


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """Generate builder stubs for all registered dialects."""
    from loom.dialect.buffer import ALL_BUFFER_OPS
    from loom.dialect.encoding import ALL_ENCODING_OPS
    from loom.dialect.func import ALL_FUNC_OPS
    from loom.dialect.index import ALL_INDEX_OPS
    from loom.dialect.kernel import ALL_KERNEL_OPS
    from loom.dialect.pool import ALL_POOL_OPS
    from loom.dialect.scalar import ALL_SCALAR_OPS
    from loom.dialect.test import ALL_TEST_OPS
    from loom.dialect.vector import ALL_VECTOR_OPS
    from loom.dialect.view import ALL_VIEW_OPS

    dialect_root = _bootstrap.REPO_ROOT / "loom" / "py" / "loom" / "dialect"

    dialects = [
        ("test", list(ALL_TEST_OPS), "TestBuilders", "test/builders.py"),
        ("scalar", list(ALL_SCALAR_OPS), "ScalarBuilders", "scalar/builders.py"),
        (
            "encoding",
            list(ALL_ENCODING_OPS),
            "EncodingBuilders",
            "encoding/builders.py",
        ),
        ("func", list(ALL_FUNC_OPS), "FuncBuilders", "func/builders.py"),
        ("pool", list(ALL_POOL_OPS), "PoolBuilders", "pool/builders.py"),
        ("buffer", list(ALL_BUFFER_OPS), "BufferBuilders", "buffer/builders.py"),
        ("view", list(ALL_VIEW_OPS), "ViewBuilders", "view/builders.py"),
        ("vector", list(ALL_VECTOR_OPS), "VectorBuilders", "vector/builders.py"),
        ("index", list(ALL_INDEX_OPS), "IndexBuilders", "index/builders.py"),
        ("kernel", list(ALL_KERNEL_OPS), "KernelBuilders", "kernel/builders.py"),
    ]

    for _dialect_name, ops, class_name, rel_path in dialects:
        code = generate_builders(ops, class_name)
        output_path = dialect_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  {rel_path} ({len(ops)} ops)")


if __name__ == "__main__":
    main()
