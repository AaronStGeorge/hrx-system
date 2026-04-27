# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Python builder stub generation."""

from loom.assembly import (
    AttrDict,
    AttrTable,
    BlockRef,
    Clause,
    IndexList,
    OperandDict,
    OptionalGroup,
    Ref,
    ResultType,
    ResultTypeList,
    SymbolRef,
)
from loom.dsl import ANY, ATTR_TYPE_I64_ARRAY, AttrDef, EnumCase, EnumDef, Op, Operand, Result, Successor
from loom.gen.builders import generate_builders


def test_builder_methods_disambiguate_dotted_short_name_collisions() -> None:
    generated = generate_builders(
        [
            Op("test.load.mask", operands=[Operand("value", ANY)], format=[Ref("value")]),
            Op("test.store.mask", operands=[Operand("value", ANY)], format=[Ref("value")]),
        ],
        "TestBuilders",
    )

    assert "def load_mask(" in generated
    assert "def store_mask(" in generated
    assert "def mask(" not in generated


def test_builder_operands_are_packed_in_declared_storage_order() -> None:
    generated = generate_builders(
        [
            Op(
                "test.load.mask",
                operands=[
                    Operand("view", ANY),
                    Operand("mask", ANY),
                    Operand("passthrough", ANY),
                    Operand("indices", ANY, variadic=True),
                ],
                attrs=[AttrDef("static_indices", ATTR_TYPE_I64_ARRAY)],
                format=[
                    Ref("view"),
                    IndexList("indices", "static_indices"),
                    Ref("mask"),
                    Ref("passthrough"),
                ],
            ),
        ],
        "TestBuilders",
    )

    assert ("def mask(self, *, view: ValueRef, indices: list[int | ValueRef], mask: ValueRef, passthrough: ValueRef) -> None:") in generated
    assert generated.index("_operands.append(view)") < generated.index("_operands.append(mask)")
    assert generated.index("_operands.append(mask)") < generated.index("_operands.append(passthrough)")
    assert generated.index("_operands.append(passthrough)") < generated.index("for _idx in indices")


def test_builder_params_include_attrs_from_inline_attr_dict() -> None:
    ordering = EnumDef("Ordering", [EnumCase("relaxed", 0)])
    scope = EnumDef("Scope", [EnumCase("workgroup", 0)])
    generated = generate_builders(
        [
            Op(
                "test.atomic",
                operands=[Operand("value", ANY)],
                attrs=[
                    AttrDef("ordering", "enum", enum_def=ordering),
                    AttrDef("scope", "enum", enum_def=scope),
                ],
                format=[
                    Ref("value"),
                    AttrDict(),
                ],
            ),
        ],
        "TestBuilders",
    )

    assert "def atomic(self, *, value: ValueRef, ordering: str, scope: str) -> None:" in generated
    assert '_attributes["ordering"] = ordering' in generated
    assert '_attributes["scope"] = scope' in generated


def test_builder_params_preserve_optional_symbol_refs() -> None:
    generated = generate_builders(
        [
            Op(
                "test.callable",
                attrs=[
                    AttrDef("target", "symbol", optional=True),
                    AttrDef("callee", "symbol"),
                ],
                format=[
                    OptionalGroup([SymbolRef("target")], anchor="target"),
                    SymbolRef("callee"),
                ],
            ),
        ],
        "TestBuilders",
    )

    assert "def callable(self, *, target: str | None = None, callee: str) -> None:" in generated
    assert "if target is not None:" in generated
    assert '_attributes["target"] = target' in generated
    assert '_attributes["callee"] = callee' in generated


def test_builder_params_descend_into_clauses() -> None:
    generated = generate_builders(
        [
            Op(
                "test.copy",
                operands=[
                    Operand("source", ANY),
                    Operand("target", ANY),
                ],
                format=[
                    Clause("source", Ref("source")),
                    Clause("target", Ref("target")),
                ],
            ),
        ],
        "TestBuilders",
    )

    assert "def copy(self, *, source: ValueRef, target: ValueRef) -> None:" in generated
    assert "_operands.append(source)" in generated
    assert "_operands.append(target)" in generated


def test_builder_params_include_operand_dict_as_keyed_values() -> None:
    generated = generate_builders(
        [
            Op(
                "test.operand_dict",
                operands=[
                    Operand("input", ANY),
                    Operand("params", ANY, variadic=True),
                ],
                results=[Result("result", ANY)],
                attrs=[AttrDef("param_names", "dict", optional=True)],
                format=[
                    Ref("input"),
                    OperandDict("params", "param_names"),
                    ResultType("result"),
                ],
            ),
        ],
        "TestBuilders",
    )

    assert "def operand_dict(self, *, input: ValueRef" in generated
    assert "params: dict[str, ValueRef]" in generated
    assert "results:" in generated
    assert "for _name in sorted(params):" in generated
    assert "_operand_dict_names[_name] = len(_operand_dict_names)" in generated
    assert "_operands.append(params[_name])" in generated
    assert '_attributes["param_names"] = _operand_dict_names' in generated


def test_builder_params_include_attr_table_as_flattened_values() -> None:
    generated = generate_builders(
        [
            Op(
                "test.attr_table",
                operands=[
                    Operand("selector", ANY),
                    Operand("values", ANY, variadic=True),
                ],
                results=[Result("results", ANY, variadic=True)],
                attrs=[AttrDef("case_keys", ATTR_TYPE_I64_ARRAY)],
                format=[
                    Ref("selector"),
                    AttrTable("case_keys", "values"),
                    ResultTypeList("results", parens=False),
                ],
            ),
        ],
        "TestBuilders",
    )

    assert "def attr_table(self, *, selector: ValueRef" in generated
    assert "case_keys: list[int]" in generated
    assert "values: list[ValueRef]" in generated
    assert "results:" in generated
    assert "_operands.append(selector)" in generated
    assert "_operands.extend(values)" in generated
    assert '_attributes["case_keys"] = case_keys' in generated


def test_builder_params_include_successors() -> None:
    generated = generate_builders(
        [
            Op(
                "test.br",
                successors=[Successor("dest")],
                format=[BlockRef("dest")],
            ),
        ],
        "TestBuilders",
    )

    assert "from loom.ir import Block" in generated
    assert "def br(self, *, dest: Block) -> None:" in generated
    assert "_successors: list[Block] = []" in generated
    assert "_successors.append(dest)" in generated
    assert "successors=_successors" in generated
