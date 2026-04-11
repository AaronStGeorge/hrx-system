# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Python builder stub generation."""

from loom.assembly import AttrDict, IndexList, Ref
from loom.dsl import ANY, ATTR_TYPE_I64_ARRAY, AttrDef, EnumCase, EnumDef, Op, Operand
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
