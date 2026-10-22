# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declaration contracts for the C type registry."""

import pytest

from loom.assembly import COMMA, EncodingOf, OptionalGroup, ScalarOf, ShapeOf, kw
from loom.builtin_types import channel_type, read_type, view_type, write_type
from loom.dsl import EncodingParam, ScalarParam, ShapeParam, TypeDef
from loom.gen.ops.type_registry import generate_type_registry


def test_view_alignment_is_part_of_the_declared_format() -> None:
    _, _, source = generate_type_registry([view_type])
    assert "{LOOM_TYPE_FMT_ALIGNMENT, 3, 0}" in source
    assert "LOOM_KW_ALIGN" in source


@pytest.mark.parametrize("type_def", [channel_type, read_type, write_type])
def test_parameterized_channel_types_retain_their_fact_domain(type_def: TypeDef) -> None:
    _, _, source = generate_type_registry([type_def])
    descriptor = source.split(f"const loom_parameterized_type_descriptor_t loom_{type_def.name}_type_parameterized_descriptor = {{", 1)[1].split("};", 1)[0]
    assert ".fact_domain = &loom_view_fact_domain," in descriptor


def test_indirect_family_keys_follow_declaration_order_not_name_order() -> None:
    header, _, source = generate_type_registry([view_type, read_type, channel_type, write_type])
    for index, name in enumerate(("read", "channel", "write")):
        key = f"LOOM_PARAMETERIZED_TYPE_{name.upper()}"
        assert f"{key} = LOOM_PARAMETERIZED_TYPE_KIND(LOOM_PARAMETERIZED_TYPE_COMMON_NAMESPACE, {index})" in header
        assert f"loom_type_parameterized_kind(type) == {key}" in header
        assert f".kind = {key}," in source
        assert f"loom_module_make_parameterized_type_by_kind(\n      module, {key}," in source
    registry = source.split("loom_type_registry_parameterized_descriptors[] = {", 1)[1].split("};", 1)[0]
    assert registry.split() == [
        "&loom_read_type_parameterized_descriptor,",
        "&loom_channel_type_parameterized_descriptor,",
        "&loom_write_type_parameterized_descriptor,",
    ]
    assert "LOOM_PARAMETERIZED_TYPE_VIEW" not in header


def _compact_tensor_type_def(name: str) -> TypeDef:
    return TypeDef(
        name=name,
        ir_kind="tensor",
        params=[
            ShapeParam("dims"),
            ScalarParam("element_type"),
            EncodingParam("encoding"),
        ],
        format=[
            ShapeOf("dims"),
            kw("x"),
            ScalarOf("element_type"),
            OptionalGroup([COMMA, EncodingOf("encoding")], anchor="encoding"),
        ],
    )


def test_generate_type_registry_rejects_duplicate_builtin_type_names() -> None:
    type_defs = [
        _compact_tensor_type_def("test.tensor"),
        _compact_tensor_type_def("test.other_tensor"),
    ]

    with pytest.raises(ValueError, match=r"Type kind 'tensor' has duplicate registry names"):
        generate_type_registry(type_defs)


def test_generate_type_registry_rejects_invalid_fact_domain_symbol() -> None:
    type_def = TypeDef(
        name="test.handle",
        fact_domain="loom.test.handle.fact_domain",
    )

    with pytest.raises(ValueError, match=r"TypeDef 'test\.handle': fact_domain must be a C symbol name"):
        generate_type_registry([type_def])
