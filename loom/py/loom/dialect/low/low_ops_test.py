# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for target-low op text behavior."""

from collections.abc import Mapping
from typing import Any

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.low import ALL_LOW_OPS
from loom.format.text.parser import NameScope, Parser
from loom.format.text.printer import Printer
from loom.ir import Module, RegisterType, StorageSpace, StorageType, Type, Value


def _parser() -> Parser:
    parser = Parser()
    parser.register_ops(ALL_LOW_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def _printer() -> Printer:
    printer = Printer()
    printer.register_ops(ALL_LOW_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer


def _scope_with_values(
    *names_and_types: tuple[str, Type],
) -> tuple[Module, NameScope]:
    module = Module()
    scope = NameScope()
    for name, value_type in names_and_types:
        value_id = module.add_value(Value(name=name, type=value_type))
        scope.define(name, value_id)
    return module, scope


def _parse_and_print(text: str, module: Module, scope: NameScope) -> tuple[Mapping[str, Any], str]:
    op = _parser().parse_operation_from_text(text, module=module, scope=scope)
    return op.attributes, _printer().print_operation(op, module)


def test_storage_address_omitted_default_offset_is_restored() -> None:
    source = " ".join(
        [
            "%addr = low.storage.address %slot : low.storage<workgroup>",
            "-> reg<amdgpu.vgpr>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.WORKGROUP)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == source


def test_storage_address_explicit_default_offset_prints_omitted_form() -> None:
    source = " ".join(
        [
            "%addr = low.storage.address %slot {offset = 0} :",
            "low.storage<workgroup> -> reg<amdgpu.vgpr>",
        ]
    )
    expected = " ".join(
        [
            "%addr = low.storage.address %slot : low.storage<workgroup>",
            "-> reg<amdgpu.vgpr>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.WORKGROUP)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == expected


def test_storage_address_non_default_offset_is_preserved() -> None:
    source = " ".join(
        [
            "%addr = low.storage.address %slot {offset = 16} :",
            "low.storage<workgroup> -> reg<amdgpu.vgpr>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.WORKGROUP)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 16
    assert text == source


def test_spill_explicit_default_offset_prints_omitted_form() -> None:
    source = " ".join(
        [
            "low.spill %value, %slot {offset = 0} :",
            "reg<amdgpu.vgpr x4>, low.storage<private>",
        ]
    )
    expected = " ".join(
        [
            "low.spill %value, %slot :",
            "reg<amdgpu.vgpr x4>, low.storage<private>",
        ]
    )
    module, scope = _scope_with_values(
        ("value", RegisterType("amdgpu.vgpr", 4)),
        ("slot", StorageType(StorageSpace.PRIVATE)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == expected


def test_reload_explicit_default_offset_prints_omitted_form() -> None:
    source = " ".join(
        [
            "%reload = low.reload %slot {offset = 0} :",
            "low.storage<private> -> reg<amdgpu.vgpr x4>",
        ]
    )
    expected = " ".join(
        [
            "%reload = low.reload %slot :",
            "low.storage<private> -> reg<amdgpu.vgpr x4>",
        ]
    )
    module, scope = _scope_with_values(
        ("slot", StorageType(StorageSpace.PRIVATE)),
    )
    attrs, text = _parse_and_print(source, module, scope)
    assert attrs["offset"] == 0
    assert text == expected
