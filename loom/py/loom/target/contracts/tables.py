# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract table records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.target.contracts.descriptors import _validate_descriptor_set_keys
from loom.target.contracts.rules import ContractCase
from loom.target.low_descriptors import DescriptorSet


@dataclass(frozen=True, slots=True)
class ContractTable:
    """Target contract table authored and validated in Python."""

    name: str
    descriptor_set: DescriptorSet
    cases: tuple[ContractCase, ...] = ()
    table_index: int = 0
    c_header_path: Path | None = None
    c_source_path: Path | None = None
    header_guard: str = ""
    public_header: str = ""
    symbol_name: str = ""
    c_table_prefix: str = ""

    def __init__(
        self,
        *,
        name: str,
        descriptor_set: DescriptorSet,
        cases: Sequence[ContractCase] = (),
        table_index: int = 0,
        c_header_path: Path | None = None,
        c_source_path: Path | None = None,
        header_guard: str = "",
        public_header: str = "",
        symbol_name: str = "",
        c_table_prefix: str = "",
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "descriptor_set", descriptor_set)
        object.__setattr__(self, "cases", tuple(cases))
        object.__setattr__(self, "table_index", table_index)
        object.__setattr__(self, "c_header_path", c_header_path)
        object.__setattr__(self, "c_source_path", c_source_path)
        object.__setattr__(self, "header_guard", header_guard)
        object.__setattr__(self, "public_header", public_header)
        object.__setattr__(self, "symbol_name", symbol_name)
        object.__setattr__(self, "c_table_prefix", c_table_prefix)
        if not name:
            raise ValueError("contract table name must be non-empty")
        if table_index < 0 or table_index > 0xFFFF:
            raise ValueError("contract table index must fit in uint16_t")
        _validate_descriptor_set_keys(descriptor_set)
        for case in self.cases:
            case.validate(descriptor_set)
