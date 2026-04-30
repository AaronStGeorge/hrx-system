# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Minimal structural verification for Python-built Loom modules."""

from __future__ import annotations

from dataclasses import dataclass

from loom.importers.core.diagnostics import DiagnosticEngine
from loom.ir import Block, Module, Operation, Region


@dataclass(slots=True)
class StructuralVerifier:
    """Verifies IR graph references independent of per-op semantics."""

    module: Module
    diagnostics: DiagnosticEngine

    def verify(self) -> None:
        self._verify_symbol_names()
        for symbol_index, symbol in enumerate(self.module.symbols):
            if symbol.op is None:
                self.diagnostics.error(
                    "symbol has no defining operation",
                    source=f"symbol[{symbol_index}] @{symbol.name}",
                )
                continue
            self._verify_operation(symbol.op, f"symbol[{symbol_index}] @{symbol.name}")

    def _verify_symbol_names(self) -> None:
        seen: dict[str, int] = {}
        for symbol_index, symbol in enumerate(self.module.symbols):
            if not symbol.name:
                self.diagnostics.error(
                    "symbol has empty name", source=f"symbol[{symbol_index}]"
                )
                continue
            previous = seen.get(symbol.name)
            if previous is not None:
                self.diagnostics.error(
                    "duplicate symbol name",
                    source=f"symbol[{symbol_index}] @{symbol.name}",
                    details=(f"previous definition is symbol[{previous}]",),
                )
            else:
                seen[symbol.name] = symbol_index

    def _verify_operation(self, operation: Operation, path: str) -> None:
        self._verify_value_ids(operation.operands, f"{path} {operation.name} operands")
        self._verify_value_ids(operation.results, f"{path} {operation.name} results")
        for region_index, region in enumerate(operation.regions):
            self._verify_region(
                region, f"{path} {operation.name}.regions[{region_index}]"
            )

    def _verify_region(self, region: Region, path: str) -> None:
        for block_index, block in enumerate(region.blocks):
            self._verify_block(block, f"{path}.blocks[{block_index}]")

    def _verify_block(self, block: Block, path: str) -> None:
        self._verify_value_ids(block.arg_ids, f"{path} args")
        for operation_index, operation in enumerate(block.ops):
            self._verify_operation(operation, f"{path}.ops[{operation_index}]")

    def _verify_value_ids(self, value_ids: list[int], source: str) -> None:
        value_count = len(self.module.values)
        for value_id in value_ids:
            if value_id < 0 or value_id >= value_count:
                self.diagnostics.error(
                    "operation references missing value",
                    source=source,
                    details=(f"value id {value_id} is outside [0, {value_count})",),
                )
