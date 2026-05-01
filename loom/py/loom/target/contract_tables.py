# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Registry for Python-authored target contract tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import cast

from loom.dsl import Op
from loom.target.contracts import ContractTable


@dataclass(frozen=True)
class ContractTableRegistration:
    key: str
    module_name: str
    symbol_name: str
    dialect_ops_symbol_name: str
    aliases: tuple[str, ...] = ()

    def load(self) -> ContractTable:
        module = import_module(self.module_name)
        return cast(ContractTable, getattr(module, self.symbol_name))

    def load_dialect_ops(self) -> Mapping[str, Sequence[Op]]:
        module = import_module(self.module_name)
        return cast(
            Mapping[str, Sequence[Op]],
            getattr(module, self.dialect_ops_symbol_name),
        )


CONTRACT_TABLE_REGISTRATIONS = (
    ContractTableRegistration(
        key="amdgpu.async",
        module_name="loom.target.arch.amdgpu.contracts.async",
        symbol_name="AMDGPU_ASYNC_CONTRACT_TABLE",
        dialect_ops_symbol_name="AMDGPU_ASYNC_CONTRACT_DIALECT_OPS",
        aliases=("amdgpu_async",),
    ),
    ContractTableRegistration(
        key="amdgpu.arithmetic",
        module_name="loom.target.arch.amdgpu.contracts.arithmetic",
        symbol_name="AMDGPU_ARITHMETIC_CONTRACT_TABLE",
        dialect_ops_symbol_name="AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS",
        aliases=("amdgpu_arithmetic",),
    ),
    ContractTableRegistration(
        key="amdgpu.compare",
        module_name="loom.target.arch.amdgpu.contracts.compare",
        symbol_name="AMDGPU_COMPARE_CONTRACT_TABLE",
        dialect_ops_symbol_name="AMDGPU_COMPARE_CONTRACT_DIALECT_OPS",
        aliases=("amdgpu_compare",),
    ),
    ContractTableRegistration(
        key="amdgpu.dot",
        module_name="loom.target.arch.amdgpu.contracts.dot",
        symbol_name="AMDGPU_DOT_CONTRACT_TABLE",
        dialect_ops_symbol_name="AMDGPU_DOT_CONTRACT_DIALECT_OPS",
        aliases=("amdgpu_dot",),
    ),
    ContractTableRegistration(
        key="amdgpu.integer",
        module_name="loom.target.arch.amdgpu.contracts.integer",
        symbol_name="AMDGPU_INTEGER_CONTRACT_TABLE",
        dialect_ops_symbol_name="AMDGPU_INTEGER_CONTRACT_DIALECT_OPS",
        aliases=("amdgpu_integer",),
    ),
    ContractTableRegistration(
        key="amdgpu.reduce",
        module_name="loom.target.arch.amdgpu.contracts.reduce",
        symbol_name="AMDGPU_REDUCE_CONTRACT_TABLE",
        dialect_ops_symbol_name="AMDGPU_REDUCE_CONTRACT_DIALECT_OPS",
        aliases=("amdgpu_reduce",),
    ),
    ContractTableRegistration(
        key="iree.vm.core",
        module_name="loom.target.emit.ireevm.contracts",
        symbol_name="IREEVM_CORE_CONTRACT_TABLE",
        dialect_ops_symbol_name="IREEVM_CORE_CONTRACT_DIALECT_OPS",
        aliases=("ireevm_core",),
    ),
    ContractTableRegistration(
        key="wasm.core.simd128",
        module_name="loom.target.emit.wasm.contracts",
        symbol_name="WASM_CORE_SIMD128_CONTRACT_TABLE",
        dialect_ops_symbol_name="WASM_CORE_SIMD128_CONTRACT_DIALECT_OPS",
        aliases=("wasm_core_simd128",),
    ),
    ContractTableRegistration(
        key="x86.avx512",
        module_name="loom.target.arch.x86.contracts.avx512",
        symbol_name="X86_AVX512_CONTRACT_TABLE",
        dialect_ops_symbol_name="X86_AVX512_CONTRACT_DIALECT_OPS",
        aliases=("x86_avx512",),
    ),
    ContractTableRegistration(
        key="test.low.core",
        module_name="loom.target.test.contracts",
        symbol_name="TEST_LOW_CORE_CONTRACT_TABLE",
        dialect_ops_symbol_name="TEST_LOW_CORE_CONTRACT_DIALECT_OPS",
        aliases=("test_low_core",),
    ),
)


def iter_contract_table_registrations() -> Iterable[ContractTableRegistration]:
    return CONTRACT_TABLE_REGISTRATIONS


def contract_table_names() -> tuple[str, ...]:
    names: list[str] = []
    for registration in CONTRACT_TABLE_REGISTRATIONS:
        names.append(registration.key)
        names.extend(registration.aliases)
    return tuple(names)


def resolve_contract_table(name: str) -> ContractTableRegistration:
    for registration in CONTRACT_TABLE_REGISTRATIONS:
        if name == registration.key or name in registration.aliases:
            return registration
    supported = ", ".join(contract_table_names())
    raise ValueError(
        f"unknown target contract table '{name}'; expected one of: {supported}"
    )
