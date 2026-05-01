# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python source schema for target contract tables."""

from loom.target.contracts.compile import (
    CONTRACT_ROW_NONE,
    CompiledCase,
    CompiledContractTable,
    CompiledDescriptorRule,
    CompiledDialectTable,
    CompiledOpEntry,
    compile_contract_table,
)
from loom.target.contracts.descriptors import (
    descriptor_by_key,
    descriptor_by_semantic_tag,
)
from loom.target.contracts.emits import EmitDescriptorOp
from loom.target.contracts.guards import Guard, GuardKind
from loom.target.contracts.immediates import AttrProject, AttrProjectKind
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.patterns import Scalar, TypePattern, Vector
from loom.target.contracts.rules import ContractCase, DescriptorRule, ValueAliasRule
from loom.target.contracts.source import ValueRef
from loom.target.contracts.tables import ContractTable

__all__ = [
    "AttrProject",
    "AttrProjectKind",
    "CONTRACT_ROW_NONE",
    "CompiledCase",
    "CompiledContractTable",
    "CompiledDescriptorRule",
    "CompiledDialectTable",
    "CompiledOpEntry",
    "ContractCase",
    "ContractSystem",
    "ContractTable",
    "DescriptorRule",
    "EmitDescriptorOp",
    "Guard",
    "GuardKind",
    "Scalar",
    "SourceValueKind",
    "TypePattern",
    "ValueAliasRule",
    "ValueRef",
    "Vector",
    "compile_contract_table",
    "descriptor_by_key",
    "descriptor_by_semantic_tag",
]
