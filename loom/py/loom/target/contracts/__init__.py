# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python source schema for target contract fragments."""

from loom.target.contracts.compile import (
    CONTRACT_ROW_NONE,
    CompiledCase,
    CompiledContractFragment,
    CompiledDescriptorRule,
    CompiledDialectTable,
    CompiledOpEntry,
    compile_contract_fragment,
)
from loom.target.contracts.descriptors import (
    descriptor_by_key,
    descriptor_by_semantic_tag,
)
from loom.target.contracts.emits import DescriptorEmitForm, EmitDescriptorOp
from loom.target.contracts.fragments import ContractFragment
from loom.target.contracts.guards import Guard, GuardDiagnostic, GuardKind
from loom.target.contracts.immediates import (
    AttrProject,
    AttrProjectKind,
    ValueProject,
    ValueProjectKind,
)
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.lower_rules import (
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN,
    LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,
    CompiledLowerRuleSet,
    LowerAttrCopy,
    LowerAttrCopyKind,
    LowerDiagnostic,
    LowerEmit,
    LowerEmitKind,
    LowerGuard,
    LowerRule,
    LowerRuleSpan,
    LowerTiedResult,
    LowerTypePattern,
    LowerValueRef,
    compile_lower_rule_set,
)
from loom.target.contracts.materializers import ValueMaterializer
from loom.target.contracts.patterns import Scalar, TypePattern, Vector
from loom.target.contracts.rules import (
    ContractCase,
    DescriptorRule,
    ValueAliasRule,
    ValueElideRule,
)
from loom.target.contracts.source import ValueRef
from loom.target.contracts.templates import (
    DirectDescriptorCase,
    DirectTypePatterns,
    PredicateDescriptorCase,
    ReductionDescriptorCase,
    SelectDescriptorCase,
    binary_descriptor_rules,
    compare_descriptor_rules,
    reduction_descriptor_rules,
    select_descriptor_rules,
    ternary_descriptor_rules,
    unary_descriptor_rules,
)

__all__ = [
    "AttrProject",
    "AttrProjectKind",
    "CONTRACT_ROW_NONE",
    "CompiledCase",
    "CompiledContractFragment",
    "CompiledDescriptorRule",
    "CompiledDialectTable",
    "CompiledLowerRuleSet",
    "CompiledOpEntry",
    "ContractCase",
    "ContractSystem",
    "ContractFragment",
    "DescriptorEmitForm",
    "DescriptorRule",
    "DirectDescriptorCase",
    "DirectTypePatterns",
    "EmitDescriptorOp",
    "Guard",
    "GuardDiagnostic",
    "GuardKind",
    "LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS",
    "LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN",
    "LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1",
    "LowerAttrCopy",
    "LowerAttrCopyKind",
    "LowerDiagnostic",
    "LowerEmit",
    "LowerEmitKind",
    "LowerGuard",
    "LowerRule",
    "LowerRuleSpan",
    "LowerTiedResult",
    "LowerTypePattern",
    "LowerValueRef",
    "PredicateDescriptorCase",
    "ReductionDescriptorCase",
    "Scalar",
    "SelectDescriptorCase",
    "SourceValueKind",
    "TypePattern",
    "ValueAliasRule",
    "ValueElideRule",
    "ValueMaterializer",
    "ValueProject",
    "ValueProjectKind",
    "ValueRef",
    "Vector",
    "binary_descriptor_rules",
    "compile_contract_fragment",
    "compile_lower_rule_set",
    "compare_descriptor_rules",
    "descriptor_by_key",
    "descriptor_by_semantic_tag",
    "reduction_descriptor_rules",
    "select_descriptor_rules",
    "ternary_descriptor_rules",
    "unary_descriptor_rules",
]
