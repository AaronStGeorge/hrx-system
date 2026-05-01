# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Contract table discriminants."""

from enum import Enum, unique


@unique
class ContractSystem(Enum):
    """Shared interpreter system used by a contract table case."""

    DESCRIPTOR_RULE = "descriptor_rule"
    VALUE_ALIAS = "value_alias"
    SOURCE_MEMORY = "source_memory"
    ENVIRONMENT = "environment"
    DESCRIPTOR_MATRIX = "descriptor_matrix"
    CUSTOM_FAMILY = "custom_family"


@unique
class SourceValueKind(Enum):
    """Source value namespace referenced by a contract row."""

    OPERAND = "operand"
    RESULT = "result"
    TEMPORARY = "temporary"
