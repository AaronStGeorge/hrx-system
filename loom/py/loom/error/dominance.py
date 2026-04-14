# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DOMINANCE domain — undefined values and use-after-consume."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_DOMINANCE_001: Use of undefined value.
ERR_DOMINANCE_001 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=1,
    severity=Severity.ERROR,
    summary="Use of undefined value.",
    message="use of undefined value '{value_name}'",
    params=(ErrorParam("value_name", ParamKind.STRING),),
)

# ERR_DOMINANCE_002: Use of consumed value (after tied result).
ERR_DOMINANCE_002 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=2,
    severity=Severity.ERROR,
    summary="Use of consumed value.",
    message="use of consumed value '{value_name}' (consumed by tied "
    "result of '{consuming_op}')",
    params=(
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("consuming_op", ParamKind.STRING),
    ),
    fix_hint="Use the tied result of '{consuming_op}' instead of '{value_name}'",
)

# ERR_DOMINANCE_003: Value ID out of range.
ERR_DOMINANCE_003 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=3,
    severity=Severity.ERROR,
    summary="Value ID out of range.",
    message="value ID {value_id} is out of range (max {max_id})",
    params=(
        ErrorParam("value_id", ParamKind.U32),
        ErrorParam("max_id", ParamKind.U32),
    ),
)

# ERR_DOMINANCE_004: Tied result index out of range.
ERR_DOMINANCE_004 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=4,
    severity=Severity.ERROR,
    summary="Tied result index out of range.",
    message="tied result index {result_index} is out of range for "
    "'{op_name}' which has {result_count} results",
    params=(
        ErrorParam("result_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("result_count", ParamKind.U32),
    ),
)

# ERR_DOMINANCE_005: Tied operand index out of range.
ERR_DOMINANCE_005 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=5,
    severity=Severity.ERROR,
    summary="Tied operand index out of range.",
    message="tied operand index {operand_index} is out of range for "
    "'{op_name}' which has {operand_count} operands",
    params=(
        ErrorParam("operand_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("operand_count", ParamKind.U32),
    ),
)

# ERR_DOMINANCE_006: Duplicate tied result index.
ERR_DOMINANCE_006 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=6,
    severity=Severity.ERROR,
    summary="Duplicate tied result index.",
    message="result {result_index} of '{op_name}' is tied more than once",
    params=(
        ErrorParam("result_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_DOMINANCE_007: Duplicate tied operand index.
ERR_DOMINANCE_007 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=7,
    severity=Severity.ERROR,
    summary="Duplicate tied operand index.",
    message="operand {operand_index} of '{op_name}' is tied by more than one result",
    params=(
        ErrorParam("operand_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_DOMINANCE_008: Ambiguous tied operand value.
ERR_DOMINANCE_008 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=8,
    severity=Severity.ERROR,
    summary="Ambiguous tied operand value.",
    message="operand {operand_index} of '{op_name}' ties value '%{value_name}', "
    "but that value appears in multiple operand slots",
    params=(
        ErrorParam("operand_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
    ),
)

# ERR_DOMINANCE_009: Value use count violates a structural lifetime contract.
ERR_DOMINANCE_009 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=9,
    severity=Severity.ERROR,
    summary="Value use count violates a structural lifetime contract.",
    message="value '%{value_name}' has {actual_count} uses, expected "
    "{expected_constraint}",
    params=(
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="Route '%{value_name}' through the required consumer shape.",
)

# ERR_DOMINANCE_010: Value user violates a structural lifetime contract.
ERR_DOMINANCE_010 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=10,
    severity=Severity.ERROR,
    summary="Value user violates a structural lifetime contract.",
    message="value '%{value_name}' is used by '{user_op}', expected "
    "{expected_constraint}",
    params=(
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("user_op", ParamKind.STRING),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="Use '%{value_name}' only where the lifetime contract allows it.",
)

ALL_DOMINANCE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_DOMINANCE_001,
    ERR_DOMINANCE_002,
    ERR_DOMINANCE_003,
    ERR_DOMINANCE_004,
    ERR_DOMINANCE_005,
    ERR_DOMINANCE_006,
    ERR_DOMINANCE_007,
    ERR_DOMINANCE_008,
    ERR_DOMINANCE_009,
    ERR_DOMINANCE_010,
)
