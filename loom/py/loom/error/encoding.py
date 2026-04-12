# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ENCODING domain — encoding mismatches."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_ENCODING_001: SameEncoding constraint violated.
ERR_ENCODING_001 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=1,
    severity=Severity.ERROR,
    summary="Operands must have the same encoding.",
    message="encoding mismatch: '{field_a}' and '{field_b}' have different encodings",
    params=(
        ErrorParam("field_a", ParamKind.STRING),
        ErrorParam("field_b", ParamKind.STRING),
    ),
    fix_hint="Operands '{field_a}' and '{field_b}' must have the same "
    "encoding; use explicit encoding casts if needed",
)

# ERR_ENCODING_002: Encoding must be preserved across slice/view ops.
ERR_ENCODING_002 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=2,
    severity=Severity.ERROR,
    summary="Encoding must be preserved across slice/view operations.",
    message="result encoding does not match source encoding for '{op_name}'",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Slice and view operations must preserve the source encoding",
)

# ERR_ENCODING_003: SSA encoding value_id out of range.
ERR_ENCODING_003 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=3,
    severity=Severity.ERROR,
    summary="SSA encoding reference out of range.",
    message="{field_name} type references encoding value %{value_id}, "
    "but module has only {value_count} values",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U32),
        ErrorParam("value_count", ParamKind.U32),
    ),
    fix_hint="The encoding value_id in the type is invalid; this "
    "indicates a corrupted or malformed module",
)

# ERR_ENCODING_004: SSA encoding value not defined at point of use.
ERR_ENCODING_004 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=4,
    severity=Severity.ERROR,
    summary="SSA encoding value not visible at point of use.",
    message="{field_name} type references encoding value '%{value_name}' "
    "which is not defined at this point",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
    ),
    fix_hint="The encoding value must be defined before it can be "
    "referenced in a type; ensure the encoding.define op dominates this use",
)

# ERR_ENCODING_005: SSA encoding value has wrong type.
ERR_ENCODING_005 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=5,
    severity=Severity.ERROR,
    summary="SSA encoding reference has wrong type.",
    message="{field_name} type references encoding value '%{value_name}' "
    "which has type {actual_type}, expected 'encoding'",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
    fix_hint="The encoding value must have type 'encoding'; "
    "it should be the result of an encoding.define op",
)

# ERR_ENCODING_006: encoding.define parameter is both static and dynamic.
ERR_ENCODING_006 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=6,
    severity=Severity.ERROR,
    summary="Encoding parameter is both static and dynamic.",
    message="encoding.define parameter '{param_name}' is both static and dynamic",
    params=(ErrorParam("param_name", ParamKind.STRING),),
    fix_hint=(
        "Keep compile-time values in #family<...> and SSA values in the "
        "encoding.define operand dictionary; each parameter name must appear "
        "in exactly one place"
    ),
)

ALL_ENCODING_ERRORS: tuple[ErrorDef, ...] = (
    ERR_ENCODING_001,
    ERR_ENCODING_002,
    ERR_ENCODING_003,
    ERR_ENCODING_004,
    ERR_ENCODING_005,
    ERR_ENCODING_006,
)
