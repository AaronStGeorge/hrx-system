# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BYTECODE domain — format errors and version mismatches."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_BYTECODE_001: Invalid magic bytes.
ERR_BYTECODE_001 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=1,
    severity=Severity.ERROR,
    summary="Invalid magic bytes.",
    message="invalid magic bytes: expected {expected_magic}, got {actual_magic}",
    params=(
        ErrorParam("expected_magic", ParamKind.STRING),
        ErrorParam("actual_magic", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_002: Unsupported format version.
ERR_BYTECODE_002 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=2,
    severity=Severity.ERROR,
    summary="Unsupported format version.",
    message="unsupported format version {actual_version}, expected {expected_version}",
    params=(
        ErrorParam("actual_version", ParamKind.U32),
        ErrorParam("expected_version", ParamKind.U32),
    ),
)

# ERR_BYTECODE_003: Unexpected end of data.
ERR_BYTECODE_003 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=3,
    severity=Severity.ERROR,
    summary="Unexpected end of data.",
    message="unexpected end of data at offset {offset}, "
    "need {needed_bytes} bytes but only {available_bytes} remain",
    params=(
        ErrorParam("offset", ParamKind.U32),
        ErrorParam("needed_bytes", ParamKind.U32),
        ErrorParam("available_bytes", ParamKind.U32),
    ),
)

# ERR_BYTECODE_004: Unknown type kind.
ERR_BYTECODE_004 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=4,
    severity=Severity.ERROR,
    summary="Unknown type kind.",
    message="unknown type kind {type_kind} at offset {offset}",
    params=(
        ErrorParam("type_kind", ParamKind.U32),
        ErrorParam("offset", ParamKind.U32),
    ),
)

# ERR_BYTECODE_005: Unknown attribute kind.
ERR_BYTECODE_005 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=5,
    severity=Severity.ERROR,
    summary="Unknown attribute kind.",
    message="unknown attribute kind {attr_kind} at offset {offset}",
    params=(
        ErrorParam("attr_kind", ParamKind.U32),
        ErrorParam("offset", ParamKind.U32),
    ),
)

ALL_BYTECODE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_BYTECODE_001,
    ERR_BYTECODE_002,
    ERR_BYTECODE_003,
    ERR_BYTECODE_004,
    ERR_BYTECODE_005,
)
