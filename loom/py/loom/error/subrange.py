# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SUBRANGE domain — offset/size counts and bounds violations."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_SUBRANGE_001: OffsetCountMatchesRank violated.
ERR_SUBRANGE_001 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=1,
    severity=Severity.ERROR,
    summary="Offset count must match operand rank.",
    message="'{operand_name}' offset count ({offset_count}) does not "
    "match rank ({rank})",
    params=(
        ErrorParam("operand_name", ParamKind.STRING),
        ErrorParam("offset_count", ParamKind.U32),
        ErrorParam("rank", ParamKind.I64),
    ),
    fix_hint="Provide exactly {rank} offsets for '{operand_name}'",
)

# ERR_SUBRANGE_002: DimIndexInBounds violated.
ERR_SUBRANGE_002 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=2,
    severity=Severity.ERROR,
    summary="Dimension index out of bounds.",
    message="dimension index {dim_index} is out of bounds for type with rank {rank}",
    params=(
        ErrorParam("dim_index", ParamKind.I64),
        ErrorParam("rank", ParamKind.I64),
    ),
    fix_hint="Use an index in range [0, {rank})",
)

# ERR_SUBRANGE_003: Size count does not match rank.
ERR_SUBRANGE_003 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=3,
    severity=Severity.ERROR,
    summary="Size count must match operand rank.",
    message="'{operand_name}' size count ({size_count}) does not match rank ({rank})",
    params=(
        ErrorParam("operand_name", ParamKind.STRING),
        ErrorParam("size_count", ParamKind.U32),
        ErrorParam("rank", ParamKind.I64),
    ),
    fix_hint="Provide exactly {rank} sizes for '{operand_name}'",
)

# ERR_SUBRANGE_004: Static slice out of bounds.
ERR_SUBRANGE_004 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=4,
    severity=Severity.ERROR,
    summary="Static slice extends beyond bounds.",
    message="dimension {dim_index}: offset ({offset}) + size ({size}) "
    "= {total} exceeds bound {bound}",
    params=(
        ErrorParam("dim_index", ParamKind.I64),
        ErrorParam("offset", ParamKind.I64),
        ErrorParam("size", ParamKind.I64),
        ErrorParam("total", ParamKind.I64),
        ErrorParam("bound", ParamKind.I64),
    ),
    fix_hint="Ensure offset + size <= {bound} for dimension {dim_index}",
)

# ERR_SUBRANGE_005: Vector memory footprint proof failed.
ERR_SUBRANGE_005 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=5,
    severity=Severity.ERROR,
    summary="Vector memory footprint is not proven in bounds.",
    message="{op_name} footprint proof failed on view axis {view_axis} "
    "(vector axis {vector_axis}): {reason}; required {required}; "
    "origin {origin}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("view_axis", ParamKind.I64),
        ErrorParam("vector_axis", ParamKind.I64),
        ErrorParam("origin", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
        ErrorParam("required", ParamKind.STRING),
    ),
    fix_hint="Refine the origin, vector extent, mask, or view layout facts so "
    "{required} is provable",
)

# ERR_SUBRANGE_006: Vector memory linear footprint proof failed.
ERR_SUBRANGE_006 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=6,
    severity=Severity.ERROR,
    summary="Vector memory linear footprint is not proven in bounds.",
    message="{op_name} linear footprint proof failed: {reason}; "
    "required {required}; origin {origin}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("origin", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
        ErrorParam("required", ParamKind.STRING),
    ),
    fix_hint="Refine the origin, offset lane range, or view storage span facts "
    "so {required} is provable",
)

ALL_SUBRANGE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_SUBRANGE_001,
    ERR_SUBRANGE_002,
    ERR_SUBRANGE_003,
    ERR_SUBRANGE_004,
    ERR_SUBRANGE_005,
    ERR_SUBRANGE_006,
)
