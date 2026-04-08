# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""STRUCTURE domain — count errors, terminators, and regions."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_STRUCTURE_001: Wrong operand count.
ERR_STRUCTURE_001 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=1,
    severity=Severity.ERROR,
    summary="Wrong operand count.",
    message="'{op_name}' has {actual_count} operands, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_002: Wrong result count.
ERR_STRUCTURE_002 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=2,
    severity=Severity.ERROR,
    summary="Wrong result count.",
    message="'{op_name}' has {actual_count} results, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_003: Wrong attribute count.
ERR_STRUCTURE_003 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=3,
    severity=Severity.ERROR,
    summary="Wrong attribute count.",
    message="'{op_name}' has {actual_count} attributes, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_004: Wrong region count.
ERR_STRUCTURE_004 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=4,
    severity=Severity.ERROR,
    summary="Wrong region count.",
    message="'{op_name}' has {actual_count} regions, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_005: Missing terminator in block.
ERR_STRUCTURE_005 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=5,
    severity=Severity.ERROR,
    summary="Missing terminator in block.",
    message="block in '{op_name}' region {region_index} is missing a terminator",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("region_index", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_006: Single-block region has wrong block count.
ERR_STRUCTURE_006 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=6,
    severity=Severity.ERROR,
    summary="Single-block region has wrong block count.",
    message="'{op_name}' region {region_index} must have exactly one "
    "block, has {block_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("region_index", ParamKind.U32),
        ErrorParam("block_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_007: BlockArgCount constraint violated.
ERR_STRUCTURE_007 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=7,
    severity=Severity.ERROR,
    summary="Region block argument count mismatch.",
    message="region has {actual_count} block arguments, expected "
    "{expected_count} (one per input tile)",
    params=(
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Ensure the region has exactly one block argument per input tile",
)

# ERR_STRUCTURE_008: YieldCountMatchesResults violated.
ERR_STRUCTURE_008 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=8,
    severity=Severity.ERROR,
    summary="Yield must produce the correct number of values.",
    message="yield has {actual_count} operands, expected {expected_count}",
    params=(
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Yield exactly {expected_count} values",
)

# ERR_STRUCTURE_009: Unknown op kind.
ERR_STRUCTURE_009 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=9,
    severity=Severity.ERROR,
    summary="Unknown op kind.",
    message="unknown op kind {op_kind} in '{op_name}'",
    params=(
        ErrorParam("op_kind", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_STRUCTURE_010: Enum attribute value out of range.
ERR_STRUCTURE_010 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=10,
    severity=Severity.ERROR,
    summary="Enum attribute value out of range.",
    message="attribute '{attr_name}' has enum value {actual_value}, "
    "but only {enum_case_count} values are defined",
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.U32),
        ErrorParam("enum_case_count", ParamKind.U32),
    ),
    fix_hint="'{attr_name}' must have a value in range [0, {enum_case_count})",
)

# ERR_STRUCTURE_011: Non-symbol op at module level.
ERR_STRUCTURE_011 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=11,
    severity=Severity.ERROR,
    summary="Non-symbol-defining op at module level.",
    message="'{op_name}' cannot appear at module level (only symbol-defining "
    "ops like func.def and func.decl are allowed here)",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Move '{op_name}' inside a function body",
)

# ERR_STRUCTURE_012: Operation appears after a block terminator.
ERR_STRUCTURE_012 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=12,
    severity=Severity.ERROR,
    summary="Operation appears after a block terminator.",
    message="'{op_name}' appears after a block terminator",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Move '{op_name}' before the block terminator or remove the "
    "unreachable op",
)

# ERR_STRUCTURE_013: Two variadic fields disagree on element count.
ERR_STRUCTURE_013 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=13,
    severity=Severity.ERROR,
    summary="Variadic field count mismatch.",
    message="'{field_a}' has {actual_count} values but '{field_b}' has "
    "{expected_count}",
    params=(
        ErrorParam("field_a", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("field_b", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Ensure '{field_a}' and '{field_b}' have the same number of values",
)

ALL_STRUCTURE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_STRUCTURE_001,
    ERR_STRUCTURE_002,
    ERR_STRUCTURE_003,
    ERR_STRUCTURE_004,
    ERR_STRUCTURE_005,
    ERR_STRUCTURE_006,
    ERR_STRUCTURE_007,
    ERR_STRUCTURE_008,
    ERR_STRUCTURE_009,
    ERR_STRUCTURE_010,
    ERR_STRUCTURE_011,
    ERR_STRUCTURE_012,
    ERR_STRUCTURE_013,
)
