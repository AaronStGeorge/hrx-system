# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LOWERING domain — pass legality and unsupported mappings."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_LOWERING_001: Pass has no legal lowering for an op.
ERR_LOWERING_001 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=1,
    severity=Severity.ERROR,
    summary="Operation has no legal lowering.",
    message="{op_name} cannot be lowered by {pass_name}: {reason}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint="Run a refinement pass that makes the operation legal for "
    "{pass_name}, or lower it with a pass that supports this operation",
)

# ERR_LOWERING_002: Pass-level refinement failed.
ERR_LOWERING_002 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=2,
    severity=Severity.ERROR,
    summary="Pass refinement failed.",
    message="{pass_name} failed while refining {scope}: {reason}",
    params=(
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("scope", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint="Refine boundary facts/types, specialize incompatible call paths, "
    "or split the recursive/SCC structure before running {pass_name}",
)

# ERR_LOWERING_003: Low descriptor set is not available.
ERR_LOWERING_003 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=3,
    severity=Severity.ERROR,
    summary="Low descriptor set is not available.",
    message=(
        "low function '@{function_name}' target '@{target_name}' requires "
        "descriptor set '{descriptor_set_key}', but the descriptor registry "
        "does not provide it"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint="Link the descriptor package named by '{descriptor_set_key}' or "
    "choose a target config whose descriptor set is available",
)

# ERR_LOWERING_004: Low descriptor is not available.
ERR_LOWERING_004 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=4,
    severity=Severity.ERROR,
    summary="Low descriptor is not available.",
    message=(
        "low function '@{function_name}' uses descriptor '{opcode}', but "
        "descriptor set '{descriptor_set_key}' does not define it"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint="Choose an opcode from descriptor set '{descriptor_set_key}' or "
    "select a target config with a descriptor set that defines '{opcode}'",
)

# ERR_LOWERING_005: Low descriptor feature is not enabled.
ERR_LOWERING_005 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=5,
    severity=Severity.ERROR,
    summary="Low descriptor feature is not enabled.",
    message=(
        "low function '@{function_name}' uses descriptor '{opcode}' from set "
        "'{descriptor_set_key}', but feature word {feature_word_index} is "
        "missing bits {missing_feature_bits}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
        ErrorParam("feature_word_index", ParamKind.U32),
        ErrorParam("missing_feature_bits", ParamKind.U64),
    ),
    fix_hint="Enable the missing target feature bits or choose a descriptor "
    "legal for the selected target config",
)

# ERR_LOWERING_006: Low descriptor register type constraint violated.
ERR_LOWERING_006 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=6,
    severity=Severity.ERROR,
    summary="Low descriptor register type constraint violated.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' "
        "{field_kind} '{field_name}' has type {actual_type}, expected "
        "register class in [{expected_reg_classes}] with "
        "{expected_unit_count} unit(s)"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("field_kind", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_classes", ParamKind.STRING),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Choose a register type accepted by descriptor '{opcode}' for "
    "field '{field_name}' or select a descriptor whose register contract "
    "matches the packet",
)

# ERR_LOWERING_007: Low descriptor immediate attribute is missing.
ERR_LOWERING_007 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=7,
    severity=Severity.ERROR,
    summary="Low descriptor immediate attribute is missing.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' requires "
        "immediate attribute '{immediate_name}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("immediate_name", ParamKind.STRING),
    ),
    fix_hint="Provide attribute '{immediate_name}' in the low packet attrs "
    "for descriptor '{opcode}'",
)

# ERR_LOWERING_008: Low descriptor immediate attribute is not declared.
ERR_LOWERING_008 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=8,
    severity=Severity.ERROR,
    summary="Low descriptor immediate attribute is not declared.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' has "
        "attribute '{attr_name}', but the descriptor declares no such "
        "immediate"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("attr_name", ParamKind.STRING),
    ),
    fix_hint="Remove attribute '{attr_name}' or add a descriptor immediate "
    "row that owns it",
)

# ERR_LOWERING_009: Low descriptor immediate attribute kind is invalid.
ERR_LOWERING_009 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=9,
    severity=Severity.ERROR,
    summary="Low descriptor immediate attribute kind is invalid.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' immediate "
        "'{immediate_name}' has attribute kind {actual_kind}, expected "
        "{expected_kind}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("immediate_name", ParamKind.STRING),
        ErrorParam("actual_kind", ParamKind.U32),
        ErrorParam("expected_kind", ParamKind.STRING),
    ),
    fix_hint="Encode immediate '{immediate_name}' using {expected_kind}",
)

# ERR_LOWERING_010: Low descriptor immediate attribute is out of range.
ERR_LOWERING_010 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=10,
    severity=Severity.ERROR,
    summary="Low descriptor immediate attribute is out of range.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' immediate "
        "'{immediate_name}' has value {actual_value}, expected "
        "{expected_range}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("immediate_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.I64),
        ErrorParam("expected_range", ParamKind.STRING),
    ),
    fix_hint="Choose an immediate value in {expected_range} for '{immediate_name}'",
)

# ERR_LOWERING_011: Low descriptor register constraint type mismatch.
ERR_LOWERING_011 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=11,
    severity=Severity.ERROR,
    summary="Low descriptor register constraint type mismatch.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' "
        "{constraint_kind} constraint requires {lhs_field_kind} "
        "'{lhs_field_name}' type {lhs_type} to match {rhs_field_kind} "
        "'{rhs_field_name}' type {rhs_type}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("constraint_kind", ParamKind.STRING),
        ErrorParam("lhs_field_kind", ParamKind.STRING),
        ErrorParam("lhs_field_name", ParamKind.STRING),
        ErrorParam("lhs_type", ParamKind.TYPE),
        ErrorParam("rhs_field_kind", ParamKind.STRING),
        ErrorParam("rhs_field_name", ParamKind.STRING),
        ErrorParam("rhs_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Choose matching register types for the constrained packet fields or "
        "select a descriptor without the {constraint_kind} constraint"
    ),
)

ALL_LOWERING_ERRORS: tuple[ErrorDef, ...] = (
    ERR_LOWERING_001,
    ERR_LOWERING_002,
    ERR_LOWERING_003,
    ERR_LOWERING_004,
    ERR_LOWERING_005,
    ERR_LOWERING_006,
    ERR_LOWERING_007,
    ERR_LOWERING_008,
    ERR_LOWERING_009,
    ERR_LOWERING_010,
    ERR_LOWERING_011,
)
