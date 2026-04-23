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

# ERR_LOWERING_012: Low descriptor enum immediate value is not in domain.
ERR_LOWERING_012 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=12,
    severity=Severity.ERROR,
    summary="Low descriptor enum immediate value is not in domain.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' immediate "
        "'{immediate_name}' has enum value '{actual_value}', expected a "
        "value from enum domain '{enum_domain}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("immediate_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.STRING),
        ErrorParam("enum_domain", ParamKind.STRING),
    ),
    fix_hint=(
        "Choose an enum token or numeric value declared by enum domain "
        "'{enum_domain}' for '{immediate_name}'"
    ),
)

# ERR_LOWERING_013: Low function call signature count mismatch.
ERR_LOWERING_013 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=13,
    severity=Severity.ERROR,
    summary="Low function call signature count mismatch.",
    message=(
        "low function call callee '@{callee_name}' {field_kind} count is "
        "{actual_count}, expected {expected_count}"
    ),
    params=(
        ErrorParam("callee_name", ParamKind.STRING),
        ErrorParam("field_kind", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint=(
        "Match the low function call {field_kind} list to the callee "
        "low-function signature"
    ),
)

# ERR_LOWERING_014: Low function call signature type mismatch.
ERR_LOWERING_014 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=14,
    severity=Severity.ERROR,
    summary="Low function call signature type mismatch.",
    message=(
        "low function call callee '@{callee_name}' {field_kind} "
        "{field_index} has type {actual_type}, expected callee "
        "{callee_field_kind} type "
        "{expected_type}"
    ),
    params=(
        ErrorParam("callee_name", ParamKind.STRING),
        ErrorParam("field_kind", ParamKind.STRING),
        ErrorParam("field_index", ParamKind.U32),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("callee_field_kind", ParamKind.STRING),
        ErrorParam("expected_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Match the low function call {field_kind} type to the callee "
        "low-function signature"
    ),
)

# ERR_LOWERING_015: Low call purity conflicts with callee effects.
ERR_LOWERING_015 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=15,
    severity=Severity.ERROR,
    summary="Low call purity conflicts with callee effects.",
    message=(
        "low pure call callee '@{callee_name}' through '{boundary_name}' "
        "is invalid: {reason} ({effect_count} effect record(s), "
        "{clobber_count} clobber record(s))"
    ),
    params=(
        ErrorParam("callee_name", ParamKind.STRING),
        ErrorParam("boundary_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
        ErrorParam("effect_count", ParamKind.U32),
        ErrorParam("clobber_count", ParamKind.U32),
    ),
    fix_hint=(
        "Remove the pure marker, mark the direct callee pure after proving its "
        "body, or remove the adapter effect/clobber records after proving the "
        "low boundary has no observable effects"
    ),
)

# ERR_LOWERING_016: Low function contract is invalid.
ERR_LOWERING_016 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=16,
    severity=Severity.ERROR,
    summary="Low function contract is invalid.",
    message=(
        "low function '@{function_name}' {contract_name} contract is invalid: {reason}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("contract_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Keep low function exactness and imported-code attrs complete and on "
        "the operation kind that owns them"
    ),
)

# ERR_LOWERING_017: Low structural storage op is invalid.
ERR_LOWERING_017 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=17,
    severity=Severity.ERROR,
    summary="Low structural storage op is invalid.",
    message=("low structural op '{op_name}' field '{field_name}' is invalid: {reason}"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Keep low structural records owned by low functions and reference "
        "them with matching function, type, and target ABI contracts"
    ),
)

# ERR_LOWERING_018: Low ABI resource register type is not accepted.
ERR_LOWERING_018 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=18,
    severity=Severity.ERROR,
    summary="Low ABI resource register type is not accepted.",
    message=(
        "low function '{function_name}' resource '{resource_name}' has ABI "
        "type {actual_type}, which descriptor set '{descriptor_set}' rejects: "
        "{reason}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("resource_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("descriptor_set", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a register class and allocation-unit count accepted by the "
        "selected low descriptor set"
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
    ERR_LOWERING_012,
    ERR_LOWERING_013,
    ERR_LOWERING_014,
    ERR_LOWERING_015,
    ERR_LOWERING_016,
    ERR_LOWERING_017,
    ERR_LOWERING_018,
)
