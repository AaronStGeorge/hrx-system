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

# ERR_LOWERING_019: Low operation reads an undefined register part.
ERR_LOWERING_019 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=19,
    severity=Severity.ERROR,
    summary="Low operation reads an undefined register part.",
    message=(
        "low function '@{function_name}' op '{op_name}' operand "
        "'{field_name}' requires register part mask {required_mask}, but "
        "the value defines only mask {defined_mask}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("required_mask", ParamKind.U64),
        ErrorParam("defined_mask", ParamKind.U64),
    ),
    fix_hint=(
        "Compose the missing register part with an explicitly tied partial "
        "write before using the value as a full-register operand"
    ),
)

# ERR_LOWERING_020: Static vector scalarization lane count is not representable.
ERR_LOWERING_020 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=20,
    severity=Severity.ERROR,
    summary="Static vector scalarization lane count is not representable.",
    message=(
        "{op_name} cannot be lowered by {pass_name} because static vector type "
        "{vector_type} has more lanes than scalarization can represent"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("vector_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Refine the vector shape before scalarization or lower it with a "
        "target primitive that preserves the vector aggregate"
    ),
)

# ERR_LOWERING_021: Low register type has no descriptor register class.
ERR_LOWERING_021 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=21,
    severity=Severity.ERROR,
    summary="Low register type has no descriptor register class.",
    message=(
        "low function '@{function_name}' {value_kind} '{value_name}' has "
        "type {actual_type}, whose register class is not defined by descriptor "
        "set '{descriptor_set}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_kind", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("descriptor_set", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a register class declared by descriptor set '{descriptor_set}' "
        "or select a target config whose descriptor set owns this ABI type"
    ),
)

# ERR_LOWERING_022: Kernel async group has no wait in the current stream.
ERR_LOWERING_022 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=22,
    severity=Severity.ERROR,
    summary="Kernel async group has no wait in the current stream.",
    message=(
        "{phase_name} requires {op_name} to be waited in the current "
        "straight-line async stream"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the async group in the same straight-line stream or lower it "
        "with a pipeline-aware async strategy"
    ),
)

# ERR_LOWERING_023: Kernel async group is carried outside the stream.
ERR_LOWERING_023 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=23,
    severity=Severity.ERROR,
    summary="Kernel async group is carried outside the stream.",
    message=(
        "{phase_name} cannot lower {op_name} whose group value has a non-wait use"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Keep async groups in a straight-line group/wait stream or run a "
        "pipeline-aware legality path before lowering"
    ),
)

# ERR_LOWERING_024: Kernel async movement cannot be described.
ERR_LOWERING_024 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=24,
    severity=Severity.ERROR,
    summary="Kernel async movement cannot be described.",
    message=(
        "{phase_name} cannot describe the movement endpoints for {op_name}; "
        "movement rejection bits are {rejection_bits}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("rejection_bits", ParamKind.U64),
    ),
    fix_hint=(
        "Refine async transfer operands so their source and destination view "
        "regions can be described by movement analysis"
    ),
)

# ERR_LOWERING_025: Kernel async token producer is not an async view movement.
ERR_LOWERING_025 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=25,
    severity=Severity.ERROR,
    summary="Kernel async token producer is not an async view movement.",
    message=(
        "{phase_name} requires the token producer for {op_name} to be an async "
        "view movement"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Group only tokens produced by kernel async transfer operations whose "
        "destination is a view"
    ),
)

# ERR_LOWERING_026: Kernel async destination overlaps a pending destination.
ERR_LOWERING_026 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=26,
    severity=Severity.ERROR,
    summary="Kernel async destination overlaps a pending destination.",
    message=(
        "{phase_name} found {op_name} whose destination may overlap an earlier "
        "uncompleted async destination"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the earlier async group before issuing an overlapping async "
        "destination or prove the destination views are disjoint"
    ),
)

# ERR_LOWERING_027: Synchronous write overlaps a pending async destination.
ERR_LOWERING_027 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=27,
    severity=Severity.ERROR,
    summary="Synchronous write overlaps a pending async destination.",
    message=(
        "{phase_name} found {op_name} writing a view that may overlap a "
        "pending async destination before wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the pending async group before the synchronous write or prove the "
        "written view is disjoint"
    ),
)

# ERR_LOWERING_028: Synchronous read overlaps a pending async destination.
ERR_LOWERING_028 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=28,
    severity=Severity.ERROR,
    summary="Synchronous read overlaps a pending async destination.",
    message=(
        "{phase_name} found {op_name} reading a view that may observe a "
        "pending async destination before wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the pending async group before the synchronous read or prove the "
        "read view is disjoint"
    ),
)

# ERR_LOWERING_029: Kernel async wait references an uncommitted group.
ERR_LOWERING_029 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=29,
    severity=Severity.ERROR,
    summary="Kernel async wait references an uncommitted group.",
    message=(
        "{phase_name} requires {op_name} to wait a group committed in the "
        "current straight-line async stream"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Move the wait after the matching kernel.async.group in the same block "
        "or use a pipeline-aware async lowering path"
    ),
)

# ERR_LOWERING_030: Kernel async wait references an already completed group.
ERR_LOWERING_030 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=30,
    severity=Severity.ERROR,
    summary="Kernel async wait references an already completed group.",
    message=(
        "{phase_name} found {op_name} waiting an async group already completed "
        "by an earlier wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Remove the duplicate wait or wait a younger group that is still outstanding"
    ),
)

# ERR_LOWERING_031: Kernel async wait count does not match stream depth.
ERR_LOWERING_031 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=31,
    severity=Severity.ERROR,
    summary="Kernel async wait count does not match stream depth.",
    message=(
        "{phase_name} found {op_name} with newer_groups {actual_newer_groups}, "
        "but {expected_newer_groups} younger async groups remain outstanding"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("actual_newer_groups", ParamKind.I64),
        ErrorParam("expected_newer_groups", ParamKind.U64),
    ),
    fix_hint=(
        "Set newer_groups to the number of younger uncompleted groups that "
        "remain after this wait"
    ),
)

# ERR_LOWERING_032: Kernel async group leaves a block before wait.
ERR_LOWERING_032 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=32,
    severity=Severity.ERROR,
    summary="Kernel async group leaves a block before wait.",
    message=("{phase_name} requires {op_name} to be waited before leaving its block"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the async group along every control-flow path before leaving the "
        "block or move the async stream into a pipeline-aware region"
    ),
)

# ERR_LOWERING_033: Kernel async group token has no producer op.
ERR_LOWERING_033 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=33,
    severity=Severity.ERROR,
    summary="Kernel async group token has no producer op.",
    message=(
        "{phase_name} requires every token operand of {op_name} to be produced "
        "by a local async transfer op"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Commit only tokens produced in the current function by kernel async "
        "transfer operations"
    ),
)

# ERR_LOWERING_034: Vector transform permutation is not statically proven.
ERR_LOWERING_034 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=34,
    severity=Severity.ERROR,
    summary="Vector transform permutation is not statically proven.",
    message=("{pass_name} requires {op_name} permutation lanes to be statically exact"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the permutation lane facts before vector scalarization or "
        "specialize the transform through a target primitive"
    ),
)

# ERR_LOWERING_035: Vector transform permutation repeats a source lane.
ERR_LOWERING_035 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=35,
    severity=Severity.ERROR,
    summary="Vector transform permutation repeats a source lane.",
    message=(
        "{pass_name} requires {op_name} permutation to name each source lane "
        "once per last-axis slice"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a bijective permutation for each last-axis slice before vector "
        "scalarization"
    ),
)

# ERR_LOWERING_036: Vector transform permutation source lane is out of bounds.
ERR_LOWERING_036 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=36,
    severity=Severity.ERROR,
    summary="Vector transform permutation source lane is out of bounds.",
    message=(
        "{pass_name} requires {op_name} permutation lanes to reference "
        "in-bounds source lanes"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint="Clamp or specialize permutation lanes to the source last-axis extent",
)

# ERR_LOWERING_037: SCF to CFG requires a positive counted-loop step.
ERR_LOWERING_037 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=37,
    severity=Severity.ERROR,
    summary="SCF to CFG requires a positive counted-loop step.",
    message="{pass_name} requires {op_name} step to be fact-proven positive",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the loop step facts or normalize the loop before converting "
        "structured control flow to CFG"
    ),
)

# ERR_LOWERING_038: SCF to CFG cannot preserve tied result ownership.
ERR_LOWERING_038 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=38,
    severity=Severity.ERROR,
    summary="SCF to CFG cannot preserve tied result ownership.",
    message="{pass_name} cannot preserve tied result ownership on {op_name}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Lower ownership transfers before CFG conversion or keep the "
        "structured op until CFG block arguments can model the transfer"
    ),
)

# ERR_LOWERING_039: Vector transform descriptor is not locally decodable.
ERR_LOWERING_039 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=39,
    severity=Severity.ERROR,
    summary="Vector transform descriptor is not locally decodable.",
    message=(
        "{pass_name} requires {op_name} transform to be a local "
        "#numeric_transform encoding.define"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Specialize the transform descriptor into the function before vector "
        "scalarization or lower the transform through a target primitive"
    ),
)

# ERR_LOWERING_040: Vector transform last-axis extent is dynamic.
ERR_LOWERING_040 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=40,
    severity=Severity.ERROR,
    summary="Vector transform last-axis extent is dynamic.",
    message=("{pass_name} requires {op_name} last-axis transform extent to be static"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the transform last-axis extent before vector scalarization or "
        "lower the transform through a target primitive"
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
    ERR_LOWERING_019,
    ERR_LOWERING_020,
    ERR_LOWERING_021,
    ERR_LOWERING_022,
    ERR_LOWERING_023,
    ERR_LOWERING_024,
    ERR_LOWERING_025,
    ERR_LOWERING_026,
    ERR_LOWERING_027,
    ERR_LOWERING_028,
    ERR_LOWERING_029,
    ERR_LOWERING_030,
    ERR_LOWERING_031,
    ERR_LOWERING_032,
    ERR_LOWERING_033,
    ERR_LOWERING_034,
    ERR_LOWERING_035,
    ERR_LOWERING_036,
    ERR_LOWERING_037,
    ERR_LOWERING_038,
    ERR_LOWERING_039,
    ERR_LOWERING_040,
)
