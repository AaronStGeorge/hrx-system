# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU domain — AMDGPU-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

_TARGET_CONTEXT_PARAMS = (
    ErrorParam("target_key", ParamKind.STRING),
    ErrorParam("export_name", ParamKind.STRING),
    ErrorParam("config_key", ParamKind.STRING),
    ErrorParam("function_name", ParamKind.STRING),
    ErrorParam("op_name", ParamKind.STRING),
)

# ERR_AMDGPU_001: AMDGPU buffer view element storage is unsupported.
ERR_AMDGPU_001 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=1,
    severity=Severity.ERROR,
    summary="AMDGPU buffer view element storage is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' field '{field_name}' in "
        "'@{function_name}': type {actual_type} is not a typed view over "
        "{required_storage}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("required_storage", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a HAL buffer view whose element type has byte-addressable AMDGPU "
        "buffer storage"
    ),
)

# ERR_AMDGPU_003: AMDGPU processor override is unknown.
ERR_AMDGPU_003 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=3,
    severity=Severity.ERROR,
    summary="AMDGPU processor override is unknown.",
    message="AMDGPU processor override '{processor}' is not known",
    params=(ErrorParam("processor", ParamKind.STRING),),
    fix_hint="Use a known AMDGPU processor name such as gfx942, gfx950, or gfx1100",
)

# ERR_AMDGPU_004: AMDGPU processor override has no native descriptor set.
ERR_AMDGPU_004 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=4,
    severity=Severity.ERROR,
    summary="AMDGPU processor override has no native descriptor set.",
    message=(
        "AMDGPU processor override '{processor}' is known but has no "
        "native target-low descriptor set"
    ),
    params=(ErrorParam("processor", ParamKind.STRING),),
    fix_hint="Use an AMDGPU processor with native target-low descriptor coverage",
)

# ERR_AMDGPU_005: AMDGPU processor override changes descriptor set.
ERR_AMDGPU_005 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=5,
    severity=Severity.ERROR,
    summary="AMDGPU processor override changes descriptor set.",
    message=(
        "AMDGPU processor override '{processor}' selects descriptor set "
        "'{target_descriptor_set}' but target record '@{target_name}' uses "
        "descriptor set '{record_descriptor_set}'"
    ),
    params=(
        ErrorParam("processor", ParamKind.STRING),
        ErrorParam("target_descriptor_set", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("record_descriptor_set", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a target record from the same AMDGPU descriptor-set family as "
        "the requested processor"
    ),
)

# ERR_AMDGPU_006: AMDGPU HAL-kernel ABI resource count overflows.
ERR_AMDGPU_006 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=6,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI resource count overflows.",
    message=(
        "AMDGPU HAL-kernel ABI has {resource_count} HAL binding resources, "
        "but at most {max_resource_count} fit in the kernarg segment"
    ),
    params=(
        ErrorParam("resource_count", ParamKind.U64),
        ErrorParam("max_resource_count", ParamKind.U64),
    ),
    fix_hint="Split the kernel ABI or reduce the number of HAL binding resources",
)

# ERR_AMDGPU_007: AMDGPU HAL-kernel ABI resource import kind is unsupported.
ERR_AMDGPU_007 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=7,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI resource import kind is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI requires low.resource import_kind enum value "
        "{expected_import_kind}, but found {actual_import_kind}"
    ),
    params=(
        ErrorParam("expected_import_kind", ParamKind.U32),
        ErrorParam("actual_import_kind", ParamKind.U32),
    ),
    fix_hint="Use low.resource<hal_binding> for AMDGPU HAL kernel resources",
)

# ERR_AMDGPU_008: AMDGPU HAL-kernel ABI resource result type is unsupported.
ERR_AMDGPU_008 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=8,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI resource result type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI requires hal_binding resources to produce "
        "descriptor register-class ID {expected_reg_class_id} with "
        "{expected_unit_count} unit(s), but found {actual_type}"
    ),
    params=(
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_class_id", ParamKind.U32),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Use a 64-bit pointer resource value for HAL binding imports",
)

# ERR_AMDGPU_009: AMDGPU HAL-kernel ABI binding index is negative.
ERR_AMDGPU_009 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=9,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is negative.",
    message=(
        "AMDGPU HAL-kernel ABI requires non-negative binding indexes, but "
        "found {binding_index}"
    ),
    params=(ErrorParam("binding_index", ParamKind.I64),),
    fix_hint="Number HAL binding resources densely from zero",
)

# ERR_AMDGPU_010: AMDGPU HAL-kernel ABI binding index is not dense.
ERR_AMDGPU_010 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=10,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is not dense.",
    message=(
        "AMDGPU HAL-kernel ABI found binding index {binding_index}, but "
        "{resource_count} resource(s) require indexes in [0, {resource_count})"
    ),
    params=(
        ErrorParam("binding_index", ParamKind.U64),
        ErrorParam("resource_count", ParamKind.U64),
    ),
    fix_hint="Number HAL binding resources densely from zero without gaps",
)

# ERR_AMDGPU_011: AMDGPU HAL-kernel ABI binding index is duplicated.
ERR_AMDGPU_011 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=11,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is duplicated.",
    message=(
        "AMDGPU HAL-kernel ABI binding index {binding_index} is defined more than once"
    ),
    params=(ErrorParam("binding_index", ParamKind.U64),),
    fix_hint="Give each HAL binding resource a unique dense index",
)

# ERR_AMDGPU_012: AMDGPU HAL-kernel ABI binding index is missing.
ERR_AMDGPU_012 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=12,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is missing.",
    message=(
        "AMDGPU HAL-kernel ABI requires binding index {binding_index}, but "
        "no low.resource defines it"
    ),
    params=(ErrorParam("binding_index", ParamKind.U64),),
    fix_hint="Define HAL binding resources densely from zero without gaps",
)

# ERR_AMDGPU_013: AMDGPU HAL buffer descriptor pseudo attribute is invalid.
ERR_AMDGPU_013 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=13,
    severity=Severity.ERROR,
    summary="AMDGPU HAL buffer descriptor pseudo attribute is invalid.",
    message=(
        "AMDGPU HAL buffer descriptor pseudo requires attribute "
        "'{attr_name}' with kind {expected_kind}, but found kind {actual_kind}"
    ),
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("expected_kind", ParamKind.U32),
        ErrorParam("actual_kind", ParamKind.U32),
    ),
    fix_hint="Run low descriptor verification before AMDGPU HAL ABI verification",
)

# ERR_AMDGPU_014: AMDGPU HAL buffer descriptor cache swizzle is unsupported.
ERR_AMDGPU_014 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=14,
    severity=Severity.ERROR,
    summary="AMDGPU HAL buffer descriptor cache swizzle is unsupported.",
    message=(
        "AMDGPU descriptor set '{descriptor_set_key}' cannot encode HAL "
        "buffer descriptor cache_swizzle_stride {cache_swizzle_stride}"
    ),
    params=(
        ErrorParam("descriptor_set_key", ParamKind.STRING),
        ErrorParam("cache_swizzle_stride", ParamKind.U64),
    ),
    fix_hint=(
        "Use zero cache_swizzle_stride or select an AMDGPU descriptor set "
        "that supports stride14 cache swizzle"
    ),
)

# ERR_AMDGPU_015: AMDGPU HAL-kernel ABI live-in is duplicated.
ERR_AMDGPU_015 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=15,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI live-in is duplicated.",
    message=(
        "AMDGPU HAL-kernel ABI live-in source '{source_name}' is defined more than once"
    ),
    params=(ErrorParam("source_name", ParamKind.STRING),),
    fix_hint="Keep at most one low.live_in for each AMDGPU ABI source",
)

# ERR_AMDGPU_016: AMDGPU HAL-kernel ABI workitem live-in forms are mixed.
ERR_AMDGPU_016 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=16,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI workitem live-in forms are mixed.",
    message=(
        "AMDGPU HAL-kernel ABI cannot mix workitem live-in source "
        "'{source_name}' with '{conflicting_source_name}'"
    ),
    params=(
        ErrorParam("source_name", ParamKind.STRING),
        ErrorParam("conflicting_source_name", ParamKind.STRING),
    ),
    fix_hint="Use either packed or unpacked workitem-id live-ins, not both",
)

# ERR_AMDGPU_017: AMDGPU HAL-kernel ABI live-in result type is unsupported.
ERR_AMDGPU_017 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=17,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI live-in result type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI live-in source '{source_name}' requires "
        "descriptor register-class ID {expected_reg_class_id} with "
        "{expected_unit_count} unit(s), but found {actual_type}"
    ),
    params=(
        ErrorParam("source_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_class_id", ParamKind.U32),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Use the register shape required by the AMDGPU ABI live-in source",
)

# ERR_AMDGPU_018: AMDGPU HAL-kernel ABI M0 live-in result type is unsupported.
ERR_AMDGPU_018 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=18,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI M0 live-in result type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI live-in source '{source_name}' must be a "
        "single unspillable physical register, but found {actual_type}"
    ),
    params=(
        ErrorParam("source_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
    fix_hint="Use the AMDGPU M0 physical register type for the M0 live-in",
)

# ERR_AMDGPU_019: AMDGPU packed memory payload does not fill registers.
ERR_AMDGPU_019 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=19,
    severity=Severity.ERROR,
    summary="AMDGPU packed memory payload footprint is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': payload "
        "type {payload_type} has {payload_bit_count} payload bit(s), but the "
        "selected packed memory register footprint holds {register_bit_count} "
        "bit(s)"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("payload_type", ParamKind.TYPE),
        ErrorParam("payload_bit_count", ParamKind.U32),
        ErrorParam("register_bit_count", ParamKind.U32),
    ),
    fix_hint=(
        "Widen the packed payload to fill complete 32-bit AMDGPU memory "
        "registers or use a supported scalar D16 access"
    ),
)

# ERR_AMDGPU_020: AMDGPU flat dynamic memory address is unsupported.
ERR_AMDGPU_020 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=20,
    severity=Severity.ERROR,
    summary="AMDGPU flat dynamic memory address is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "{operation_kind} in {memory_space} memory has flat dynamic address "
        "term {dynamic_term_index} with byte stride {byte_stride}, byte range "
        "[{byte_range_lo}, {byte_range_hi}], and byte shift {byte_shift}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("operation_kind", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("dynamic_term_index", ParamKind.U32),
        ErrorParam("byte_stride", ParamKind.I64),
        ErrorParam("byte_range_lo", ParamKind.I64),
        ErrorParam("byte_range_hi", ParamKind.I64),
        ErrorParam("byte_shift", ParamKind.U32),
    ),
    fix_hint=(
        "Prove a non-negative 32-bit dynamic byte address before AMDGPU "
        "flat memory lowering, or lower through an address form that supports "
        "the dynamic term"
    ),
)

# ERR_AMDGPU_021: AMDGPU matrix descriptor selection rejected a source contract.
ERR_AMDGPU_021 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=21,
    severity=Severity.ERROR,
    summary="AMDGPU matrix descriptor selection rejected a source contract.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': matrix "
        "constraint '{matrix_constraint}' is not satisfied "
        "(source_bits={source_rejection_bits}, target_bits={target_rejection_bits})"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("matrix_constraint", ParamKind.STRING),
        ErrorParam("source_rejection_bits", ParamKind.U32),
        ErrorParam("target_rejection_bits", ParamKind.U32),
    ),
    fix_hint=(
        "Select a matrix shape, payload, feature set, and fragment schema "
        "supported by the AMDGPU descriptor set"
    ),
)

# ERR_AMDGPU_022: AMDGPU matrix descriptor has no target-low packet mapping.
ERR_AMDGPU_022 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=22,
    severity=Severity.ERROR,
    summary="AMDGPU matrix descriptor has no target-low packet mapping.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' descriptor '{descriptor_name}' "
        "in '@{function_name}': descriptor constraint '{descriptor_constraint}' "
        "is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("descriptor_constraint", ParamKind.STRING),
    ),
    fix_hint=(
        "Use an AMDGPU matrix descriptor that has a selected target-low packet "
        "mapping in the active descriptor set"
    ),
)

# ERR_AMDGPU_023: AMDGPU source-to-low constraint is not satisfied.
ERR_AMDGPU_023 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=23,
    severity=Severity.ERROR,
    summary="AMDGPU source-to-low constraint is not satisfied.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "source-to-low constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Legalize, refine, or decompose the operation before AMDGPU "
        "source-to-low lowering"
    ),
)

# ERR_AMDGPU_024: AMDGPU memory cache policy is not encodable.
ERR_AMDGPU_024 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=24,
    severity=Severity.ERROR,
    summary="AMDGPU memory cache policy is not encodable.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': memory "
        "cache policy {cache_scope}/{cache_temporal} for {memory_space} "
        "memory is not encodable by descriptor set '{descriptor_set_key}' "
        "because constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("cache_scope", ParamKind.STRING),
        ErrorParam("cache_temporal", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Use an AMDGPU cache policy supported by the selected descriptor set "
        "or legalize the access before source-to-low lowering"
    ),
)

# ERR_AMDGPU_025: AMDGPU HAL-kernel ABI direct argument type is unsupported.
ERR_AMDGPU_025 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=25,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI direct argument type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI direct argument {argument_index} requires "
        "descriptor register-class ID {expected_reg_class_id} with "
        "1-{expected_unit_count} unit(s), but found {actual_type}"
    ),
    params=(
        ErrorParam("argument_index", ParamKind.U32),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_class_id", ParamKind.U32),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Lower direct HAL kernel arguments to the ABI register shape",
)

# ERR_AMDGPU_026: AMDGPU target subgroup size is unsupported.
ERR_AMDGPU_026 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=26,
    severity=Severity.ERROR,
    summary="AMDGPU target subgroup size is unsupported.",
    message=(
        "AMDGPU target record '@{target_name}' selects subgroup_size "
        "{subgroup_size}, but processor '{processor}' does not support that "
        "wavefront size"
    ),
    params=(
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("subgroup_size", ParamKind.U64),
        ErrorParam("processor", ParamKind.STRING),
    ),
    fix_hint="Use a wavefront size supported by the selected AMDGPU processor",
)

ALL_AMDGPU_ERRORS = (
    ERR_AMDGPU_001,
    ERR_AMDGPU_003,
    ERR_AMDGPU_004,
    ERR_AMDGPU_005,
    ERR_AMDGPU_006,
    ERR_AMDGPU_007,
    ERR_AMDGPU_008,
    ERR_AMDGPU_009,
    ERR_AMDGPU_010,
    ERR_AMDGPU_011,
    ERR_AMDGPU_012,
    ERR_AMDGPU_013,
    ERR_AMDGPU_014,
    ERR_AMDGPU_015,
    ERR_AMDGPU_016,
    ERR_AMDGPU_017,
    ERR_AMDGPU_018,
    ERR_AMDGPU_019,
    ERR_AMDGPU_020,
    ERR_AMDGPU_021,
    ERR_AMDGPU_022,
    ERR_AMDGPU_023,
    ERR_AMDGPU_024,
    ERR_AMDGPU_025,
    ERR_AMDGPU_026,
)
