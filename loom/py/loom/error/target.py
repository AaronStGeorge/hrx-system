# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TARGET domain — shared target contract and legality diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

_TARGET_CONTEXT_PARAMS = (
    ErrorParam("target_key", ParamKind.STRING),
    ErrorParam("export_name", ParamKind.STRING),
    ErrorParam("config_key", ParamKind.STRING),
    ErrorParam("function_name", ParamKind.STRING),
    ErrorParam("op_name", ParamKind.STRING),
)

# ERR_TARGET_001: Target has no lowering contract for an operation.
ERR_TARGET_001 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=1,
    severity=Severity.ERROR,
    summary="Target has no lowering contract for an operation.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "has no target-low contract for '{op_name}' in '@{function_name}'"
    ),
    params=_TARGET_CONTEXT_PARAMS,
    fix_hint=(
        "Select a target contract that handles '{op_name}' or refine the "
        "operation before target-low lowering"
    ),
)

# ERR_TARGET_002: Target contract rejected a value type.
ERR_TARGET_002 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=2,
    severity=Severity.ERROR,
    summary="Target contract rejected a value type.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "type {actual_type} does not match {expected_type}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_type", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine '{field_name}' to a type accepted by the selected target contract"
    ),
)

# ERR_TARGET_003: Target contract rejected an attribute kind.
ERR_TARGET_003 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=3,
    severity=Severity.ERROR,
    summary="Target contract rejected an attribute kind.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "expected a {expected_attr_kind} attribute"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("expected_attr_kind", ParamKind.STRING),
    ),
)

# ERR_TARGET_004: Target contract rejected an enum attribute case.
ERR_TARGET_004 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=4,
    severity=Severity.ERROR,
    summary="Target contract rejected an enum attribute case.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' enum attribute '{field_name}' in "
        "'@{function_name}': expected '{expected_case}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("expected_case", ParamKind.STRING),
    ),
)

# ERR_TARGET_005: Target contract rejected an i64 attribute range.
ERR_TARGET_005 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=5,
    severity=Severity.ERROR,
    summary="Target contract rejected an i64 attribute range.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' i64 attribute '{field_name}' in "
        "'@{function_name}': expected range [{minimum}, {maximum}]"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.I64),
        ErrorParam("maximum", ParamKind.I64),
    ),
)

# ERR_TARGET_006: Target descriptor is unavailable.
ERR_TARGET_006 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=6,
    severity=Severity.ERROR,
    summary="Target descriptor is unavailable.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot lower '{op_name}' in '@{function_name}': descriptor "
        "'{descriptor_key}' is unavailable or requires disabled features"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_007: Target contract required a materializable value.
ERR_TARGET_007 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=7,
    severity=Severity.ERROR,
    summary="Target contract required a materializable value.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "materializer '{materializer_key}' cannot produce the value"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("materializer_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_008: Target contract rejected a low register class.
ERR_TARGET_008 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=8,
    severity=Severity.ERROR,
    summary="Target contract rejected a low register class.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "expected low register class '{register_class}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("register_class", ParamKind.STRING),
    ),
)

# ERR_TARGET_009: Target contract rejected a static dimension multiple.
ERR_TARGET_009 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=9,
    severity=Severity.ERROR,
    summary="Target contract rejected a static dimension multiple.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "static dimension 0 must be divisible by {multiple}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("multiple", ParamKind.U32),
    ),
)

# ERR_TARGET_010: Target contract rejected low register unit counts.
ERR_TARGET_010 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=10,
    severity=Severity.ERROR,
    summary="Target contract rejected low register unit counts.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' fields '{field_name}' and '{other_field_name}' "
        "in '@{function_name}': low register unit counts must match"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("other_field_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_011: Target contract rejected an operand segment count.
ERR_TARGET_011 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=11,
    severity=Severity.ERROR,
    summary="Target contract rejected an operand segment count.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' operand segment '{field_name}' in "
        "'@{function_name}': expected {expected_count} value(s)"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_TARGET_012: Target contract rejected an i64 array count.
ERR_TARGET_012 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=12,
    severity=Severity.ERROR,
    summary="Target contract rejected an i64 array count.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' i64 array '{field_name}' in '@{function_name}': "
        "expected {expected_count} element(s)"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_TARGET_013: Target contract rejected an i64 array element range.
ERR_TARGET_013 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=13,
    severity=Severity.ERROR,
    summary="Target contract rejected an i64 array element range.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' i64 array '{field_name}' element {element} in "
        "'@{function_name}': expected range [{minimum}, {maximum}]"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("element", ParamKind.U32),
        ErrorParam("minimum", ParamKind.I64),
        ErrorParam("maximum", ParamKind.I64),
    ),
)

# ERR_TARGET_014: Target contract rejected an i64 array element set.
ERR_TARGET_014 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=14,
    severity=Severity.ERROR,
    summary="Target contract rejected an i64 array element set.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' i64 array '{field_name}' in '@{function_name}': "
        "all elements must be in range [{minimum}, {maximum}]"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.I64),
        ErrorParam("maximum", ParamKind.I64),
    ),
)

# ERR_TARGET_015: Target contract required a bounded integer value fact.
ERR_TARGET_015 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=15,
    severity=Severity.ERROR,
    summary="Target contract required a bounded integer value fact.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' value '{field_name}' in '@{function_name}': "
        "requires a {signedness} {bit_count}-bit bound"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("signedness", ParamKind.STRING),
        ErrorParam("bit_count", ParamKind.U32),
    ),
)

# ERR_TARGET_016: Target contract required an exact integer value fact.
ERR_TARGET_016 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=16,
    severity=Severity.ERROR,
    summary="Target contract required an exact integer value fact.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' value '{field_name}' in '@{function_name}': "
        "requires an exact integer value fact"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_017: Target contract rejected an integer value fact range.
ERR_TARGET_017 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=17,
    severity=Severity.ERROR,
    summary="Target contract rejected an integer value fact range.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' value '{field_name}' in '@{function_name}': "
        "requires value range [{minimum}, {maximum}]"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.I64),
        ErrorParam("maximum", ParamKind.I64),
    ),
)

# ERR_TARGET_018: Target contract rejected a source memory access.
ERR_TARGET_018 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=18,
    severity=Severity.ERROR,
    summary="Target contract rejected a source memory access.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' source memory access in '@{function_name}': "
        "operation '{operation_kind}' does not match the selected contract"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("operation_kind", ParamKind.STRING),
    ),
)

# ERR_TARGET_019: Target contract rejected a named subject.
ERR_TARGET_019 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=19,
    severity=Severity.ERROR,
    summary="Target contract rejected a named subject.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {subject_kind} '{subject_name}' in "
        "'@{function_name}': {detail}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("subject_kind", ParamKind.STRING),
        ErrorParam("subject_name", ParamKind.STRING),
        ErrorParam("detail", ParamKind.STRING),
    ),
)

# ERR_TARGET_020: Target pipeline entry symbol was not found.
ERR_TARGET_020 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=20,
    severity=Severity.ERROR,
    summary="Target pipeline entry symbol was not found.",
    message=(
        "target pipeline '{pipeline_name}' cannot find selected entry '@{symbol_name}'"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("symbol_name", ParamKind.STRING),
    ),
    fix_hint="Select an existing function symbol for pipeline '{pipeline_name}'",
)

# ERR_TARGET_021: Target pipeline artifact symbol was not found.
ERR_TARGET_021 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=21,
    severity=Severity.ERROR,
    summary="Target pipeline artifact symbol was not found.",
    message=(
        "target pipeline '{pipeline_name}' cannot find selected artifact "
        "'@{artifact_name}'"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("artifact_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Select an existing target.artifact symbol for pipeline '{pipeline_name}'"
    ),
)

# ERR_TARGET_022: Target pipeline entry is not a function body.
ERR_TARGET_022 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=22,
    severity=Severity.ERROR,
    summary="Target pipeline entry is not a function body.",
    message=(
        "target pipeline '{pipeline_name}' entry '@{symbol_name}' must "
        "resolve to a function with a body"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("symbol_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_023: Target pipeline entry has no target profile.
ERR_TARGET_023 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=23,
    severity=Severity.ERROR,
    summary="Target pipeline entry has no target profile.",
    message=(
        "target pipeline '{pipeline_name}' entry '@{function_name}' must "
        "declare a target profile"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Attach a target profile compatible with pipeline '{pipeline_name}' "
        "to '@{function_name}'"
    ),
)

# ERR_TARGET_024: Target pipeline rejected the selected target bundle.
ERR_TARGET_024 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=24,
    severity=Severity.ERROR,
    summary="Target pipeline rejected the selected target bundle.",
    message=(
        "target pipeline '{pipeline_name}' cannot compile '@{function_name}' "
        "with target '{target_key}' export '{export_name}' config "
        "'{config_key}' ({codegen_format}/{artifact_format}/{abi_kind}, "
        "triple '{target_triple}')"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("codegen_format", ParamKind.STRING),
        ErrorParam("artifact_format", ParamKind.STRING),
        ErrorParam("abi_kind", ParamKind.STRING),
        ErrorParam("target_triple", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a target profile/export whose artifact kind is consumed by "
        "pipeline '{pipeline_name}'"
    ),
)

# ERR_TARGET_025: Target pipeline found no compatible entry.
ERR_TARGET_025 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=25,
    severity=Severity.ERROR,
    summary="Target pipeline found no compatible entry.",
    message=(
        "module contains no function with a target profile compatible with "
        "target pipeline '{pipeline_name}'"
    ),
    params=(ErrorParam("pipeline_name", ParamKind.STRING),),
    fix_hint=(
        "Select an entry symbol explicitly or add a target profile compatible "
        "with pipeline '{pipeline_name}'"
    ),
)

# ERR_TARGET_026: Target pipeline found multiple compatible entries.
ERR_TARGET_026 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=26,
    severity=Severity.ERROR,
    summary="Target pipeline found multiple compatible entries.",
    message=(
        "module contains {candidate_count} functions compatible with target "
        "pipeline '{pipeline_name}'"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("candidate_count", ParamKind.U32),
    ),
    fix_hint="Select one entry symbol explicitly for pipeline '{pipeline_name}'",
)

# ERR_TARGET_027: Target artifact exports no entries.
ERR_TARGET_027 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=27,
    severity=Severity.ERROR,
    summary="Target artifact exports no entries.",
    message=(
        "target pipeline '{pipeline_name}' selected artifact "
        "'@{artifact_name}' with no exported entries"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("artifact_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_028: Target pipeline selection options conflict.
ERR_TARGET_028 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=28,
    severity=Severity.ERROR,
    summary="Target pipeline selection options conflict.",
    message=(
        "target pipeline '{pipeline_name}' cannot select both entry "
        "'@{entry_symbol}' and artifact '@{artifact_symbol}'"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("entry_symbol", ParamKind.STRING),
        ErrorParam("artifact_symbol", ParamKind.STRING),
    ),
    fix_hint=(
        "Select either a single entry symbol or an artifact symbol for "
        "pipeline '{pipeline_name}'"
    ),
)

ALL_TARGET_ERRORS = (
    ERR_TARGET_001,
    ERR_TARGET_002,
    ERR_TARGET_003,
    ERR_TARGET_004,
    ERR_TARGET_005,
    ERR_TARGET_006,
    ERR_TARGET_007,
    ERR_TARGET_008,
    ERR_TARGET_009,
    ERR_TARGET_010,
    ERR_TARGET_011,
    ERR_TARGET_012,
    ERR_TARGET_013,
    ERR_TARGET_014,
    ERR_TARGET_015,
    ERR_TARGET_016,
    ERR_TARGET_017,
    ERR_TARGET_018,
    ERR_TARGET_019,
    ERR_TARGET_020,
    ERR_TARGET_021,
    ERR_TARGET_022,
    ERR_TARGET_023,
    ERR_TARGET_024,
    ERR_TARGET_025,
    ERR_TARGET_026,
    ERR_TARGET_027,
    ERR_TARGET_028,
)
