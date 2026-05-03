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

# ERR_TARGET_023: Target pipeline entry has no target record.
ERR_TARGET_023 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=23,
    severity=Severity.ERROR,
    summary="Target pipeline entry has no target record.",
    message=(
        "target pipeline '{pipeline_name}' entry '@{function_name}' must "
        "declare a target record"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Attach a target record compatible with pipeline '{pipeline_name}' "
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
        "Select a target record/export whose artifact kind is consumed by "
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
        "module contains no function with a target record compatible with "
        "target pipeline '{pipeline_name}'"
    ),
    params=(ErrorParam("pipeline_name", ParamKind.STRING),),
    fix_hint=(
        "Select an entry symbol explicitly or add a target record compatible "
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

# ERR_TARGET_029: Target artifact plan root is not an artifact.
ERR_TARGET_029 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=29,
    severity=Severity.ERROR,
    summary="Target artifact plan root is not an artifact.",
    message=(
        "target artifact plan root '@{symbol_name}' must resolve to "
        "target.artifact facts"
    ),
    params=(ErrorParam("symbol_name", ParamKind.STRING),),
    fix_hint="Select a target.artifact symbol for artifact planning",
)

# ERR_TARGET_030: Target artifact closure reaches a non-function symbol.
ERR_TARGET_030 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=30,
    severity=Severity.ERROR,
    summary="Target artifact closure reaches a non-function symbol.",
    message=(
        "target artifact '@{artifact_name}' closure reaches non-function "
        "symbol '@{symbol_name}'"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("symbol_name", ParamKind.STRING),
    ),
    fix_hint="Only call function symbols from functions exported by '@{artifact_name}'",
)

# ERR_TARGET_031: Target artifact closure crosses into another artifact.
ERR_TARGET_031 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=31,
    severity=Severity.ERROR,
    summary="Target artifact closure crosses into another artifact.",
    message=(
        "target artifact '@{artifact_name}' closure reaches function "
        "'@{function_name}' exported by artifact '@{other_artifact_name}'"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("other_artifact_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Keep each artifact's exported function closure private to that artifact"
    ),
)

# ERR_TARGET_032: Target artifact entry has no function body.
ERR_TARGET_032 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=32,
    severity=Severity.ERROR,
    summary="Target artifact entry has no function body.",
    message=(
        "target artifact '@{artifact_name}' entry '@{function_name}' must "
        "have a function body"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_033: Target artifact entry target mismatches the artifact target.
ERR_TARGET_033 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=33,
    severity=Severity.ERROR,
    summary="Target artifact entry target mismatches the artifact target.",
    message=(
        "target artifact '@{artifact_name}' targets '@{artifact_target_name}' "
        "but entry '@{function_name}' targets '@{function_target_name}'"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("artifact_target_name", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("function_target_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Export only functions targeting '@{artifact_target_name}' from "
        "'@{artifact_name}'"
    ),
)

# ERR_TARGET_034: Target artifact export ordinals are partially assigned.
ERR_TARGET_034 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=34,
    severity=Severity.ERROR,
    summary="Target artifact export ordinals are partially assigned.",
    message=(
        "target artifact '@{artifact_name}' has {ordinal_count} explicit "
        "export ordinal(s) for {entry_count} exported entries"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("ordinal_count", ParamKind.U32),
        ErrorParam("entry_count", ParamKind.U32),
    ),
    fix_hint=(
        "Either assign export ordinals to every entry in '@{artifact_name}' "
        "or omit all export ordinals"
    ),
)

# ERR_TARGET_035: Target artifact export ordinal is outside the dense range.
ERR_TARGET_035 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=35,
    severity=Severity.ERROR,
    summary="Target artifact export ordinal is outside the dense range.",
    message=(
        "target artifact '@{artifact_name}' entry '@{function_name}' export "
        "ordinal {export_ordinal} is outside dense range [0, {entry_count})"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("export_ordinal", ParamKind.U32),
        ErrorParam("entry_count", ParamKind.U32),
    ),
)

# ERR_TARGET_036: Target artifact export ordinal is assigned more than once.
ERR_TARGET_036 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=36,
    severity=Severity.ERROR,
    summary="Target artifact export ordinal is assigned more than once.",
    message=(
        "target artifact '@{artifact_name}' export ordinal {export_ordinal} "
        "is assigned to both '@{first_function_name}' and "
        "'@{second_function_name}'"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("export_ordinal", ParamKind.U32),
        ErrorParam("first_function_name", ParamKind.STRING),
        ErrorParam("second_function_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_037: Target artifact export ordinal is not assigned.
ERR_TARGET_037 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=37,
    severity=Severity.ERROR,
    summary="Target artifact export ordinal is not assigned.",
    message=(
        "target artifact '@{artifact_name}' export ordinal {export_ordinal} "
        "is not assigned"
    ),
    params=(
        ErrorParam("artifact_name", ParamKind.STRING),
        ErrorParam("export_ordinal", ParamKind.U32),
    ),
)

# ERR_TARGET_038: Function target is not a target record.
ERR_TARGET_038 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=38,
    severity=Severity.ERROR,
    summary="Function target is not a target record.",
    message=(
        "function '@{function_name}' target '@{target_name}' must resolve to "
        "target-record facts"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_039: Function ABI override field is unsupported.
ERR_TARGET_039 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=39,
    severity=Severity.ERROR,
    summary="Function ABI override field is unsupported.",
    message=(
        "function '@{function_name}' ABI override field '{field_name}' is not supported"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_040: Kernel entry cannot override its ABI.
ERR_TARGET_040 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=40,
    severity=Severity.ERROR,
    summary="Kernel entry cannot override its ABI.",
    message=(
        "kernel entry '@{function_name}' derives its ABI from the target "
        "record and cannot declare an ABI override"
    ),
    params=(ErrorParam("function_name", ParamKind.STRING),),
)

# ERR_TARGET_041: Function target contract has no concrete ABI.
ERR_TARGET_041 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=41,
    severity=Severity.ERROR,
    summary="Function target contract has no concrete ABI.",
    message=(
        "function '@{function_name}' target record '@{target_name}' must "
        "resolve a concrete ABI"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_042: Kernel entry target contract is not a HAL kernel ABI.
ERR_TARGET_042 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=42,
    severity=Severity.ERROR,
    summary="Kernel entry target contract is not a HAL kernel ABI.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "must resolve a HAL kernel ABI"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_043: HAL kernel binding alignment is zero.
ERR_TARGET_043 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=43,
    severity=Severity.ERROR,
    summary="HAL kernel binding alignment is zero.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "must resolve a non-zero HAL binding alignment"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_044: HAL kernel workgroup size is partial.
ERR_TARGET_044 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=44,
    severity=Severity.ERROR,
    summary="HAL kernel workgroup size is partial.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "requires workgroup size dimensions to be all zero or all non-zero"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_045: HAL kernel flat workgroup range is partial.
ERR_TARGET_045 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=45,
    severity=Severity.ERROR,
    summary="HAL kernel flat workgroup range is partial.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "requires flat workgroup size min and max to be both zero or both "
        "non-zero"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_046: HAL kernel flat workgroup range is unordered.
ERR_TARGET_046 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=46,
    severity=Severity.ERROR,
    summary="HAL kernel flat workgroup range is unordered.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "flat workgroup size min {minimum} exceeds max {maximum}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.U32),
        ErrorParam("maximum", ParamKind.U32),
    ),
)

# ERR_TARGET_047: HAL kernel flat workgroup max exceeds target limit.
ERR_TARGET_047 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=47,
    severity=Severity.ERROR,
    summary="HAL kernel flat workgroup max exceeds target limit.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "flat workgroup max {maximum} exceeds target limit {limit}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("maximum", ParamKind.U32),
        ErrorParam("limit", ParamKind.U32),
    ),
)

# ERR_TARGET_048: HAL kernel required workgroup dimension exceeds target limit.
ERR_TARGET_048 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=48,
    severity=Severity.ERROR,
    summary="HAL kernel required workgroup dimension exceeds target limit.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "required workgroup {axis} size {size} exceeds target limit {limit}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("axis", ParamKind.STRING),
        ErrorParam("size", ParamKind.U32),
        ErrorParam("limit", ParamKind.U32),
    ),
)

# ERR_TARGET_049: HAL kernel required flat workgroup size overflows.
ERR_TARGET_049 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=49,
    severity=Severity.ERROR,
    summary="HAL kernel required flat workgroup size overflows.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "required flat workgroup size overflows uint64"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_050: HAL kernel required flat workgroup size exceeds target limit.
ERR_TARGET_050 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=50,
    severity=Severity.ERROR,
    summary="HAL kernel required flat workgroup size exceeds target limit.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "required flat workgroup size {size} exceeds target limit {limit}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("size", ParamKind.U64),
        ErrorParam("limit", ParamKind.U32),
    ),
)

# ERR_TARGET_051: HAL kernel required flat workgroup size is outside range.
ERR_TARGET_051 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=51,
    severity=Severity.ERROR,
    summary="HAL kernel required flat workgroup size is outside range.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "required flat workgroup size {size} is outside range "
        "[{minimum}, {maximum}]"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("size", ParamKind.U64),
        ErrorParam("minimum", ParamKind.U32),
        ErrorParam("maximum", ParamKind.U32),
    ),
)

# ERR_TARGET_052: Source-to-low found no compatible source functions.
ERR_TARGET_052 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=52,
    severity=Severity.ERROR,
    summary="Source-to-low found no compatible source functions.",
    message=(
        "module contains no function compatible with target pipeline '{pipeline_name}'"
    ),
    params=(ErrorParam("pipeline_name", ParamKind.STRING),),
)

# ERR_TARGET_053: Function has no target record.
ERR_TARGET_053 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=53,
    severity=Severity.ERROR,
    summary="Function has no target record.",
    message="function '@{function_name}' must declare a target record",
    params=(ErrorParam("function_name", ParamKind.STRING),),
)

# ERR_TARGET_054: Target-low ABI policy did not map a value.
ERR_TARGET_054 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=54,
    severity=Severity.ERROR,
    summary="Target-low ABI policy did not map a value.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {value_kind} value {value_id} in "
        "'@{function_name}': target-low ABI policy did not produce a "
        "register type"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("value_kind", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U64),
    ),
)

# ERR_TARGET_055: Function predicates reached target-low lowering.
ERR_TARGET_055 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=55,
    severity=Severity.ERROR,
    summary="Function predicates reached target-low lowering.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': {predicate_count} "
        "function predicate(s) require value remapping before target-low "
        "lowering"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("predicate_count", ParamKind.U32),
    ),
)

# ERR_TARGET_056: Tied function results reached target-low lowering.
ERR_TARGET_056 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=56,
    severity=Severity.ERROR,
    summary="Tied function results reached target-low lowering.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': {tied_result_count} "
        "tied result(s) require explicit ABI ownership lowering"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("tied_result_count", ParamKind.U32),
    ),
)

# ERR_TARGET_057: Operation with regions reached target-low source lowering.
ERR_TARGET_057 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=57,
    severity=Severity.ERROR,
    summary="Operation with regions reached target-low source lowering.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': {region_count} nested "
        "region(s) must be lowered away before target-low source lowering"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("region_count", ParamKind.U32),
    ),
)

# ERR_TARGET_058: FP8 scalar type reached target-low legality.
ERR_TARGET_058 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=58,
    severity=Severity.ERROR,
    summary="FP8 scalar type reached target-low legality.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' type {actual_type} in '@{function_name}': "
        "FP8 scalar types require explicit decode or a selected target-low "
        "contract"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

# ERR_TARGET_059: Unknown scalar type reached target-low legality.
ERR_TARGET_059 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=59,
    severity=Severity.ERROR,
    summary="Unknown scalar type reached target-low legality.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' type {actual_type} in '@{function_name}': "
        "scalar type is not known to target-low legality"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

# ERR_TARGET_060: Contract-owned type escaped its contract family.
ERR_TARGET_060 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=60,
    severity=Severity.ERROR,
    summary="Contract-owned type escaped its contract family.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' type {actual_type} in '@{function_name}': "
        "{type_semantic} types must be produced or consumed by an op from a "
        "matching target contract family before target-low lowering"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("type_semantic", ParamKind.STRING),
    ),
)

# ERR_TARGET_061: Vector type shape is not target-low legal.
ERR_TARGET_061 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=61,
    severity=Severity.ERROR,
    summary="Vector type shape is not target-low legal.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' type {actual_type} in '@{function_name}': "
        "target-low legality requires specialized static one-dimensional "
        "vectors"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

# ERR_TARGET_062: Vector lane count is not target-low representable.
ERR_TARGET_062 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=62,
    severity=Severity.ERROR,
    summary="Vector lane count is not target-low representable.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' type {actual_type} in '@{function_name}': "
        "vector lane count is not representable"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

# ERR_TARGET_063: Type has no target-low legality mapping.
ERR_TARGET_063 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=63,
    severity=Severity.ERROR,
    summary="Type has no target-low legality mapping.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' type {actual_type} in '@{function_name}': "
        "no target-low legality mapping exists for this type"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

# ERR_TARGET_064: Operation references an invalid SSA value.
ERR_TARGET_064 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=64,
    severity=Severity.ERROR,
    summary="Operation references an invalid SSA value.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' value {value_id} in '@{function_name}': SSA "
        "value id is outside the module value table"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("value_id", ParamKind.U64),
    ),
)

# ERR_TARGET_065: Target-low legality requires a target provider.
ERR_TARGET_065 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=65,
    severity=Severity.ERROR,
    summary="Target-low legality requires a target provider.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': operation belongs to a "
        "target contract family and requires a target legality provider"
    ),
    params=_TARGET_CONTEXT_PARAMS,
)

# ERR_TARGET_066: Structured control flow reached target-low legality.
ERR_TARGET_066 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=66,
    severity=Severity.ERROR,
    summary="Structured control flow reached target-low legality.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': structured SCF control "
        "flow must be lowered to CFG before target-low lowering"
    ),
    params=_TARGET_CONTEXT_PARAMS,
)

# ERR_TARGET_067: Structured control-flow terminator reached target-low legality.
ERR_TARGET_067 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=67,
    severity=Severity.ERROR,
    summary="Structured control-flow terminator reached target-low legality.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': SCF terminators must be "
        "lowered with their parent structured control-flow op before "
        "target-low lowering"
    ),
    params=_TARGET_CONTEXT_PARAMS,
)

# ERR_TARGET_068: Source-only operation reached target-low legality.
ERR_TARGET_068 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=68,
    severity=Severity.ERROR,
    summary="Source-only operation reached target-low legality.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': source-only operation "
        "must be lowered before target-low legality"
    ),
    params=_TARGET_CONTEXT_PARAMS,
)

# ERR_TARGET_069: Module metadata operation reached executable target-low legality.
ERR_TARGET_069 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=69,
    severity=Severity.ERROR,
    summary="Module metadata operation reached executable target-low legality.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': target record ops are "
        "module metadata and cannot appear inside executable regions"
    ),
    params=_TARGET_CONTEXT_PARAMS,
)

# ERR_TARGET_070: Target-low policy has no source value type mapping.
ERR_TARGET_070 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=70,
    severity=Severity.ERROR,
    summary="Target-low policy has no source value type mapping.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "source type {actual_type} has no target-low register mapping in the "
        "selected policy"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
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
    ERR_TARGET_029,
    ERR_TARGET_030,
    ERR_TARGET_031,
    ERR_TARGET_032,
    ERR_TARGET_033,
    ERR_TARGET_034,
    ERR_TARGET_035,
    ERR_TARGET_036,
    ERR_TARGET_037,
    ERR_TARGET_038,
    ERR_TARGET_039,
    ERR_TARGET_040,
    ERR_TARGET_041,
    ERR_TARGET_042,
    ERR_TARGET_043,
    ERR_TARGET_044,
    ERR_TARGET_045,
    ERR_TARGET_046,
    ERR_TARGET_047,
    ERR_TARGET_048,
    ERR_TARGET_049,
    ERR_TARGET_050,
    ERR_TARGET_051,
    ERR_TARGET_052,
    ERR_TARGET_053,
    ERR_TARGET_054,
    ERR_TARGET_055,
    ERR_TARGET_056,
    ERR_TARGET_057,
    ERR_TARGET_058,
    ERR_TARGET_059,
    ERR_TARGET_060,
    ERR_TARGET_061,
    ERR_TARGET_062,
    ERR_TARGET_063,
    ERR_TARGET_064,
    ERR_TARGET_065,
    ERR_TARGET_066,
    ERR_TARGET_067,
    ERR_TARGET_068,
    ERR_TARGET_069,
    ERR_TARGET_070,
)
