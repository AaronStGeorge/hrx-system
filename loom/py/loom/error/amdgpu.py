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

# ERR_AMDGPU_002: AMDGPU buffer view byte offset is not statically encodable.
ERR_AMDGPU_002 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=2,
    severity=Severity.ERROR,
    summary="AMDGPU buffer view byte offset is not statically encodable.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' field '{field_name}' in "
        "'@{function_name}': byte offset must be an exact non-negative static "
        "integer"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Propagate an exact non-negative byte-offset fact before AMDGPU "
        "target-low lowering"
    ),
)

# ERR_AMDGPU_003: AMDGPU target CPU override is unknown.
ERR_AMDGPU_003 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=3,
    severity=Severity.ERROR,
    summary="AMDGPU target CPU override is unknown.",
    message="AMDGPU target CPU override '{target_cpu}' is not a known processor",
    params=(ErrorParam("target_cpu", ParamKind.STRING),),
    fix_hint="Use a known AMDGPU processor name such as gfx1100 or gfx950",
)

# ERR_AMDGPU_004: AMDGPU target CPU override has no native descriptor set.
ERR_AMDGPU_004 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=4,
    severity=Severity.ERROR,
    summary="AMDGPU target CPU override has no native descriptor set.",
    message=(
        "AMDGPU target CPU override '{target_cpu}' is known but has no "
        "native target-low descriptor set"
    ),
    params=(ErrorParam("target_cpu", ParamKind.STRING),),
    fix_hint="Use an AMDGPU processor with native target-low descriptor coverage",
)

# ERR_AMDGPU_005: AMDGPU target CPU override changes descriptor set.
ERR_AMDGPU_005 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=5,
    severity=Severity.ERROR,
    summary="AMDGPU target CPU override changes descriptor set.",
    message=(
        "AMDGPU target CPU override '{target_cpu}' selects descriptor set "
        "'{target_descriptor_set}' but target record '@{target_name}' uses "
        "descriptor set '{record_descriptor_set}'"
    ),
    params=(
        ErrorParam("target_cpu", ParamKind.STRING),
        ErrorParam("target_descriptor_set", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("record_descriptor_set", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a target record from the same AMDGPU descriptor-set family as "
        "the requested processor"
    ),
)

ALL_AMDGPU_ERRORS = (
    ERR_AMDGPU_001,
    ERR_AMDGPU_002,
    ERR_AMDGPU_003,
    ERR_AMDGPU_004,
    ERR_AMDGPU_005,
)
