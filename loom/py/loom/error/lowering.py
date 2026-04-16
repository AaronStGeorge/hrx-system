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

ALL_LOWERING_ERRORS: tuple[ErrorDef, ...] = (
    ERR_LOWERING_001,
    ERR_LOWERING_002,
)
