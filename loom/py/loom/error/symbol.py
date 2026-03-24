# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SYMBOL domain — unresolved references."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_SYMBOL_001: Symbol reference out of range.
ERR_SYMBOL_001 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=1,
    severity=Severity.ERROR,
    summary="Symbol reference out of range.",
    message="symbol reference index {symbol_index} is out of range (max {max_index})",
    params=(
        ErrorParam("symbol_index", ParamKind.U32),
        ErrorParam("max_index", ParamKind.U32),
    ),
)

ALL_SYMBOL_ERRORS: tuple[ErrorDef, ...] = (ERR_SYMBOL_001,)
