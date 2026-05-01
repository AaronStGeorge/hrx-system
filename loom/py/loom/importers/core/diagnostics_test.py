# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.core import DiagnosticEngine, LoomImportError


def test_diagnostic_engine_raises_only_for_errors() -> None:
    diagnostics = DiagnosticEngine()
    diagnostics.warning("watch this")

    diagnostics.raise_if_errors()

    diagnostics.error("failed", source="arith.weird")
    error_message = None
    try:
        diagnostics.raise_if_errors()
    except LoomImportError as exc:
        error_message = str(exc)
    if error_message is None:
        raise AssertionError("expected diagnostic errors to raise LoomImportError")
    assert "arith.weird" in error_message
