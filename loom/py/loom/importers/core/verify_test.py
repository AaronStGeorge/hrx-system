# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.core import DiagnosticEngine, StructuralVerifier
from loom.ir import Module, Operation, Symbol


def test_structural_verifier_reports_missing_operand_value() -> None:
    module = Module()
    module.symbols.append(
        Symbol(name="broken", op=Operation(name="test.op", operands=[7]))
    )
    diagnostics = DiagnosticEngine()

    StructuralVerifier(module, diagnostics).verify()

    assert diagnostics.has_errors
    assert "outside [0, 0)" in str(diagnostics.diagnostics[0])


def test_structural_verifier_reports_duplicate_symbols() -> None:
    module = Module()
    module.symbols.append(Symbol(name="same", op=Operation(name="test.op")))
    module.symbols.append(Symbol(name="same", op=Operation(name="test.op")))
    diagnostics = DiagnosticEngine()

    StructuralVerifier(module, diagnostics).verify()

    assert diagnostics.has_errors
    assert "duplicate symbol name" in str(diagnostics.diagnostics[0])
