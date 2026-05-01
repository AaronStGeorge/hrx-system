# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostics shared by Python Loom APIs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.errors import ErrorDef

__all__ = [
    "Diagnostic",
    "DiagnosticEngine",
    "DiagnosticSeverity",
    "LoomDiagnosticError",
]


class DiagnosticSeverity(StrEnum):
    """Severity for Python diagnostics."""

    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A diagnostic attached to a Python API step."""

    severity: DiagnosticSeverity
    message: str
    source: str | None = None
    details: tuple[str, ...] = ()
    error_def: ErrorDef | None = None

    def __str__(self) -> str:
        prefix = self.severity.value
        if self.error_def is not None:
            prefix = f"{prefix} {self.error_def}"
        pieces = [f"{prefix}: {self.message}"]
        if self.source:
            pieces.append(f"  source: {self.source}")
        pieces.extend(f"  {detail}" for detail in self.details)
        return "\n".join(pieces)


class LoomDiagnosticError(RuntimeError):
    """Raised when diagnostics contain errors at a fail-loud API boundary."""

    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        message = "\n".join(str(diagnostic) for diagnostic in self.diagnostics)
        super().__init__(message or "Loom operation failed")


@dataclass(slots=True)
class DiagnosticEngine:
    """Collects diagnostics and owns fail-loud API boundaries."""

    _diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return tuple(self._diagnostics)

    @property
    def has_errors(self) -> bool:
        return any(
            diagnostic.severity == DiagnosticSeverity.ERROR
            for diagnostic in self._diagnostics
        )

    def note(
        self,
        message: str,
        *,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.NOTE,
            message,
            source=source,
            details=details,
            error_def=error_def,
        )

    def warning(
        self,
        message: str,
        *,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.WARNING,
            message,
            source=source,
            details=details,
            error_def=error_def,
        )

    def error(
        self,
        message: str,
        *,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.ERROR,
            message,
            source=source,
            details=details,
            error_def=error_def,
        )

    def unsupported(
        self,
        operation: str,
        reason: str,
        *,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.error(
            f"unsupported source operation: {reason}",
            source=operation,
            details=details,
            error_def=error_def,
        )

    def emit(
        self,
        severity: DiagnosticSeverity,
        message: str,
        *,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        diagnostic = Diagnostic(
            severity=severity,
            message=message,
            source=source,
            details=tuple(details),
            error_def=error_def,
        )
        self._diagnostics.append(diagnostic)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self._diagnostics.extend(diagnostics)

    def raise_if_errors(self) -> None:
        if self.has_errors:
            raise LoomDiagnosticError(self._diagnostics)
