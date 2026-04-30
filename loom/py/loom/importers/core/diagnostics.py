# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostics for whole-module import attempts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum


class DiagnosticSeverity(StrEnum):
    """Severity for importer diagnostics."""

    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A diagnostic attached to a source importer step."""

    severity: DiagnosticSeverity
    message: str
    source: str | None = None
    details: tuple[str, ...] = ()

    def __str__(self) -> str:
        pieces = [f"{self.severity.value}: {self.message}"]
        if self.source:
            pieces.append(f"  source: {self.source}")
        pieces.extend(f"  {detail}" for detail in self.details)
        return "\n".join(pieces)


class LoomImportError(RuntimeError):
    """Raised when an importer cannot produce a complete Loom module."""

    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        message = "\n".join(str(diagnostic) for diagnostic in self.diagnostics)
        super().__init__(message or "Loom import failed")


@dataclass(slots=True)
class DiagnosticEngine:
    """Collects diagnostics and owns the fail-loud import boundary."""

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
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.NOTE, message, source=source, details=details
        )

    def warning(
        self,
        message: str,
        *,
        source: str | None = None,
        details: Iterable[str] = (),
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.WARNING,
            message,
            source=source,
            details=details,
        )

    def error(
        self,
        message: str,
        *,
        source: str | None = None,
        details: Iterable[str] = (),
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.ERROR,
            message,
            source=source,
            details=details,
        )

    def unsupported(
        self,
        operation: str,
        reason: str,
        *,
        details: Iterable[str] = (),
    ) -> Diagnostic:
        return self.error(
            f"unsupported source operation: {reason}",
            source=operation,
            details=details,
        )

    def emit(
        self,
        severity: DiagnosticSeverity,
        message: str,
        *,
        source: str | None = None,
        details: Iterable[str] = (),
    ) -> Diagnostic:
        diagnostic = Diagnostic(
            severity=severity,
            message=message,
            source=source,
            details=tuple(details),
        )
        self._diagnostics.append(diagnostic)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self._diagnostics.extend(diagnostics)

    def raise_if_errors(self) -> None:
        if self.has_errors:
            raise LoomImportError(self._diagnostics)
