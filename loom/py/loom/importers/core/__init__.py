# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core infrastructure shared by Loom foreign IR importers."""

from loom.importers.core.diagnostics import (
    Diagnostic,
    DiagnosticEngine,
    DiagnosticSeverity,
    LoomImportError,
)
from loom.importers.core.names import NameAllocator, sanitize_identifier, source_name
from loom.importers.core.options import ImportOptions, ImportResult
from loom.importers.core.session import (
    ConversionRecord,
    ImportBodyReport,
    SourceImportSession,
)
from loom.importers.core.verify import StructuralVerifier

__all__ = [
    "ConversionRecord",
    "Diagnostic",
    "DiagnosticEngine",
    "DiagnosticSeverity",
    "ImportBodyReport",
    "ImportOptions",
    "ImportResult",
    "LoomImportError",
    "NameAllocator",
    "SourceImportSession",
    "StructuralVerifier",
    "sanitize_identifier",
    "source_name",
]
