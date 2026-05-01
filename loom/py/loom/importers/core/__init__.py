# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core infrastructure shared by Loom foreign IR importers."""

from loom.diagnostics import (
    Diagnostic,
    DiagnosticEngine,
    DiagnosticSeverity,
    LoomDiagnosticError,
)
from loom.importers.core.kernel import (
    KernelArgumentSpec,
    KernelModuleShell,
    KernelModuleSpec,
    create_kernel_module,
    normalize_workgroup_size,
)
from loom.importers.core.names import (
    NameAllocator,
    sanitize_identifier,
    sanitize_symbol,
    source_name,
)
from loom.importers.core.options import ImportOptions, ImportResult
from loom.importers.core.printing import print_loom_module
from loom.importers.core.session import (
    ConversionRecord,
    ImportBodyReport,
    SourceImportSession,
)

__all__ = [
    "ConversionRecord",
    "Diagnostic",
    "DiagnosticEngine",
    "DiagnosticSeverity",
    "ImportBodyReport",
    "ImportOptions",
    "ImportResult",
    "KernelArgumentSpec",
    "KernelModuleShell",
    "KernelModuleSpec",
    "LoomDiagnosticError",
    "NameAllocator",
    "SourceImportSession",
    "create_kernel_module",
    "normalize_workgroup_size",
    "print_loom_module",
    "sanitize_identifier",
    "sanitize_symbol",
    "source_name",
]
