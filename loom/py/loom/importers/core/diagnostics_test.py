# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.importers.core import DiagnosticEngine, LoomImportError


def test_diagnostic_engine_raises_only_for_errors() -> None:
    diagnostics = DiagnosticEngine()
    diagnostics.warning("watch this")

    diagnostics.raise_if_errors()

    diagnostics.error("failed", source="arith.weird")
    with pytest.raises(LoomImportError, match="arith.weird"):
        diagnostics.raise_if_errors()
