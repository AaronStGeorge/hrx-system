# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Data model for TileLang importer results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from loom.importers.core import ImportBodyReport
from loom.ir import Type


@dataclass(frozen=True, slots=True)
class TileLangBinding:
    """One TileLang kernel parameter imported as a Loom kernel argument."""

    ordinal: int
    name: str
    source: object
    type: Type
    buffer: object | None = None
    aliases: tuple[object, ...] = ()
    noalias: bool = False


@dataclass(frozen=True, slots=True)
class TileLangKernelFacts:
    """Facts extracted from one TileLang kernel source object."""

    function_name: str
    target_preset: str
    workgroup_size: tuple[int, int, int]
    bindings: tuple[TileLangBinding, ...]
    operation_counts: Counter[str]
    converted_body: ImportBodyReport
