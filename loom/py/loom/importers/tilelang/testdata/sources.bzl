# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ordered source closures for TileLang importer fixtures."""

load("//loom/py/loom/importers/tilelang/testdata/tilekernels:sources.bzl", TILEKERNEL_IMPORT_SOURCES = "IMPORT_SOURCES", TILEKERNEL_UNIT_SOURCES = "UNIT_SOURCES")

UNIT_SOURCES = [
    "//loom/py/loom/importers/tilelang/testdata:__init__.py",
] + TILEKERNEL_UNIT_SOURCES

# Import cases run in declaration order: scalar cases, kernel cases, then tile
# and vector cases.
IMPORT_SOURCES = [
    "//loom/py/loom/importers/tilelang/testdata:allocations.py",
    "//loom/py/loom/importers/tilelang/testdata:analysis.py",
    "//loom/py/loom/importers/tilelang/testdata:control_flow.py",
    "//loom/py/loom/importers/tilelang/testdata:diagnostics.py",
    "//loom/py/loom/importers/tilelang/testdata:fp8.py",
    "//loom/py/loom/importers/tilelang/testdata:lowered_metadata.py",
    "//loom/py/loom/importers/tilelang/testdata:memory_effects.py",
    "//loom/py/loom/importers/tilelang/testdata:scalar_calls.py",
] + TILEKERNEL_IMPORT_SOURCES + [
    "//loom/py/loom/importers/tilelang/testdata:tileops.py",
    "//loom/py/loom/importers/tilelang/testdata:topology.py",
    "//loom/py/loom/importers/tilelang/testdata:vector_arithmetic.py",
    "//loom/py/loom/importers/tilelang/testdata:vector_casts.py",
    "//loom/py/loom/importers/tilelang/testdata:vector_memory.py",
]
