# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ordered source closures for TileLang kernel fixtures."""

UNIT_SOURCES = [
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:__init__.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:engram.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_expand.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_gate.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_histogram.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_normalize.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_reduce.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:transpose.py",
]

IMPORT_SOURCES = [
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:engram.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:mhc.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_expand.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_gate.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_histogram.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_normalize.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_reduce.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:moe_routing.py",
    "//loom/py/loom/importers/tilelang/testdata/tilekernels:transpose.py",
]
