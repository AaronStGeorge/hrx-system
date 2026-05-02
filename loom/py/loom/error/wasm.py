# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""WASM domain — WebAssembly-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef

ALL_WASM_ERRORS: tuple[ErrorDef, ...] = ()
