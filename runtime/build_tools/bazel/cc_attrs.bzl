# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ attribute helpers."""

_RUNTIME_DEFINES_DEP = Label("//runtime/src:defines")

def _with_runtime_deps(deps):
    if deps == None:
        deps = []
    return deps + [_RUNTIME_DEFINES_DEP]

runtime_cc_attrs = struct(
    with_runtime_deps = _with_runtime_deps,
)
