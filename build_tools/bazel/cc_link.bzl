# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C/C++ platform link options for shared IREE support libraries."""

def _pthreads_linkopts():
    """Returns link options for targets that use POSIX pthread APIs."""
    return select({
        "@platforms//os:android": [],
        "@platforms//os:emscripten": [],
        "@platforms//os:wasi": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["-pthread"],
    })

def _dl_linkopts():
    """Returns link options for targets that use dlopen-style APIs."""
    return select({
        "@platforms//os:android": ["-ldl"],
        "@platforms//os:linux": ["-ldl"],
        "//conditions:default": [],
    })

def _rt_linkopts():
    """Returns link options for targets that use Linux librt APIs."""
    return select({
        "@platforms//os:linux": ["-lrt"],
        "//conditions:default": [],
    })

cc_link = struct(
    dl_linkopts = _dl_linkopts,
    pthreads_linkopts = _pthreads_linkopts,
    rt_linkopts = _rt_linkopts,
)
