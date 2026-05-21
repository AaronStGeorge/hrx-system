# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selection helpers shared by IREE Bazel packages."""

def iree_select(selector):
    """Returns a Bazel `select()` that has an explicit default branch.

    IREE BUILD files are also source material for non-Bazel generators. Requiring
    `//conditions:default` keeps generator fallbacks obvious at the callsite
    instead of depending on whatever configuration happened to be active during
    a Bazel analysis.

    Args:
      selector: Dictionary passed through to `select()`.
    """
    if "//conditions:default" not in selector:
        fail("iree_select requires a //conditions:default branch")
    return select(selector)
