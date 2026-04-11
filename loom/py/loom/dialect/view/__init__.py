# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""View dialect: layout construction and logical subviews."""

from loom.dialect.view.defs import (
    ALL_VIEW_OPS,
    view_layout_dense,
    view_layout_strided,
    view_ops,
    view_subview,
)

__all__ = [
    "view_ops",
    "ALL_VIEW_OPS",
    "view_layout_dense",
    "view_layout_strided",
    "view_subview",
]
