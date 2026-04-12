# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""View dialect: logical view operations."""

from loom.dialect.view.defs import (
    ALL_VIEW_OPS,
    view_ops,
    view_subview,
)

__all__ = [
    "view_ops",
    "ALL_VIEW_OPS",
    "view_subview",
]
