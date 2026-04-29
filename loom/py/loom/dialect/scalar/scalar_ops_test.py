# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.dialect.scalar import (
    ALL_SCALAR_OPS,
    SCALAR_OP_CATEGORIES,
    SCALAR_OP_CATEGORY_GROUPS,
    scalar_ops,
)


def test_scalar_op_category_groups_cover_registry_once() -> None:
    grouped_ops = [op for _, category_ops in SCALAR_OP_CATEGORY_GROUPS for op in category_ops]
    grouped_names = [op.name for op in grouped_ops]

    assert tuple(grouped_ops) == ALL_SCALAR_OPS
    assert len(grouped_names) == len(set(grouped_names))
    assert tuple(category for category, _ in SCALAR_OP_CATEGORY_GROUPS) == SCALAR_OP_CATEGORIES
    assert scalar_ops.categories == SCALAR_OP_CATEGORIES
