# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P floating-point dot-product contracts."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.floating_dot import (
    AIE2P_FLOATING_DOT_RULES,
)
from loom.target.contracts import Guard, Vector


def test_bf16_pair_dot_rules_cover_packed_and_interleaved_representations() -> None:
    assert len(AIE2P_FLOATING_DOT_RULES) == 16
    packed_rules = [rule for rule in AIE2P_FLOATING_DOT_RULES if not rule.source_nodes]
    interleaved_rules = [rule for rule in AIE2P_FLOATING_DOT_RULES if rule.source_nodes]
    assert len(packed_rules) == 8
    assert len(interleaved_rules) == 8
    assert all(rule.source_op is vector.vector_dot2f for rule in packed_rules)
    assert all(
        Guard.value_type(
            "lhs",
            Vector("bf16", minimum_static_elements=2, maximum_static_elements=32),
        )
        in rule.guards
        for rule in packed_rules
    )
