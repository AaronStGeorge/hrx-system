# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 packed-dot feature descriptor rows."""

from __future__ import annotations

from pathlib import Path

from loom.target.arch.x86.packed_dot_data import X86_PACKED_DOT_DESCRIPTORS
from loom.target.low_descriptors import (
    DescriptorSet,
    IssueUse,
    LatencyKind,
    ModelQuality,
    RegClass,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    SpillSlotSpace,
)

from .common import (
    _REG_XMM,
    _REG_YMM,
    _REG_ZMM,
    _RESOURCE_DOT,
    _SCHEDULE_VECTOR_DOT_XMM,
    _SCHEDULE_VECTOR_DOT_YMM,
    _SCHEDULE_VECTOR_DOT_ZMM,
    _packed_dot_descriptor,
    _vector_lane_units,
)

X86_PACKED_DOT_DESCRIPTOR_SET = DescriptorSet(
    key="x86.packed_dot.core",
    target_key="x86",
    feature_key="x86.packed_dot.v1",
    c_header_path=Path("loom/src/loom/target/arch/x86/packed_dot_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/x86/packed_dot_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_PACKED_DOT_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/packed_dot_descriptors.h",
    function_name="loom_x86_packed_dot_core_descriptor_set",
    c_table_prefix="X86PackedDotCore",
    c_enum_prefix="X86_PACKED_DOT_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_XMM,
            128,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
            alias_set_id=2,
        ),
        RegClass(
            _REG_YMM,
            256,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
            alias_set_id=2,
        ),
        RegClass(
            _REG_ZMM,
            512,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
            alias_set_id=2,
        ),
    ),
    resources=(
        Resource(
            _RESOURCE_DOT,
            capacity_per_cycle=4,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=1,
        ),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_YMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(256)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
    ),
    descriptors=tuple(
        _packed_dot_descriptor(descriptor) for descriptor in X86_PACKED_DOT_DESCRIPTORS
    ),
)
