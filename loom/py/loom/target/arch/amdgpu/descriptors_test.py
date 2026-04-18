# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amdgpu.descriptors import (
    _GFX11_CORE_OVERLAY_DESCRIPTORS,
    _GFX11_ISA_SNAPSHOT,
    AMDGPU_GFX11_CORE_DESCRIPTOR_SET,
)


def test_gfx11_core_descriptors_are_materialized_from_snapshot() -> None:
    descriptor_by_key = {
        descriptor.key: descriptor for descriptor in _GFX11_CORE_OVERLAY_DESCRIPTORS
    }

    assert list(descriptor_by_key) == [
        "amdgpu.s_add_u32",
        "amdgpu.v_add_u32",
        "amdgpu.s_buffer_load_dword",
        "amdgpu.buffer_load_dword",
        "amdgpu.buffer_store_dword",
        "amdgpu.v_wmma_f32_16x16x16_f16",
        "amdgpu.s_waitcnt",
        "amdgpu.s_waitcnt_depctr",
        "amdgpu.s_wait_idle",
    ]
    assert descriptor_by_key["amdgpu.v_add_u32"].encoding_id == 37
    assert descriptor_by_key["amdgpu.s_buffer_load_dword"].encoding_id == 8
    assert descriptor_by_key["amdgpu.buffer_load_dword"].encoding_id == 20
    assert descriptor_by_key["amdgpu.buffer_store_dword"].encoding_id == 26
    assert descriptor_by_key["amdgpu.v_wmma_f32_16x16x16_f16"].encoding_id == 64
    assert descriptor_by_key["amdgpu.s_waitcnt"].encoding_id == 9
    assert descriptor_by_key["amdgpu.s_waitcnt_depctr"].encoding_id == 8
    assert descriptor_by_key["amdgpu.s_wait_idle"].encoding_id == 10

    assert AMDGPU_GFX11_CORE_DESCRIPTOR_SET.descriptors[0].key == "amdgpu.s_mov_b32"
    assert (
        AMDGPU_GFX11_CORE_DESCRIPTOR_SET.descriptors[1:]
        == _GFX11_CORE_OVERLAY_DESCRIPTORS
    )


def test_gfx11_snapshot_covers_only_overlay_backed_source_facts() -> None:
    instruction_names = tuple(
        instruction.name for instruction in _GFX11_ISA_SNAPSHOT.instructions
    )

    assert instruction_names == (
        "BUFFER_LOAD_B32",
        "BUFFER_STORE_B32",
        "S_ADD_U32",
        "S_BUFFER_LOAD_B32",
        "S_WAITCNT",
        "S_WAITCNT_DEPCTR",
        "S_WAIT_IDLE",
        "V_ADD_NC_U32",
        "V_WMMA_F32_16X16X16_F16",
    )
    assert "S_MOV_B32" not in _GFX11_ISA_SNAPSHOT.instruction_map()
    assert (
        _GFX11_ISA_SNAPSHOT.instruction_map(include_aliases=True)[
            "BUFFER_LOAD_DWORD"
        ].name
        == "BUFFER_LOAD_B32"
    )
