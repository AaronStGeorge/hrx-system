# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU processor and descriptor-set row data for native target emission.

The public C representation of these facts lives in
`loom/src/loom/target/arch/amdgpu/target_info_defs.h`. This module owns the
Python input rows consumed by the table generator, not emitted C ABI shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

AMDGPU_AMDHSA_TARGET_TRIPLE = "amdgcn-amd-amdhsa"

AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE = "none"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11 = "gfx11"

AMDGPU_MATRIX_FEATURE_PROFILE_NONE = "none"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908 = "mfma_gfx908"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A = "mfma_gfx90a"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940 = "mfma_gfx940"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950 = "mfma_gfx950"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11 = "wmma_gfx11"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12 = "wmma_gfx12"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250 = "wmma_gfx1250"

AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE = "none"
AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT = "stride14_enable_bit"

AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE = "none"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC = "gfx9_11_glc_slc_dlc"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH = "gfx12_nv_scope_th"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1 = "gfx950_nt_sc0_sc1"

AMDGPU_SOPP_S_ENDPGM_GFX9_GFX10_GFX13_OPCODE = 0x001
AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE = 0x030


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorSetInfo:
    key: str
    low_preset_key: str
    s_endpgm_opcode: int
    supports_descriptor_packet_encoding: bool
    buffer_resource_cache_swizzle: str = AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE
    vector_memory_cache_policy_encoding: str = (
        AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE
    )


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorInfo:
    target_cpu: str
    descriptor_set_key: str
    low_preset_key: str
    elf_machine_flags: int
    elf_feature_flags: int
    default_wavefront_size: int
    kernel_descriptor_profile: str
    matrix_feature_profile: str = AMDGPU_MATRIX_FEATURE_PROFILE_NONE
    kernel_descriptor_vgpr_encoding_granule_wave32: int = 0
    kernel_descriptor_vgpr_encoding_granule_wave64: int = 0
    kernel_descriptor_has_architected_flat_scratch: bool = False
    kernel_descriptor_uses_gfx10_sgpr_encoding: bool = False
    kernel_descriptor_has_dx10_clamp_and_ieee_mode: bool = False


def gfx11_processor_info(
    target_cpu: str, elf_machine_flags: int
) -> AmdgpuProcessorInfo:
    return AmdgpuProcessorInfo(
        target_cpu=target_cpu,
        descriptor_set_key="amdgpu.gfx11.core",
        low_preset_key="amdgpu-gfx11",
        elf_machine_flags=elf_machine_flags,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
        kernel_descriptor_vgpr_encoding_granule_wave32=8,
        kernel_descriptor_vgpr_encoding_granule_wave64=4,
        kernel_descriptor_has_architected_flat_scratch=True,
        kernel_descriptor_uses_gfx10_sgpr_encoding=True,
        kernel_descriptor_has_dx10_clamp_and_ieee_mode=True,
    )


def gfx1170_processor_info() -> AmdgpuProcessorInfo:
    return AmdgpuProcessorInfo(
        target_cpu="gfx1170",
        descriptor_set_key="",
        low_preset_key="",
        elf_machine_flags=0x05D,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    )


AMDGPU_DESCRIPTOR_SET_INFOS: tuple[AmdgpuDescriptorSetInfo, ...] = (
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx1250.core",
        low_preset_key="amdgpu-gfx1250",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE,
        supports_descriptor_packet_encoding=True,
        vector_memory_cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
    ),
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx11.core",
        low_preset_key="amdgpu-gfx11",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE,
        supports_descriptor_packet_encoding=True,
        vector_memory_cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
    ),
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx12.core",
        low_preset_key="amdgpu-gfx12",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE,
        supports_descriptor_packet_encoding=True,
        vector_memory_cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
    ),
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx950.core",
        low_preset_key="amdgpu-gfx950",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX9_GFX10_GFX13_OPCODE,
        supports_descriptor_packet_encoding=True,
        buffer_resource_cache_swizzle=AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
        vector_memory_cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
    ),
)


AMDGPU_PROCESSOR_INFOS: tuple[AmdgpuProcessorInfo, ...] = (
    AmdgpuProcessorInfo(
        target_cpu="gfx908",
        descriptor_set_key="",
        low_preset_key="",
        elf_machine_flags=0x030,
        elf_feature_flags=0,
        default_wavefront_size=64,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx90a",
        descriptor_set_key="",
        low_preset_key="",
        elf_machine_flags=0x03F,
        elf_feature_flags=0,
        default_wavefront_size=64,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx940",
        descriptor_set_key="",
        low_preset_key="",
        elf_machine_flags=0,
        elf_feature_flags=0,
        default_wavefront_size=64,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx941",
        descriptor_set_key="",
        low_preset_key="",
        elf_machine_flags=0,
        elf_feature_flags=0,
        default_wavefront_size=64,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx942",
        descriptor_set_key="",
        low_preset_key="",
        elf_machine_flags=0x04C,
        elf_feature_flags=0,
        default_wavefront_size=64,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    ),
    gfx11_processor_info(target_cpu="gfx1100", elf_machine_flags=0x041),
    gfx11_processor_info(target_cpu="gfx1101", elf_machine_flags=0x046),
    gfx11_processor_info(target_cpu="gfx1102", elf_machine_flags=0x047),
    gfx11_processor_info(target_cpu="gfx1103", elf_machine_flags=0x044),
    gfx11_processor_info(target_cpu="gfx1150", elf_machine_flags=0x043),
    gfx11_processor_info(target_cpu="gfx1151", elf_machine_flags=0x04A),
    gfx11_processor_info(target_cpu="gfx1152", elf_machine_flags=0x055),
    gfx11_processor_info(target_cpu="gfx1153", elf_machine_flags=0x058),
    gfx1170_processor_info(),
    AmdgpuProcessorInfo(
        target_cpu="gfx1200",
        descriptor_set_key="amdgpu.gfx12.core",
        low_preset_key="amdgpu-gfx12",
        elf_machine_flags=0x048,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx1201",
        descriptor_set_key="amdgpu.gfx12.core",
        low_preset_key="amdgpu-gfx12",
        elf_machine_flags=0x04E,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx1250",
        descriptor_set_key="amdgpu.gfx1250.core",
        low_preset_key="amdgpu-gfx1250",
        elf_machine_flags=0x049,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx1251",
        descriptor_set_key="amdgpu.gfx1250.core",
        low_preset_key="amdgpu-gfx1250",
        elf_machine_flags=0x05A,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx1252",
        descriptor_set_key="amdgpu.gfx1250.core",
        low_preset_key="amdgpu-gfx1250",
        elf_machine_flags=0,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx950",
        descriptor_set_key="amdgpu.gfx950.core",
        low_preset_key="amdgpu-gfx950",
        elf_machine_flags=0x04F,
        elf_feature_flags=0,
        default_wavefront_size=64,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    ),
)


def sorted_descriptor_set_infos() -> tuple[AmdgpuDescriptorSetInfo, ...]:
    return tuple(sorted(AMDGPU_DESCRIPTOR_SET_INFOS, key=lambda info: info.key))


def sorted_processor_infos() -> tuple[AmdgpuProcessorInfo, ...]:
    return tuple(sorted(AMDGPU_PROCESSOR_INFOS, key=lambda info: info.target_cpu))
