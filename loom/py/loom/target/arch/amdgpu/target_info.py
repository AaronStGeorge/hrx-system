# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU processor and descriptor-set facts for native target emission."""

from __future__ import annotations

from dataclasses import dataclass

AMDGPU_AMDHSA_TARGET_TRIPLE = "amdgcn-amd-amdhsa"

AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE = "none"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11 = "gfx11"

AMDGPU_SOPP_S_ENDPGM_GFX9_GFX10_GFX13_OPCODE = 0x001
AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE = 0x030


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorSetInfo:
    key: str
    low_preset_key: str
    s_endpgm_opcode: int
    supports_descriptor_packet_encoding: bool


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorInfo:
    target_cpu: str
    descriptor_set_key: str
    low_preset_key: str
    elf_machine_flags: int
    elf_feature_flags: int
    default_wavefront_size: int
    kernel_descriptor_profile: str
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
        kernel_descriptor_vgpr_encoding_granule_wave32=8,
        kernel_descriptor_vgpr_encoding_granule_wave64=4,
        kernel_descriptor_has_architected_flat_scratch=True,
        kernel_descriptor_uses_gfx10_sgpr_encoding=True,
        kernel_descriptor_has_dx10_clamp_and_ieee_mode=True,
    )


AMDGPU_DESCRIPTOR_SET_INFOS: tuple[AmdgpuDescriptorSetInfo, ...] = (
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx1250.core",
        low_preset_key="amdgpu-gfx1250",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE,
        supports_descriptor_packet_encoding=False,
    ),
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx11.core",
        low_preset_key="amdgpu-gfx11",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE,
        supports_descriptor_packet_encoding=True,
    ),
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx12.core",
        low_preset_key="amdgpu-gfx12",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX11_GFX12_OPCODE,
        supports_descriptor_packet_encoding=False,
    ),
    AmdgpuDescriptorSetInfo(
        key="amdgpu.gfx950.core",
        low_preset_key="amdgpu-gfx950",
        s_endpgm_opcode=AMDGPU_SOPP_S_ENDPGM_GFX9_GFX10_GFX13_OPCODE,
        supports_descriptor_packet_encoding=False,
    ),
)


AMDGPU_PROCESSOR_INFOS: tuple[AmdgpuProcessorInfo, ...] = (
    gfx11_processor_info(target_cpu="gfx1100", elf_machine_flags=0x041),
    gfx11_processor_info(target_cpu="gfx1101", elf_machine_flags=0x046),
    gfx11_processor_info(target_cpu="gfx1102", elf_machine_flags=0x047),
    gfx11_processor_info(target_cpu="gfx1103", elf_machine_flags=0x044),
    gfx11_processor_info(target_cpu="gfx1150", elf_machine_flags=0x043),
    gfx11_processor_info(target_cpu="gfx1151", elf_machine_flags=0x04A),
    gfx11_processor_info(target_cpu="gfx1152", elf_machine_flags=0x055),
    gfx11_processor_info(target_cpu="gfx1153", elf_machine_flags=0x058),
    gfx11_processor_info(target_cpu="gfx1170", elf_machine_flags=0x05D),
    AmdgpuProcessorInfo(
        target_cpu="gfx1200",
        descriptor_set_key="amdgpu.gfx12.core",
        low_preset_key="amdgpu-gfx12",
        elf_machine_flags=0x048,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx1201",
        descriptor_set_key="amdgpu.gfx12.core",
        low_preset_key="amdgpu-gfx12",
        elf_machine_flags=0x04E,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx1250",
        descriptor_set_key="amdgpu.gfx1250.core",
        low_preset_key="amdgpu-gfx1250",
        elf_machine_flags=0x049,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx1251",
        descriptor_set_key="amdgpu.gfx1250.core",
        low_preset_key="amdgpu-gfx1250",
        elf_machine_flags=0x05A,
        elf_feature_flags=0,
        default_wavefront_size=32,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    ),
    AmdgpuProcessorInfo(
        target_cpu="gfx950",
        descriptor_set_key="amdgpu.gfx950.core",
        low_preset_key="amdgpu-gfx950",
        elf_machine_flags=0x04F,
        elf_feature_flags=0,
        default_wavefront_size=64,
        kernel_descriptor_profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    ),
)


def sorted_descriptor_set_infos() -> tuple[AmdgpuDescriptorSetInfo, ...]:
    return tuple(sorted(AMDGPU_DESCRIPTOR_SET_INFOS, key=lambda info: info.key))


def sorted_processor_infos() -> tuple[AmdgpuProcessorInfo, ...]:
    return tuple(sorted(AMDGPU_PROCESSOR_INFOS, key=lambda info: info.target_cpu))
