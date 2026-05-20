# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU core descriptor-set assembly."""

from __future__ import annotations

from .alu import *
from .atomic import *
from .common import *
from .control import *
from .matrix import *
from .memory import *
from .workgroup import *

_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES = (
    "DWORD",
    "DWORDX2",
    "DWORDX3",
    "DWORDX4",
)
_CDNA_DWORD_MEMORY_MNEMONIC_SUFFIXES = (
    "dword",
    "dwordx2",
    "dwordx3",
    "dwordx4",
)
_BYTE_MEMORY_INSTRUCTION_SUFFIXES = ("B32", "B64", "B96", "B128")
_BYTE_MEMORY_MNEMONIC_SUFFIXES = ("b32", "b64", "b96", "b128")


def _s_load_dword_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    include_b96: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_load_dword_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_load_dwordx2_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        *(
            (
                _s_load_96_overlay(
                    offset_field_name,
                    offset_bit_width=offset_bit_width,
                    fixed_encoding_fields=fixed_encoding_fields,
                ),
            )
            if include_b96
            else ()
        ),
        _s_load_dwordx4_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_load_dwordx8_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_load_dwordx16_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _s_load_dword_offset_only_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue,
    include_b96: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_load_dword_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            descriptor_key="amdgpu.s_load_dword_offset_only",
            fixed_soffset=fixed_soffset,
        ),
        _s_load_dwordx2_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            descriptor_key="amdgpu.s_load_dwordx2_offset_only",
            fixed_soffset=fixed_soffset,
        ),
        *(
            (
                _s_load_96_overlay(
                    offset_field_name,
                    offset_bit_width=offset_bit_width,
                    fixed_encoding_fields=fixed_encoding_fields,
                    descriptor_key="amdgpu.s_load_b96_offset_only",
                    fixed_soffset=fixed_soffset,
                ),
            )
            if include_b96
            else ()
        ),
        _s_load_dwordx4_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            descriptor_key="amdgpu.s_load_dwordx4_offset_only",
            fixed_soffset=fixed_soffset,
        ),
        _s_load_dwordx8_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            descriptor_key="amdgpu.s_load_dwordx8_offset_only",
            fixed_soffset=fixed_soffset,
        ),
        _s_load_dwordx16_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            descriptor_key="amdgpu.s_load_dwordx16_offset_only",
            fixed_soffset=fixed_soffset,
        ),
    )


def _s_buffer_load_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    include_b96: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_buffer_load_dword_overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_buffer_load_64_overlay(
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        *(
            (
                _s_buffer_load_96_overlay(
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    fixed_encoding_fields=fixed_encoding_fields,
                ),
            )
            if include_b96
            else ()
        ),
        _s_buffer_load_128_overlay(
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_buffer_load_256_overlay(
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_buffer_load_512_overlay(
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _cdna_s_buffer_load_width_overlays(
    *,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_buffer_load_dword_overlay(fixed_encoding_fields=fixed_encoding_fields),
        _s_buffer_load_64_overlay(
            descriptor_key="amdgpu.s_buffer_load_dwordx2",
            instruction_name="S_BUFFER_LOAD_DWORDX2",
            mnemonic="s_buffer_load_dwordx2",
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_buffer_load_128_overlay(
            descriptor_key="amdgpu.s_buffer_load_dwordx4",
            instruction_name="S_BUFFER_LOAD_DWORDX4",
            mnemonic="s_buffer_load_dwordx4",
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_buffer_load_256_overlay(
            descriptor_key="amdgpu.s_buffer_load_dwordx8",
            instruction_name="S_BUFFER_LOAD_DWORDX8",
            mnemonic="s_buffer_load_dwordx8",
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _s_buffer_load_512_overlay(
            descriptor_key="amdgpu.s_buffer_load_dwordx16",
            instruction_name="S_BUFFER_LOAD_DWORDX16",
            mnemonic="s_buffer_load_dwordx16",
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _cdna_core_overlays(
    *,
    global_load_lds_variants: tuple[tuple[str, str, str, int, int], ...],
    buffer_load_lds_variants: tuple[
        tuple[str, str, str, int, int, AmdgpuImplicitOperandOverlay], ...
    ],
    include_v_dot2_f32_bf16: bool,
    include_v_cvt_pk_bf16_f32: bool,
    include_ds_transpose_reads: bool,
    matrix_overlays: tuple[AmdgpuDescriptorOverlay, ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("default"),
        _v_add_u32_overlay("V_ADD_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_U32"),
        _v_add_u32_literal_overlay("V_ADD_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(
            instruction_name="V_ADDC_CO_U32", mnemonic="v_addc_co_u32"
        ),
        _v_sub_u32_overlay("V_SUB_U32", "v_sub_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp_legacy_overlay(),
        _v_mov_b32_sdwa_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_src0_inline_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(include_literal_forms=False),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_add_f32_src0_inline_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_sub_f32_src0_inline_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_mul_f32_src0_inline_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_min_f32_src0_inline_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_max_f32_src0_inline_overlay(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        *((_v_cvt_pk_bf16_f32_overlay(),) if include_v_cvt_pk_bf16_f32 else ()),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(include_literal_forms=False),
        *_s_load_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_scratch_load_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_load_dword_offset_only_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_s_scratch_load_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_cdna_s_buffer_load_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_scratch_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_s_scratch_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_s_buffer_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx2_off_zero",
        ),
        _buffer_load_64_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx3",
            instruction_name="BUFFER_LOAD_DWORDX3",
            mnemonic="buffer_load_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx3_off_zero",
        ),
        _buffer_load_96_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx3_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX3",
            mnemonic="buffer_load_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx4_off_zero",
        ),
        _buffer_load_128_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx2_off_zero",
        ),
        _buffer_store_64_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2_off_zero",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx3",
            instruction_name="BUFFER_STORE_DWORDX3",
            mnemonic="buffer_store_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx3_off_zero",
        ),
        _buffer_store_96_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx3_off_zero",
            instruction_name="BUFFER_STORE_DWORDX3",
            mnemonic="buffer_store_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx4_off_zero",
        ),
        _buffer_store_128_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4_off_zero",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_byte_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_load_lds_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=buffer_load_lds_variants,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_CDNA_DWORD_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_CDNA_DWORD_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=None,
            fixed_saddr=_SCRATCH_CDNA_SADDR_OFF,
            implicit_flat_scratch=True,
            implicit_m0=True,
            descriptor_key_suffix="_vaddr",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=0,
            fixed_saddr=_SCRATCH_CDNA_SADDR_OFF,
            implicit_flat_scratch=True,
            implicit_m0=True,
            descriptor_key_suffix="_offset_only",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_load_lds_overlays(
            address_units=2,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=global_load_lds_variants,
        ),
        *_global_load_lds_overlays(
            address_units=1,
            saddr_off=None,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=global_load_lds_variants,
        ),
        *_global_atomic_overlays(
            rows=_GLOBAL_ATOMIC_GFX940_ROWS,
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            return_field_name="SC0",
            return_field_value=1,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX950_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP",
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            offset_signed=False,
            return_field_name="SC0",
            return_field_value=1,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            implicit_flat_scratch=True,
            implicit_m0=True,
            allow_accumulator_operands=True,
        ),
        *_ds_memory_overlays(include_packed_half_atomic_add=True),
        *_ds_crosslane_overlays(),
        _v_dot2_f32_f16_overlay(),
        *((_v_dot2_f32_bf16_overlay(),) if include_v_dot2_f32_bf16 else ()),
        _v_dot4_i32_i8_overlay(signedness_modifiers=False),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=False),
        _v_dot8_u32_u4_overlay(),
        *(_gfx950_ds_transpose_read_overlays() if include_ds_transpose_reads else ()),
        *matrix_overlays,
        _s_barrier_overlay(),
        *_gfx950_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _VMEM_STORE_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            ),
            lgkmcnt_immediate=_LGKMCNT_4BIT_IMMEDIATE,
        ),
    )


def _gfx940_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        global_load_lds_variants=_GLOBAL_LOAD_LDS_CDNA3_VARIANTS,
        buffer_load_lds_variants=_BUFFER_LOAD_LDS_CDNA3_VARIANTS,
        include_v_dot2_f32_bf16=False,
        include_v_cvt_pk_bf16_f32=False,
        include_ds_transpose_reads=False,
        matrix_overlays=(*_cdna3_mfma_overlays(), *_cdna3_smfmac_overlays()),
    )


def _gfx950_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        global_load_lds_variants=_GLOBAL_LOAD_LDS_GFX950_VARIANTS,
        buffer_load_lds_variants=_BUFFER_LOAD_LDS_GFX950_VARIANTS,
        include_v_dot2_f32_bf16=True,
        include_v_cvt_pk_bf16_f32=True,
        include_ds_transpose_reads=True,
        matrix_overlays=(*_cdna4_mfma_overlays(), *_cdna4_smfmac_overlays()),
    )


def _gfx940_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx940_core_overlays())
    )


def _gfx950_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx950_core_overlays())
    )


def _gfx11_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp16_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_src0_inline_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_add_f32_src0_inline_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_sub_f32_src0_inline_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_mul_f32_src0_inline_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_min_f32_src0_inline_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_max_f32_src0_inline_overlay(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        *_s_load_dword_width_overlays(),
        *_s_load_dword_offset_only_overlays(fixed_soffset=_predefined("NULL")),
        *_s_buffer_load_width_overlays(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_byte_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=None,
            fixed_saddr=_predefined("NULL"),
            implicit_flat_scratch=True,
            descriptor_key_suffix="_vaddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=0,
            fixed_saddr=_predefined("NULL"),
            implicit_flat_scratch=True,
            descriptor_key_suffix="_offset_only",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX11_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
            implicit_flat_scratch=True,
        ),
        *_ds_memory_overlays(include_u16_d16_loads=True),
        *_ds_crosslane_overlays(),
        _v_dot2_f32_f16_overlay(),
        _v_dot2_f32_bf16_overlay(),
        _v_dot4_i32_i8_overlay(signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot4_i32_iu8_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot8_i32_iu4_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot8_u32_u4_overlay(),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay(input_units=8)),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay(input_units=8)),
        *_with_zero_accumulator_form(
            _v_wmma_f16_16x16x16_f16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(
            _v_wmma_bf16_16x16x16_bf16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay(operand_units=4)),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay(operand_units=2)),
        _s_barrier_overlay(),
        *_gfx11_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            )
        ),
        _s_waitcnt_vscnt_overlay(),
        _s_waitcnt_depctr_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx11_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx11_core_overlays())
    )


def _gfx12_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp16_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_src0_inline_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_add_f32_src0_inline_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_sub_f32_src0_inline_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_mul_f32_src0_inline_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_min_f32_src0_inline_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_max_f32_src0_inline_overlay(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        *_s_load_dword_width_overlays("IOFFSET", offset_bit_width=24, include_b96=True),
        *_s_load_narrow_overlays("IOFFSET", offset_bit_width=24),
        *_s_load_dword_offset_only_overlays(
            "IOFFSET",
            offset_bit_width=24,
            fixed_soffset=_predefined("NULL"),
            include_b96=True,
        ),
        *_s_load_narrow_overlays(
            "IOFFSET", offset_bit_width=24, fixed_soffset=_predefined("NULL")
        ),
        *_s_buffer_load_width_overlays(
            "IOFFSET", offset_bit_width=24, include_b96=True
        ),
        *_s_buffer_load_narrow_overlays("IOFFSET", offset_bit_width=24),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_96_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_96_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        *_buffer_byte_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            include_packed_half_add=True,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VSCRATCH",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            fixed_vaddr=None,
            fixed_saddr=_predefined("NULL"),
            descriptor_key_suffix="_vaddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VSCRATCH",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            fixed_vaddr=0,
            fixed_saddr=_predefined("NULL"),
            descriptor_key_suffix="_offset_only",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            include_packed_half_add=True,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX12_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_VFLAT",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            implicit_flat_scratch=False,
        ),
        *_ds_memory_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(("OFFSET1", 0),),
            include_packed_half_atomic_add=True,
            include_u16_d16_loads=True,
        ),
        *_ds_crosslane_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(),
            include_fetch_invalid=True,
        ),
        _v_dot2_f32_f16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot2_f32_bf16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_i32_i8_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot4_u32_u8_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot8_i32_i4_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot8_u32_u4_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="bf8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="bf8"),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f16_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_bf16_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay()),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x32_iu4_overlay()),
        *_rdna4_swmmac_overlays(),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx12_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx12_core_overlays())
    )


def _gfx1250_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp16_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_src0_inline_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_add_f32_src0_inline_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_sub_f32_src0_inline_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_mul_f32_src0_inline_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_min_f32_src0_inline_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_max_f32_src0_inline_overlay(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        *_s_load_dword_width_overlays("IOFFSET", offset_bit_width=24, include_b96=True),
        *_s_load_narrow_overlays("IOFFSET", offset_bit_width=24),
        *_s_load_dword_offset_only_overlays(
            "IOFFSET",
            offset_bit_width=24,
            fixed_soffset=_predefined("NULL"),
            include_b96=True,
        ),
        *_s_load_narrow_overlays(
            "IOFFSET", offset_bit_width=24, fixed_soffset=_predefined("NULL")
        ),
        *_s_buffer_load_width_overlays(
            "IOFFSET", offset_bit_width=24, include_b96=True
        ),
        *_s_buffer_load_narrow_overlays("IOFFSET", offset_bit_width=24),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_96_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_96_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        *_buffer_byte_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            include_packed_half_add=True,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VSCRATCH",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            fixed_vaddr=None,
            fixed_saddr=_predefined("NULL"),
            descriptor_key_suffix="_vaddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VSCRATCH",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            fixed_vaddr=0,
            fixed_saddr=_predefined("NULL"),
            descriptor_key_suffix="_offset_only",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            include_packed_half_add=True,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX12_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_VFLAT",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            implicit_flat_scratch=False,
        ),
        *_ds_memory_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(("OFFSET1", 0),),
            include_packed_half_atomic_add=True,
            include_u16_d16_loads=True,
        ),
        *_ds_crosslane_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(),
            include_fetch_invalid=True,
        ),
        _v_dot2_f32_f16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot2_f32_bf16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_i32_i8_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot4_u32_u8_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot8_i32_i4_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot8_u32_u4_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="bf8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="bf8"),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f16_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_bf16_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay()),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x32_iu4_overlay()),
        *_rdna4_swmmac_overlays(),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx1250_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx1250_core_overlays())
    )


_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.cdna4.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_AGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    ),
    register_parts=_VGPR_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_MFMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            smem_store_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),
            ),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MFMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_MFMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_MFMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_MEMORY,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_GFX950_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)


_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.cdna3.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=102,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_AGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    ),
    register_parts=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.register_parts,
    resources=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.resources,
    schedule_classes=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
)


_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna3.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    ),
    register_parts=_VGPR_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            smem_store_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_MEMORY,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_GFX11_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)

_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    ),
    register_parts=_VGPR_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_SWMMAC, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            smem_store_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SWMMAC,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_SWMMAC, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_SWMMAC),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_LDS_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_SMEM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_SMEM_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)


def _encoded_operand(operand: Operand, field_name: str) -> Operand:
    return replace(operand, encoding_field_id=amdgpu_encoding_field_id(field_name))


def _encoded_immediate(
    immediate: Immediate,
    field_name: str,
    *,
    bit_width: int | None = None,
    unsigned_max: int | None = None,
    default_value: int | None = None,
) -> Immediate:
    flags = immediate.flags
    if default_value is not None and ImmediateFlag.DEFAULT_VALUE not in flags:
        flags = (*flags, ImmediateFlag.DEFAULT_VALUE)
    return replace(
        immediate,
        flags=flags,
        encoding_field_id=amdgpu_encoding_field_id(field_name),
        bit_width=immediate.bit_width if bit_width is None else bit_width,
        unsigned_max=immediate.unsigned_max if unsigned_max is None else unsigned_max,
        default_value=immediate.default_value
        if default_value is None
        else default_value,
    )


def _gfx1250_wmma_scale_immediates() -> tuple[Immediate, ...]:
    return (
        _encoded_immediate(
            _MATRIX_A_FORMAT_IMMEDIATE, "MATRIX_A_FMT", bit_width=3, unsigned_max=7
        ),
        _encoded_immediate(
            _MATRIX_B_FORMAT_IMMEDIATE, "MATRIX_B_FMT", bit_width=3, unsigned_max=7
        ),
        _encoded_immediate(
            _MATRIX_A_SCALE_IMMEDIATE, "MATRIX_A_SCALE", bit_width=1, unsigned_max=1
        ),
        _encoded_immediate(
            _MATRIX_B_SCALE_IMMEDIATE, "MATRIX_B_SCALE", bit_width=1, unsigned_max=1
        ),
        _encoded_immediate(
            _MATRIX_A_SCALE_FORMAT_IMMEDIATE,
            "MATRIX_A_SCALE_FMT",
            bit_width=2,
            unsigned_max=3,
        ),
        _encoded_immediate(
            _MATRIX_B_SCALE_FORMAT_IMMEDIATE,
            "MATRIX_B_SCALE_FMT",
            bit_width=2,
            unsigned_max=3,
        ),
        _encoded_immediate(_MATRIX_A_REUSE_IMMEDIATE, "MATRIX_A_REUSE"),
        _encoded_immediate(_MATRIX_B_REUSE_IMMEDIATE, "MATRIX_B_REUSE"),
    )


def _gfx1250_matrix_reuse_immediates() -> tuple[Immediate, ...]:
    return (
        _encoded_immediate(
            _MATRIX_A_REUSE_IMMEDIATE, "MATRIX_A_REUSE", default_value=0
        ),
        _encoded_immediate(
            _MATRIX_B_REUSE_IMMEDIATE, "MATRIX_B_REUSE", default_value=0
        ),
    )


def _gfx1250_swmmac_index_immediate(field_name: str) -> Immediate:
    # gfx1250 SWMMAC index-key variants share encoding bit 11.
    return _encoded_immediate(
        Immediate(
            field_name,
            ImmediateKind.UNSIGNED,
            bit_width=32,
            unsigned_max=(2**32) - 1,
        ),
        "INDEX_KEY_16BIT",
        bit_width=1,
        unsigned_max=1,
        default_value=0,
    )


_GFX1250_MATRIX_REUSE_IMMEDIATE_FIELDS = ("matrix_a_reuse", "matrix_b_reuse")

_GFX1250_WMMA_ROWS = (
    ("f32.16x16x4.f32", 0x5D, 2, 2, 8, 8, True),
    ("f32.16x16x32.f16", 0x60, 8, 8, 8, 8, True),
    ("f16.16x16x32.f16", 0x61, 8, 8, 4, 4, True),
    ("f32.16x16x32.bf16", 0x62, 8, 8, 8, 8, True),
    ("bf16.16x16x32.bf16", 0x63, 8, 8, 4, 4, True),
    ("bf16f32.16x16x32.bf16", 0x64, 8, 8, 8, 4, True),
    ("f32.16x16x64.fp8.fp8", 0x6A, 8, 8, 8, 8, True),
    ("f32.16x16x64.fp8.bf8", 0x6B, 8, 8, 8, 8, True),
    ("f32.16x16x64.bf8.fp8", 0x6C, 8, 8, 8, 8, True),
    ("f32.16x16x64.bf8.bf8", 0x6D, 8, 8, 8, 8, True),
    ("f16.16x16x64.fp8.fp8", 0x6E, 8, 8, 4, 4, True),
    ("f16.16x16x64.fp8.bf8", 0x6F, 8, 8, 4, 4, True),
    ("f16.16x16x64.bf8.fp8", 0x70, 8, 8, 4, 4, True),
    ("f16.16x16x64.bf8.bf8", 0x71, 8, 8, 4, 4, True),
    ("i32.16x16x64.iu8", 0x72, 8, 8, 8, 8, True),
    ("f32.16x16x128.fp8.fp8", 0x80, 16, 16, 8, 8, True),
    ("f32.16x16x128.fp8.bf8", 0x81, 16, 16, 8, 8, True),
    ("f32.16x16x128.bf8.fp8", 0x82, 16, 16, 8, 8, True),
    ("f32.16x16x128.bf8.bf8", 0x83, 16, 16, 8, 8, True),
    ("f16.16x16x128.fp8.fp8", 0x84, 16, 16, 4, 4, True),
    ("f16.16x16x128.fp8.bf8", 0x85, 16, 16, 4, 4, True),
    ("f16.16x16x128.bf8.fp8", 0x86, 16, 16, 4, 4, True),
    ("f16.16x16x128.bf8.bf8", 0x87, 16, 16, 4, 4, True),
    ("f32.32x16x128.f4", 0x88, 16, 8, 16, 16, False),
)


def _gfx1250_wmma_descriptor(
    name: str,
    encoding_id: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    has_reuse_immediates: bool,
) -> Descriptor:
    suffix = name.replace(".", "_")
    immediates = _gfx1250_matrix_reuse_immediates() if has_reuse_immediates else ()
    immediate_fields = (
        _GFX1250_MATRIX_REUSE_IMMEDIATE_FIELDS if has_reuse_immediates else ()
    )
    return Descriptor(
        key=f"amdgpu.v_wmma_{suffix}",
        mnemonic=f"v_wmma_{suffix}",
        semantic_tag=f"matrix.wmma.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("a", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("b", units=rhs_units), "SRC1"),
            _encoded_operand(
                _vgpr_const_operand("acc", units=accumulator_units), "SRC2"
            ),
        ),
        immediates=immediates,
        constraints=_destructive_accumulator_constraints(3),
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 3),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b", "acc"),
            immediates=immediate_fields,
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_WMMA,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx1250_wmma_descriptors() -> tuple[Descriptor, ...]:
    return tuple(_gfx1250_wmma_descriptor(*row) for row in _GFX1250_WMMA_ROWS)


_GFX1250_WMMA_SCALE_ROWS = (
    ("scale.f32.16x16x128.f8f6f4.f8.f8", 0x33, 0x35, 16, 16, 8, 8, 1),
    ("scale16.f32.16x16x128.f8f6f4.f8.f8", 0x33, 0x3A, 16, 16, 8, 8, 2),
    ("scale.f32.32x16x128.f4", 0x88, 0x35, 16, 8, 16, 16, 1),
    ("scale16.f32.32x16x128.f4", 0x88, 0x3A, 16, 8, 16, 16, 2),
)


def _gfx1250_wmma_scale_descriptor(
    name: str,
    encoding_id: int,
    x2_encoding: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    scale_units: int,
) -> Descriptor:
    suffix = name.replace(".", "_")
    return Descriptor(
        key=f"amdgpu.v_wmma_{suffix}",
        mnemonic=f"v_wmma_{suffix}",
        semantic_tag=f"matrix.wmma.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("a", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("b", units=rhs_units), "SRC1"),
            _encoded_operand(
                _vgpr_const_operand("acc", units=accumulator_units), "SRC2"
            ),
            _encoded_operand(
                _sgpr_vgpr_operand("scale_src0", units=scale_units), "SCALE_SRC0"
            ),
            _encoded_operand(
                _sgpr_vgpr_operand("scale_src1", units=scale_units), "SCALE_SRC1"
            ),
        ),
        immediates=_gfx1250_wmma_scale_immediates(),
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("X2ENCODING"), x2_encoding),
        ),
        constraints=_destructive_accumulator_constraints(3),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b", "acc", "scale_src0", "scale_src1"),
            immediates=(
                "matrix_a_fmt",
                "matrix_b_fmt",
                "matrix_a_scale",
                "matrix_b_scale",
                "matrix_a_scale_fmt",
                "matrix_b_scale_fmt",
                "matrix_a_reuse",
                "matrix_b_reuse",
            ),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_WMMA_SCALE,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3PX2,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx1250_wmma_scale_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _gfx1250_wmma_scale_descriptor(*row) for row in _GFX1250_WMMA_SCALE_ROWS
    )


_GFX1250_SWMMAC_ROWS = (
    ("f32.16x16x64.f16", 0x65, 8, 16, 8, 8, "index_key_16bit", 1),
    ("f32.16x16x64.bf16", 0x66, 8, 16, 8, 8, "index_key_16bit", 1),
    ("f16.16x16x64.f16", 0x67, 8, 16, 4, 4, "index_key_16bit", 1),
    ("bf16.16x16x64.bf16", 0x68, 8, 16, 4, 4, "index_key_16bit", 1),
    ("bf16f32.16x16x64.bf16", 0x69, 8, 16, 8, 8, "index_key_16bit", 1),
    ("f32.16x16x128.fp8.fp8", 0x73, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f32.16x16x128.fp8.bf8", 0x74, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f32.16x16x128.bf8.fp8", 0x75, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f32.16x16x128.bf8.bf8", 0x76, 8, 16, 8, 8, "index_key_32bit", 2),
    ("f16.16x16x128.fp8.fp8", 0x77, 8, 16, 4, 4, "index_key_32bit", 2),
    ("f16.16x16x128.fp8.bf8", 0x78, 8, 16, 4, 4, "index_key_32bit", 2),
    ("f16.16x16x128.bf8.fp8", 0x79, 8, 16, 4, 4, "index_key_32bit", 2),
    ("f16.16x16x128.bf8.bf8", 0x7A, 8, 16, 4, 4, "index_key_32bit", 2),
    ("i32.16x16x128.iu8", 0x7B, 8, 16, 8, 8, "index_key_32bit", 2),
)


def _gfx1250_swmmac_descriptor(
    name: str,
    encoding_id: int,
    lhs_units: int,
    rhs_units: int,
    accumulator_units: int,
    result_units: int,
    index_immediate: str,
    index_units: int,
) -> Descriptor:
    suffix = name.replace(".", "_")
    return Descriptor(
        key=f"amdgpu.v_swmmac_{suffix}",
        mnemonic=f"v_swmmac_{suffix}",
        semantic_tag=f"matrix.swmmac.{name}",
        operands=(
            _encoded_operand(_vgpr_result(units=result_units), "VDST"),
            _encoded_operand(_vgpr_operand("acc", units=accumulator_units), "VDST"),
            _encoded_operand(_vgpr_operand("a", units=lhs_units), "SRC0"),
            _encoded_operand(_vgpr_operand("b", units=rhs_units), "SRC1"),
            _encoded_operand(_sgpr_vgpr_operand("index", units=index_units), "SRC2"),
        ),
        immediates=(
            _gfx1250_swmmac_index_immediate(index_immediate),
            *_gfx1250_matrix_reuse_immediates(),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        encoding_field_values=(
            EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 3),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("acc", "a", "b", "index"),
            immediates=(index_immediate, *_GFX1250_MATRIX_REUSE_IMMEDIATE_FIELDS),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_SWMMAC,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gfx1250_swmmac_descriptors() -> tuple[Descriptor, ...]:
    return tuple(_gfx1250_swmmac_descriptor(*row) for row in _GFX1250_SWMMAC_ROWS)


def _gfx125x_reg_classes() -> tuple[RegClass, ...]:
    return (
        *(
            replace(reg_class, allocatable_count=1024)
            if reg_class.name == _REG_VGPR
            else reg_class
            for reg_class in _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.reg_classes
        ),
        RegClass(
            _REG_MODE,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    )


_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.gfx125x.core",
    reg_classes=_gfx125x_reg_classes(),
    register_parts=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.register_parts,
    resources=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.resources,
    schedule_classes=(
        *_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
        ScheduleClass(
            _SCHEDULE_WMMA_SCALE,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MODE_CONTROL,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=(
        _s_set_vgpr_msb_descriptor(),
        *_gfx1250_wmma_descriptors(),
        *_gfx1250_wmma_scale_descriptors(),
        *_gfx1250_swmmac_descriptors(),
    ),
)


def _amdgpu_core_descriptor_set_bases() -> tuple[DescriptorSet, ...]:
    return (
        _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
    )


def _amdgpu_descriptor_ref_key_set() -> set[str]:
    keys: set[str] = set()
    keys.update(_MANUAL_SCALAR_DESCRIPTOR_KEYS)
    keys.update(descriptor.key for descriptor in _hal_buffer_descriptor_pseudos())
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        keys.update(descriptor.key for descriptor in descriptor_set.descriptors)
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        keys.update(overlay.descriptor_key for overlay in overlays)
    return keys


__all__ = (
    "_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE",
    "_amdgpu_core_descriptor_set_bases",
    "_amdgpu_descriptor_ref_key_set",
    "_cdna_core_overlays",
    "_gfx125x_reg_classes",
    "_gfx11_core_overlay_descriptors",
    "_gfx11_core_overlays",
    "_gfx1250_core_overlay_descriptors",
    "_gfx1250_core_overlays",
    "_gfx12_core_overlay_descriptors",
    "_gfx12_core_overlays",
    "_gfx940_core_overlay_descriptors",
    "_gfx940_core_overlays",
    "_gfx950_core_overlay_descriptors",
    "_gfx950_core_overlays",
)
