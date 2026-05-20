# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Scalar, buffer, global, and global-to-LDS memory descriptor overlays."""

from __future__ import annotations

from .common import *

_MEMORY_DWORD_VECTOR_WIDTHS = ((32, 1), (64, 2), (96, 3), (128, 4))


def _s_buffer_load_dword_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_buffer_load_dword",
        instruction_name="S_BUFFER_LOAD_DWORD",
        mnemonic="s_buffer_load_dword",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result()),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_EFFECT,),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b64",
    instruction_name: str = "S_BUFFER_LOAD_B64",
    mnemonic: str = "s_buffer_load_b64",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_load_dwordx2_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx2",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_result(units=2)),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx2_offset_only"
            ),
            _soffset_offset_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx2_offset_only"
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX2",
        mnemonic="s_load_dwordx2",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=operands,
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        operand_forms=operand_forms,
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=() if fixed_soffset is not None else None,
    )


def _s_load_dwordx4_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx4",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_result(units=4)),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx4_offset_only"
            ),
            _soffset_offset_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx4_offset_only"
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX4",
        mnemonic="s_load_dwordx4",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u128",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=operands,
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B128,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_B128_EFFECT,),
        operand_forms=operand_forms,
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=() if fixed_soffset is not None else None,
    )


def _s_load_dword_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dword",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_result()),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor="amdgpu.s_load_dword_offset_only"
            ),
            _soffset_offset_operand_form(
                replacement_descriptor="amdgpu.s_load_dword_offset_only"
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORD",
        mnemonic="s_load_dword",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=operands,
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_EFFECT,),
        operand_forms=operand_forms,
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=() if fixed_soffset is not None else None,
    )


def _buffer_load_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_dword_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_dword",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_load_off_zero_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", _predefined("v0")),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("IDXEN", 0),
            ("OFFEN", 0),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_load_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key="amdgpu.buffer_load_dword_off_zero",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        semantic_tag="memory.load.u32",
        payload_units=1,
        memory_effect=_GLOBAL_LOAD_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_b16_d16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_b16_d16",
        instruction_name="BUFFER_LOAD_SHORT_D16",
        mnemonic="buffer_load_short_d16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.d16.low",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_read_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_u16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_u16",
        instruction_name="BUFFER_LOAD_USHORT",
        mnemonic="buffer_load_u16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.zero_extend",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_U16,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_i8_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_i8",
        instruction_name="BUFFER_LOAD_SBYTE",
        mnemonic="buffer_load_i8",
        encoding_name=encoding_name,
        semantic_tag="memory.load.i8.sign_extend",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_I8,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_read_effect(8),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_sized_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_read_memory(width_bits),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_read_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b64_off_zero",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=64,
        payload_units=2,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
    )


def _buffer_load_64_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64_off_zero",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u64",
        payload_units=2,
        memory_effect=_GLOBAL_LOAD_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_96_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b96",
    instruction_name: str = "BUFFER_LOAD_B96",
    mnemonic: str = "buffer_load_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b96_off_zero",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=96,
        payload_units=3,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
    )


def _buffer_load_96_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b96_off_zero",
    instruction_name: str = "BUFFER_LOAD_B96",
    mnemonic: str = "buffer_load_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u96",
        payload_units=3,
        memory_effect=_GLOBAL_LOAD_B96_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B96,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b128_off_zero",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=128,
        payload_units=4,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
    )


def _buffer_load_128_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128_off_zero",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u128",
        payload_units=4,
        memory_effect=_GLOBAL_LOAD_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_dword_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_dword",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u32",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value")),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_store_off_zero_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", _predefined("v0")),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("IDXEN", 0),
            ("OFFEN", 0),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_store_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key="amdgpu.buffer_store_dword_off_zero",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        semantic_tag="memory.store.u32",
        payload_units=1,
        memory_effect=_GLOBAL_STORE_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_b16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_b16",
        instruction_name="BUFFER_STORE_SHORT",
        mnemonic="buffer_store_short",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u16.low",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_write_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_b8_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_b8",
        instruction_name="BUFFER_STORE_BYTE",
        mnemonic="buffer_store_b8",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u8",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value")),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(8),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_write_effect(8),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_sized_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(width_bits),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_write_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_store_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b64_off_zero",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=64,
        payload_units=2,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
    )


def _buffer_store_64_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64_off_zero",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u64",
        payload_units=2,
        memory_effect=_GLOBAL_STORE_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_96_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b96",
    instruction_name: str = "BUFFER_STORE_B96",
    mnemonic: str = "buffer_store_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b96_off_zero",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=96,
        payload_units=3,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
    )


def _buffer_store_96_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b96_off_zero",
    instruction_name: str = "BUFFER_STORE_B96",
    mnemonic: str = "buffer_store_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u96",
        payload_units=3,
        memory_effect=_GLOBAL_STORE_B96_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B96,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b128_off_zero",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=128,
        payload_units=4,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
    )


def _buffer_store_128_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128_off_zero",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u128",
        payload_units=4,
        memory_effect=_GLOBAL_STORE_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_b16_memory_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _buffer_load_u16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
        _buffer_load_b16_d16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
        _buffer_store_b16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
    )


def _buffer_byte_memory_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _buffer_load_i8_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
        _buffer_store_b8_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
    )


def _global_load_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    global_read_memory: AmdgpuImplicitOperandOverlay | None = None,
    semantic_tag: str | None = None,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        global_read_memory or _ignore_global_read_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(data_field_name, _vgpr_result(units=units)),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag or f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic=mnemonic,
            results=("dst",),
            operands=("addr", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic=mnemonic,
            results=("dst",),
            operands=("addr",),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_load_b16_d16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_read_memory(16),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(
            data_field_name,
            _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
            size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
        ),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_LOAD_SHORT_D16",
        mnemonic="global_load_short_d16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.d16.low",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic="global_load_short_d16",
            results=("dst",),
            operands=("addr", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic="global_load_short_d16",
            results=("dst",),
            operands=("addr",),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_load_u16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _global_load_overlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_LOAD_USHORT",
        mnemonic="global_load_u16",
        encoding_name=encoding_name,
        address_field_name=address_field_name,
        data_field_name=data_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        saddr_off=saddr_off,
        width_bits=16,
        units=1,
        address_units=address_units,
        implicit_m0=implicit_m0,
        global_read_memory=_IGNORE_GLOBAL_READ_MEMORY_U16,
        cache_fields=cache_fields,
    )


def _global_load_i8_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _global_load_overlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_LOAD_SBYTE",
        mnemonic="global_load_i8",
        encoding_name=encoding_name,
        address_field_name=address_field_name,
        data_field_name=data_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        saddr_off=saddr_off,
        width_bits=8,
        units=1,
        address_units=address_units,
        implicit_m0=implicit_m0,
        global_read_memory=_IGNORE_GLOBAL_READ_MEMORY_I8,
        semantic_tag="memory.load.i8.sign_extend",
        cache_fields=cache_fields,
    )


def _scratch_load_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    width_bits: int,
    units: int,
    fixed_vaddr: AmdgpuFixedEncodingValue | None,
    fixed_saddr: AmdgpuFixedEncodingValue | None,
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(data_field_name, _vgpr_result(units=units)),
    )
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_scratch_memory(width_bits=width_bits, is_input=True),
    )
    if implicit_flat_scratch:
        implicit_operands = (*implicit_operands, _IGNORE_FLAT_SCRATCH_INPUT)
    if implicit_m0:
        implicit_operands = (*implicit_operands, _implicit_m0_clobber())
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("SVE", 1 if fixed_vaddr is None else 0),
    )
    if fixed_vaddr is None:
        operands += (AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr")),)
    else:
        fixed_encoding_fields += ((address_field_name, fixed_vaddr),)
    if fixed_saddr is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr")),)
    else:
        fixed_encoding_fields += (("SADDR", fixed_saddr),)
    if fixed_vaddr is None and fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_svs"
        asm_operands: tuple[str, ...] = ("addr", "saddr")
    elif fixed_vaddr is None:
        asm_mnemonic = f"{mnemonic}_vaddr"
        asm_operands = ("addr",)
    elif fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_saddr"
        asm_operands = ("saddr",)
    else:
        asm_mnemonic = f"{mnemonic}_offset_only"
        asm_operands = ()
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.stack.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width)
            if offset_signed
            else _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_stack_memory_effect(EffectKind.READ, width_bits),),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=asm_mnemonic,
            results=("dst",),
            operands=asm_operands,
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _scratch_store_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    width_bits: int,
    units: int,
    fixed_vaddr: AmdgpuFixedEncodingValue | None,
    fixed_saddr: AmdgpuFixedEncodingValue | None,
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = ()
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_scratch_memory(width_bits=width_bits, is_input=False),
    )
    if implicit_flat_scratch:
        implicit_operands = (*implicit_operands, _IGNORE_FLAT_SCRATCH_INPUT)
    if implicit_m0:
        implicit_operands = (*implicit_operands, _implicit_m0_clobber())
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("SVE", 1 if fixed_vaddr is None else 0),
    )
    if fixed_vaddr is None:
        operands += (AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr")),)
    else:
        fixed_encoding_fields += ((address_field_name, fixed_vaddr),)
    operands += (
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=units)),
    )
    if fixed_saddr is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr")),)
    else:
        fixed_encoding_fields += (("SADDR", fixed_saddr),)
    if fixed_vaddr is None and fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_svs"
        asm_operands: tuple[str, ...] = ("addr", "value", "saddr")
    elif fixed_vaddr is None:
        asm_mnemonic = f"{mnemonic}_vaddr"
        asm_operands = ("addr", "value")
    elif fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_saddr"
        asm_operands = ("value", "saddr")
    else:
        asm_mnemonic = f"{mnemonic}_offset_only"
        asm_operands = ("value",)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.stack.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width)
            if offset_signed
            else _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_stack_memory_effect(EffectKind.WRITE, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=asm_mnemonic,
            operands=asm_operands,
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _scratch_memory_overlays(
    *,
    instruction_suffixes: tuple[str, ...],
    mnemonic_suffixes: tuple[str, ...],
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    fixed_vaddr: AmdgpuFixedEncodingValue | None,
    fixed_saddr: AmdgpuFixedEncodingValue | None,
    implicit_flat_scratch: bool = False,
    implicit_m0: bool = False,
    descriptor_key_suffix: str = "",
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *(
            _scratch_load_overlay(
                descriptor_key=(
                    f"amdgpu.scratch_load_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"SCRATCH_LOAD_{instruction_suffix}",
                mnemonic=f"scratch_load_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=load_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                offset_signed=offset_signed,
                width_bits=width_bits,
                units=units,
                fixed_vaddr=fixed_vaddr,
                fixed_saddr=fixed_saddr,
                implicit_flat_scratch=implicit_flat_scratch,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
        *(
            _scratch_store_overlay(
                descriptor_key=(
                    f"amdgpu.scratch_store_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"SCRATCH_STORE_{instruction_suffix}",
                mnemonic=f"scratch_store_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=store_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                offset_signed=offset_signed,
                width_bits=width_bits,
                units=units,
                fixed_vaddr=fixed_vaddr,
                fixed_saddr=fixed_saddr,
                implicit_flat_scratch=implicit_flat_scratch,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
    )


def _global_store_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_write_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=units)),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic=mnemonic,
            operands=("addr", "value", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic=mnemonic,
            operands=("addr", "value"),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_store_b16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_write_memory(16),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(
            data_field_name,
            _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
            size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_STORE_SHORT",
        mnemonic="global_store_short",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u16.low",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic="global_store_short",
            operands=("addr", "value", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic="global_store_short",
            operands=("addr", "value"),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_load_lds_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    address_units: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 13,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand("ADDR", units=address_units, has_saddr=saddr_off is None),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_FLAT_GLBL",
        semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        operands=operands,
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="legacy-lds-dma-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        ),
        implicit_operands=(_implicit_m0_input(),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_to_lds_effects(width_bits),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


_GLOBAL_LOAD_LDS_DWORD_VARIANTS = (("DWORD", "dword", 32),)

_GLOBAL_LOAD_LDS_GFX950_VARIANTS = (
    ("DWORD", "dword", 32),
    ("DWORDX3", "dwordx3", 96),
    ("DWORDX4", "dwordx4", 128),
)


def _global_load_lds_overlays(
    *,
    descriptor_key_suffix: str = "",
    address_units: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    cache_fields: tuple[tuple[str, int], ...] = (),
    variants: tuple[tuple[str, str, int], ...] = _GLOBAL_LOAD_LDS_GFX950_VARIANTS,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _global_load_lds_overlay(
            descriptor_key=(
                f"amdgpu.global_load_lds_{mnemonic_suffix}{descriptor_key_suffix}"
            ),
            instruction_name=f"GLOBAL_LOAD_LDS_{instruction_suffix}",
            mnemonic=f"global_load_lds_{mnemonic_suffix}",
            width_bits=width_bits,
            address_units=address_units,
            saddr_off=saddr_off,
            cache_fields=cache_fields,
        )
        for instruction_suffix, mnemonic_suffix, width_bits in variants
    )


def _global_b16_memory_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _global_load_u16_overlay(
            descriptor_key=f"amdgpu.global_load_u16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_load_b16_d16_overlay(
            descriptor_key=f"amdgpu.global_load_b16_d16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_store_b16_overlay(
            descriptor_key=f"amdgpu.global_store_b16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=store_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
    )


def _global_byte_memory_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _global_load_i8_overlay(
            descriptor_key=f"amdgpu.global_load_i8{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_store_overlay(
            descriptor_key=f"amdgpu.global_store_b8{descriptor_key_suffix}",
            instruction_name="GLOBAL_STORE_BYTE",
            mnemonic="global_store_b8",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=store_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            width_bits=8,
            units=1,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
    )


def _global_memory_overlays(
    *,
    instruction_suffixes: tuple[str, ...],
    mnemonic_suffixes: tuple[str, ...],
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *(
            _global_load_overlay(
                descriptor_key=(
                    f"amdgpu.global_load_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_LOAD_{instruction_suffix}",
                mnemonic=f"global_load_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=load_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
        *(
            _global_store_overlay(
                descriptor_key=(
                    f"amdgpu.global_store_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_STORE_{instruction_suffix}",
                mnemonic=f"global_store_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=store_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
    )


__all__ = (
    "_GLOBAL_LOAD_LDS_DWORD_VARIANTS",
    "_GLOBAL_LOAD_LDS_GFX950_VARIANTS",
    "_MEMORY_DWORD_VECTOR_WIDTHS",
    "_buffer_b16_memory_overlays",
    "_buffer_byte_memory_overlays",
    "_buffer_load_128_off_zero_overlay",
    "_buffer_load_128_overlay",
    "_buffer_load_96_off_zero_overlay",
    "_buffer_load_96_overlay",
    "_buffer_load_64_off_zero_overlay",
    "_buffer_load_64_overlay",
    "_buffer_load_b16_d16_overlay",
    "_buffer_load_dword_off_zero_overlay",
    "_buffer_load_dword_overlay",
    "_buffer_load_i8_overlay",
    "_buffer_load_off_zero_overlay",
    "_buffer_load_u16_overlay",
    "_buffer_store_128_off_zero_overlay",
    "_buffer_store_128_overlay",
    "_buffer_store_96_off_zero_overlay",
    "_buffer_store_96_overlay",
    "_buffer_store_64_off_zero_overlay",
    "_buffer_store_64_overlay",
    "_buffer_store_b16_overlay",
    "_buffer_store_b8_overlay",
    "_buffer_store_dword_off_zero_overlay",
    "_buffer_store_dword_overlay",
    "_buffer_store_off_zero_overlay",
    "_global_b16_memory_overlays",
    "_global_byte_memory_overlays",
    "_global_load_b16_d16_overlay",
    "_global_load_i8_overlay",
    "_global_load_lds_overlay",
    "_global_load_lds_overlays",
    "_global_load_overlay",
    "_global_load_u16_overlay",
    "_global_memory_overlays",
    "_global_store_b16_overlay",
    "_global_store_overlay",
    "_s_buffer_load_64_overlay",
    "_s_buffer_load_dword_overlay",
    "_s_load_dword_overlay",
    "_s_load_dwordx2_overlay",
    "_s_load_dwordx4_overlay",
    "_scratch_memory_overlays",
)
