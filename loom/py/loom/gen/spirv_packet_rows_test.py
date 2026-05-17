# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.spirv_packet_rows import generate_tables
from loom.target.arch.spirv.scalar_memory import STORAGE_BUFFER_SCALARS


def test_generation_emits_scalar_memory_packet_rows() -> None:
    tables = generate_tables()

    for scalar in STORAGE_BUFFER_SCALARS:
        suffix = scalar.suffix.upper()
        assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_PTR_ACCESS_CHAIN_STORAGE_BUFFER_{suffix}_BYTE_OFFSET" in tables
        assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_STORAGE_BUFFER_{suffix}" in tables
        assert f"SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_STORE_STORAGE_BUFFER_{suffix}" in tables
        assert scalar.scalar_enum in tables
        assert f".memory_alignment = {scalar.byte_width}" in tables

    assert "LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS" in tables
    assert "LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER" in tables


def test_generation_keeps_storage_buffer_address_untyped_until_access_chain() -> None:
    tables = generate_tables()

    access_row_start = tables.index("SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_PTR_ACCESS_CHAIN_STORAGE_BUFFER_F32_BYTE_OFFSET")
    access_row = tables[access_row_start : access_row_start + 800]
    assert "LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS" in access_row
    assert "LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER" in access_row
    assert "LOOM_SPIRV_SCALAR_TYPE_F32" in access_row


def test_generation_emits_coordinate_arithmetic_packet_rows() -> None:
    tables = generate_tables()

    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ISUB_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_FADD_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_FMUL_F64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SDIV_I16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SREM_I64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_UDIV_U32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_UMOD_U64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IMUL_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IMUL_ADD_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ISUB_OFFSET64" in tables
    assert "LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD" in tables


def test_generation_emits_integer_compare_and_select_rows() -> None:
    tables = generate_tables()

    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_LESS_THAN_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_GREATER_THAN_I64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_LESS_THAN_U8" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_LESS_THAN_EQUAL_OFFSET64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_I64" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_OFFSET64" in tables
    assert "LOOM_SPIRV_VALUE_CLASS_BOOL" in tables
    assert "LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE" in tables
    assert "LOOM_SPIRV_PACKET_FORM_SELECT" in tables


def test_generation_emits_scalar_conversion_rows() -> None:
    tables = generate_tables()

    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_CONVERT_I8_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_CONVERT_I64_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_S_TO_F_I16_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_F_TO_S_F32_I16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_CONVERT_U8_U32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_U_TO_F_U16_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONVERT_F_TO_U_F32_U16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_I32_U32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_U32_I32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_F_CONVERT_F16_F32" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_F_CONVERT_F32_F16" in tables
    assert "SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_I32_F32" in tables
    assert "LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT" in tables
