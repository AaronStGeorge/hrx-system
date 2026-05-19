# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

from loom.gen import amdgpu_matrix_contract_tables
from loom.target.arch.amdgpu.descriptors import amdgpu_descriptor_ref_keys
from loom.target.arch.amdgpu.matrix_contracts import (
    AMDGPU_MATRIX_CONTRACTS,
    AmdgpuMatrixContract,
    payload,
)


def _contract(name: str) -> AmdgpuMatrixContract:
    for contract in AMDGPU_MATRIX_CONTRACTS:
        if contract.name == name:
            return contract
    raise ValueError(f"unknown AMDGPU matrix contract {name!r}")


def _contract_initializer(contract: AmdgpuMatrixContract) -> str:
    return amdgpu_matrix_contract_tables._contract_initializer(
        contract,
        descriptor_ref_key_set=set(amdgpu_descriptor_ref_keys()),
        keys_by_semantic_tag=(amdgpu_matrix_contract_tables._matrix_descriptor_keys_by_semantic_tag()),
        descriptor_shapes_by_key=(amdgpu_matrix_contract_tables._matrix_descriptor_shapes_by_key()),
    )


def test_generation_derives_semantic_tag_descriptor_ref() -> None:
    initializer = _contract_initializer(_contract("swmmac.f32.16x16x32.f16"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_F16" in initializer


def test_generation_accepts_one_matching_descriptor_shape_variant() -> None:
    initializer = _contract_initializer(_contract("wmma.f32.16x16x16.f16"))

    assert ".lhs_payload" in initializer


def test_generation_rejects_low_descriptor_payload_shape_drift() -> None:
    contract = replace(
        _contract("swmmac.f32.16x16x32.f16"),
        lhs=payload("f16", 0, 0),
    )

    try:
        _contract_initializer(contract)
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'swmmac.f32.16x16x32.f16'" in message
        assert "payload shape" in message
        assert "low descriptor 'amdgpu.v_swmmac_f32_16x16x32_f16'" in message
    else:
        raise AssertionError("expected payload shape validation to fail")
