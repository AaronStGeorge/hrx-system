# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

from loom.gen.low_descriptors import DescriptorAllowlist, generate_descriptor_set
from loom.target.arch.amdgpu.descriptors import (
    AMDGPU_GFX11_CORE_DESCRIPTOR_SET,
    AMDGPU_GFX12_CORE_DESCRIPTOR_SET,
    AMDGPU_GFX950_CORE_DESCRIPTOR_SET,
    AMDGPU_GFX1250_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.arch.x86.descriptors import (
    X86_AVX512_CORE_DESCRIPTOR_SET,
    X86_PACKED_DOT_DESCRIPTOR_SET,
)
from loom.target.emit.ireevm.descriptors import IREEVM_CORE_DESCRIPTOR_SET
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    LOW_DESCRIPTOR_SET_ABI_VERSION,
    AsmForm,
    AsmImmediate,
    DescriptorFlag,
    EnumDomain,
    EnumValue,
    Hazard,
    HazardKind,
    ImmediateKind,
    OperandRole,
)
from loom.target.test.descriptors import TEST_LOW_CORE_DESCRIPTOR_SET


def test_generate_ireevm_core_descriptor_set() -> None:
    generated = generate_descriptor_set(IREEVM_CORE_DESCRIPTOR_SET)

    assert "loom_ireevm_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x0c", "iree.vm.core")' in generated.source
    assert "iree.vm.add.i32" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "iree.vm.core"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 9
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["asm_forms"] >= 9
    assert any(descriptor["key"] == "iree.vm.call.import.i32" for descriptor in manifest["descriptors"])
    assert any(form["mnemonic"] == "vm.add.i32" for form in manifest["asm_forms"])


def test_allowlist_closes_over_referenced_descriptor_tables() -> None:
    generated = generate_descriptor_set(
        IREEVM_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(keys=("iree.vm.add.i32",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == ["iree.vm.add.i32"]
    assert manifest["table_counts"]["descriptors"] == 1
    assert manifest["table_counts"]["descriptor_refs"] == 1
    assert manifest["table_counts"]["asm_forms"] == 1
    assert manifest["table_counts"]["reg_classes"] == 6
    assert manifest["table_counts"]["reg_class_alts"] == 1
    assert manifest["table_counts"]["schedule_classes"] == 1
    assert manifest["table_counts"]["resources"] == 1
    assert "iree.vm.call.import.i32" not in generated.source


def test_generate_wasm_core_simd128_descriptor_set() -> None:
    generated = generate_descriptor_set(WASM_CORE_SIMD128_DESCRIPTOR_SET)

    assert "loom_wasm_core_simd128_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x11", "wasm.core.simd128")' in generated.source
    assert "wasm.i32x4.add" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "wasm.core.simd128"
    assert manifest["target"] == "wasm"
    assert manifest["feature_namespace"] == "wasm.simd128.v1"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 12
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["asm_forms"] >= 12
    assert any(descriptor["key"] == "wasm.v128.load" for descriptor in manifest["descriptors"])
    assert any(form["mnemonic"] == "i32x4.add" for form in manifest["asm_forms"])


def test_generate_amdgpu_gfx950_core_descriptor_set() -> None:
    generated = generate_descriptor_set(AMDGPU_GFX950_CORE_DESCRIPTOR_SET)

    assert "loom_amdgpu_gfx950_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x12", "amdgpu.gfx950.core")' in generated.source
    assert "amdgpu.v_add_u32" in generated.source
    assert "amdgpu.buffer_load_dword" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "amdgpu.gfx950.core"
    assert manifest["target"] == "amdgpu"
    assert manifest["feature_namespace"] == "amdgpu.gfx950.v1"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 8
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["asm_forms"] >= 8
    assert manifest["table_counts"]["reg_classes"] == 3
    assert manifest["table_counts"]["resources"] >= 6
    assert manifest["table_counts"]["hazards"] >= 1
    assert "LOOM_LOW_HAZARD_KIND_WAIT_COUNTER" in generated.source
    assert "LOOM_LOW_HAZARD_KIND_MIN_DISTANCE" in generated.source
    assert any(descriptor["key"] == "amdgpu.v_mfma_f32_16x16x16_f16" for descriptor in manifest["descriptors"])


def test_generate_amdgpu_gfx11_core_descriptor_set() -> None:
    generated = generate_descriptor_set(AMDGPU_GFX11_CORE_DESCRIPTOR_SET)

    assert "loom_amdgpu_gfx11_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x11", "amdgpu.gfx11.core")' in generated.source
    assert "amdgpu.v_add_u32" in generated.source
    assert "amdgpu.buffer_load_dword" in generated.source
    assert "amdgpu.v_wmma_f32_16x16x16_f16" in generated.source
    assert "amdgpu.s_waitcnt_depctr" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "amdgpu.gfx11.core"
    assert manifest["target"] == "amdgpu"
    assert manifest["feature_namespace"] == "amdgpu.gfx11.v1"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 10
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["asm_forms"] >= 10
    assert manifest["table_counts"]["reg_classes"] == 2
    assert manifest["table_counts"]["resources"] >= 7
    assert manifest["table_counts"]["hazards"] >= 1
    assert "amdgpu.wait.memory" in generated.source
    assert "amdgpu.wait.alu" in generated.source
    assert "amdgpu.wait.idle" in generated.source
    assert "LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER" in generated.source
    assert "LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE" in generated.source
    assert any(descriptor["key"] == "amdgpu.v_wmma_f32_16x16x16_f16" for descriptor in manifest["descriptors"])
    assert any(descriptor["key"] == "amdgpu.s_waitcnt" for descriptor in manifest["descriptors"])
    assert any(descriptor["key"] == "amdgpu.s_waitcnt_depctr" for descriptor in manifest["descriptors"])
    assert any(form["mnemonic"] == "s_waitcnt" and {"field": "vmcnt", "name": "vmcnt"} in form["immediates"] for form in manifest["asm_forms"])


def test_generate_amdgpu_gfx12_core_descriptor_set() -> None:
    generated = generate_descriptor_set(AMDGPU_GFX12_CORE_DESCRIPTOR_SET)

    assert "loom_amdgpu_gfx12_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x11", "amdgpu.gfx12.core")' in generated.source
    assert "amdgpu.v_add_u32" in generated.source
    assert "amdgpu.buffer_load_dword" in generated.source
    assert "amdgpu.v_wmma_f32_16x16x16_f16" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "amdgpu.gfx12.core"
    assert manifest["target"] == "amdgpu"
    assert manifest["feature_namespace"] == "amdgpu.gfx12.v1"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 10
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["asm_forms"] >= 10
    assert manifest["table_counts"]["reg_classes"] == 2
    assert manifest["table_counts"]["resources"] >= 6
    assert manifest["table_counts"]["hazards"] >= 1
    assert "amdgpu.wait.load" in generated.source
    assert "amdgpu.wait.store" in generated.source
    assert "amdgpu.wait.alu" in generated.source
    assert "amdgpu.wait.idle" in generated.source
    assert any(descriptor["key"] == "amdgpu.v_wmma_f32_16x16x16_f16" for descriptor in manifest["descriptors"])
    assert any(descriptor["key"] == "amdgpu.s_wait_loadcnt" for descriptor in manifest["descriptors"])


def test_generate_amdgpu_gfx1250_core_descriptor_set() -> None:
    generated = generate_descriptor_set(AMDGPU_GFX1250_CORE_DESCRIPTOR_SET)

    assert "loom_amdgpu_gfx1250_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x13", "amdgpu.gfx1250.core")' in generated.source
    assert "amdgpu.v_wmma_f32_16x16x32_f16" in generated.source
    assert "amdgpu.v_wmma_scale_f32_16x16x128_f8f6f4_f8_f8" in generated.source
    assert "amdgpu.v_swmmac_f32_16x16x64_f16" in generated.source
    assert ".encoding_id = LOOM_LOW_ID_NONE" in generated.source
    assert "LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "amdgpu.gfx1250.core"
    assert manifest["target"] == "amdgpu"
    assert manifest["feature_namespace"] == "amdgpu.gfx1250.v1"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 14
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["asm_forms"] >= 14
    assert manifest["table_counts"]["reg_classes"] == 2
    assert manifest["table_counts"]["resources"] >= 8
    assert manifest["table_counts"]["hazards"] >= 1
    assert "LOOM_LOW_HAZARD_KIND_WAIT_COUNTER" in generated.source
    assert "LOOM_LOW_HAZARD_KIND_MIN_DISTANCE" in generated.source
    assert any(descriptor["key"] == "amdgpu.v_wmma_scale16_f32_16x16x128_f8f6f4_f8_f8" for descriptor in manifest["descriptors"])
    assert any(descriptor["key"] == "amdgpu.v_swmmac_f32_16x16x64_f16" for descriptor in manifest["descriptors"])
    assert any(descriptor["key"] == "amdgpu.v_wmma_f32_16x16x32_f16" and "LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO" in descriptor["flags"] for descriptor in manifest["descriptors"])


def test_generate_x86_avx512_core_descriptor_set() -> None:
    generated = generate_descriptor_set(X86_AVX512_CORE_DESCRIPTOR_SET)

    assert "loom_x86_avx512_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x0f", "x86.avx512.core")' in generated.source
    assert "x86.avx512.vpaddd.zmm" in generated.source
    assert "x86.avx512.vpdpbusd.zmm" in generated.source
    assert "x86.avx512.vdpbf16ps.zmm" in generated.source
    assert "x86.avx512.kandq" in generated.source
    assert "x86.avx512.jmp" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "x86.avx512.core"
    assert manifest["target"] == "x86"
    assert manifest["feature_namespace"] == "x86.avx512.v1"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 7
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["asm_forms"] >= 6
    assert manifest["table_counts"]["reg_classes"] == 3
    assert manifest["table_counts"]["resources"] >= 7
    assert any(descriptor["key"] == "x86.avx512.vpdpbusd.zmm" for descriptor in manifest["descriptors"])
    assert any(descriptor["key"] == "x86.avx512.kandq" for descriptor in manifest["descriptors"])
    assert any(descriptor["key"] == "x86.avx512.jmp" for descriptor in manifest["descriptors"])
    assert any(form["mnemonic"] == "vpaddd" for form in manifest["asm_forms"])


def test_generate_x86_packed_dot_descriptor_set() -> None:
    generated = generate_descriptor_set(X86_PACKED_DOT_DESCRIPTOR_SET)

    assert "loom_x86_packed_dot_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x13", "x86.packed_dot.core")' in generated.source
    assert "x86.avx512-vnni.vpdpbusd.512" in generated.source
    assert "x86.avx512-bf16.vdpbf16ps.512" in generated.source
    assert "x86.avx-vnni-int8.vpdpbssd.256" in generated.source
    assert "x86.avx10.2.vdpphps.512" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "x86.packed_dot.core"
    assert manifest["target"] == "x86"
    assert manifest["feature_namespace"] == "x86.packed_dot.v1"
    assert manifest["abi_version"] == LOW_DESCRIPTOR_SET_ABI_VERSION
    assert manifest["table_counts"]["descriptors"] >= 53
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert manifest["table_counts"]["reg_classes"] == 3
    assert manifest["table_counts"]["feature_mask_words"] >= 53
    assert any(descriptor["key"] == "x86.avx10.2.vdpphps.512" for descriptor in manifest["descriptors"])


def test_generate_test_low_core_descriptor_set() -> None:
    generated = generate_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    manifest = json.loads(generated.manifest_json)

    assert manifest["key"] == "test.low.core"
    assert manifest["target"] == "test.low"
    assert manifest["table_counts"]["asm_forms"] >= 13
    assert any(descriptor["key"] == "test.spv.op_iadd.i32" for descriptor in manifest["descriptors"])
    assert any(form["mnemonic"] == "OpIAdd" for form in manifest["asm_forms"])


def test_generator_resolves_symbolic_hazard_resources() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(
                Hazard(
                    HazardKind.MIN_DISTANCE,
                    resource="test.scalar",
                    producer_stage=0,
                    consumer_stage=1,
                    distance=2,
                ),
            ),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    generated = generate_descriptor_set(
        descriptor_set,
        DescriptorAllowlist(keys=("test.add.i32",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert manifest["table_counts"]["hazards"] == 1
    assert ".reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE" in generated.source
    assert ".reference_id = 0" in generated.source


def test_generator_rejects_unknown_hazard_resource() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(Hazard(HazardKind.MIN_DISTANCE, resource="missing"),),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    with pytest.raises(
        ValueError,
        match=("schedule class 'test.scalar.alu' hazard references unknown resource 'missing'"),
    ):
        generate_descriptor_set(
            descriptor_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_generator_rejects_ambiguous_hazard_reference() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(Hazard(HazardKind.MIN_DISTANCE, resource="test.scalar", counter_id=0),),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    with pytest.raises(
        ValueError,
        match=("schedule class 'test.scalar.alu' hazard must reference exactly one resource, counter, or target id"),
    ):
        generate_descriptor_set(
            descriptor_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_allowlist_accepts_semantic_tags() -> None:
    generated = generate_descriptor_set(
        IREEVM_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(semantic_tags=("control.return.void",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == ["iree.vm.return.void"]


def test_allowlist_rejects_unknown_descriptor_key() -> None:
    with pytest.raises(ValueError, match="allowlist references unknown descriptor key 'missing'"):
        generate_descriptor_set(
            IREEVM_CORE_DESCRIPTOR_SET,
            DescriptorAllowlist(keys=("missing",)),
        )


def test_generator_rejects_asm_form_unknown_operand_field() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "missing")),),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' asm form 'vm.add.i32' operand references unknown operand field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_result_with_operand_role() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        asm_forms=(AsmForm(results=("lhs",), operands=("rhs",)),),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' asm form 'vm.add.i32' result field 'lhs' names a non-result operand"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_ambiguous_asm_mnemonics() -> None:
    first = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        asm_forms=(AsmForm(mnemonic="dup", results=("dst",), operands=("lhs", "rhs")),),
    )
    second = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[2],
        asm_forms=(AsmForm(mnemonic="dup", results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(first, second))

    with pytest.raises(
        ValueError,
        match=("asm mnemonic 'dup' is ambiguous between descriptors 'iree.vm.add.i32' and 'iree.vm.sub.i32'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_unknown_immediate_field() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        asm_forms=(AsmForm(results=("dst",), immediates=(AsmImmediate("missing"),)),),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.const.i32' asm form 'vm.const.i32' immediate references unknown immediate field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_duplicate_asm_immediate_name() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[5],
        asm_forms=(
            AsmForm(
                operands=("cond",),
                immediates=(
                    AsmImmediate("true_block", name="target"),
                    AsmImmediate("false_block", name="target"),
                ),
            ),
        ),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.cond_br.i32' asm form 'vm.cond_br.i32' uses immediate name 'target' more than once"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_without_mnemonic() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        mnemonic=None,
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' asm form must specify a mnemonic because the descriptor has no default mnemonic"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_empty_asm_form_mnemonic() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        asm_forms=(AsmForm(mnemonic="", results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' asm form specifies an empty mnemonic"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_missing_schedule_resource() -> None:
    bad_resource_set = replace(IREEVM_CORE_DESCRIPTOR_SET, resources=())

    with pytest.raises(
        ValueError,
        match="schedule class 'vm.alu.i32' references unknown resource 'vm.alu'",
    ):
        generate_descriptor_set(
            bad_resource_set,
            DescriptorAllowlist(keys=("iree.vm.add.i32",)),
        )


def test_generator_emits_enum_immediate_domains() -> None:
    domain = EnumDomain(
        "vm.condition",
        values=(EnumValue("ne", 1), EnumValue("eq", 0)),
    )
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="vm.condition",
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(
        IREEVM_CORE_DESCRIPTOR_SET,
        enum_domains=(domain,),
        descriptors=(descriptor,),
    )

    generated = generate_descriptor_set(descriptor_set)
    manifest = json.loads(generated.manifest_json)

    assert "loom_low_enum_domain_t" in generated.source
    assert "vm.condition" in generated.source
    assert "eq" in generated.source
    assert "ne" in generated.source
    assert manifest["table_counts"]["enum_domains"] == 1
    assert manifest["table_counts"]["enum_values"] == 2


def test_generator_rejects_missing_enum_immediate_domain() -> None:
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain=None,
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.const.i32' enum immediate 'i32_value' has no enum domain"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_unknown_enum_immediate_domain() -> None:
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="missing",
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.const.i32' enum immediate 'i32_value' references unknown enum domain 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_non_enum_immediate_domain() -> None:
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        enum_domain="vm.condition",
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.const.i32' non-enum immediate 'i32_value' references enum domain 'vm.condition'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_missing_schedule_class() -> None:
    bad_descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        schedule_class=cast(str, None),
    )
    bad_set = replace(
        IREEVM_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match="descriptor 'iree.vm.add.i32' has no schedule class",
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_result_after_operand() -> None:
    bad_descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        operands=(
            IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[1],
            IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[0],
        ),
    )
    bad_set = replace(
        IREEVM_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' has result operand 'dst' after non-result operands"),
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_operand_result_role() -> None:
    destructive_result = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[0],
        role=OperandRole.OPERAND_RESULT,
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        operands=(
            destructive_result,
            *IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[1:],
        ),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' operand 'dst' uses OPERAND_RESULT; use separate result and operand rows plus an explicit constraint"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_implicit_operand_without_implicit_flag() -> None:
    implicit_operand = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[1],
        role=OperandRole.IMPLICIT,
        flags=(),
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        operands=(
            IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[0],
            implicit_operand,
            *IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[2:],
        ),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' implicit operand 'lhs' must set the implicit flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_absent_encoding_without_pseudo_flag() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' uses absent encoding id without the pseudo flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_pseudo_flag_with_target_encoding() -> None:
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        flags=(
            *IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].flags,
            DescriptorFlag.PSEUDO,
        ),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' uses the pseudo flag with a target encoding id"),
    ):
        generate_descriptor_set(descriptor_set)
