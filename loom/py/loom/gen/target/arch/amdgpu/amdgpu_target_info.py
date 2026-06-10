# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU target-info overlay -> compact C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import c_string_arg as _c_string_arg  # noqa: E402
from loom.gen.support.c import c_string_literal as _c_string_literal  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    AmdgpuIsaFactSource,
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_AMDHSA_TARGET_TRIPLE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE,
    AmdgpuDescriptorSetInfo,
    AmdgpuProcessorInfo,
    amdgpu_descriptor_set_ordinal,
    kernel_descriptor_profile_supports_wavefront_size,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    validate_amdgpu_descriptor_set_isa_xml,
)


@dataclass(frozen=True, slots=True)
class _AmdgpuDescriptorSetRow:
    info: AmdgpuDescriptorSetInfo
    s_nop_opcode: int
    s_endpgm_opcode: int
    s_branch_opcode: int
    s_cbranch_scc0_opcode: int
    s_cbranch_scc1_opcode: int


def _bool_literal(value: bool) -> str:
    return "true" if value else "false"


def _u16_expr(value: int) -> str:
    return f"UINT16_C({value})"


def _padded_arg(value: str, width: int) -> str:
    return f"{value},{' ' * (width - len(value) + 1)}"


def _descriptor_set_ordinal_suffix(key: str) -> str:
    prefix = "amdgpu."
    suffix = ".core"
    if not key.startswith(prefix) or not key.endswith(suffix):
        raise ValueError(f"AMDGPU descriptor-set key '{key}' must be a core key")
    return key.removeprefix(prefix).removesuffix(suffix).replace(".", "_").upper()


def _descriptor_set_ordinal_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_{_descriptor_set_ordinal_suffix(key)}"


def _kernel_descriptor_profile_expr(profile: str) -> str:
    if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
        return "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE"
    if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9:
        return "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9"
    if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11:
        return "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11"
    if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12:
        return "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12"
    if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125:
        return "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125"
    raise ValueError(f"unknown AMDGPU kernel descriptor profile '{profile}'")


def _matrix_feature_profile_expr(profile: str) -> str:
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_NONE:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE"
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908"
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A"
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940"
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950"
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11"
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12"
    if profile == AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250:
        return "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250"
    raise ValueError(f"unknown AMDGPU matrix feature profile '{profile}'")


def _buffer_resource_cache_swizzle_expr(kind: str) -> str:
    if kind == AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE:
        return "LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE"
    if kind == AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT:
        return "LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT"
    raise ValueError(f"unknown AMDGPU buffer-resource cache swizzle kind '{kind}'")


def _vector_memory_cache_policy_encoding_expr(kind: str) -> str:
    if kind == AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
        return "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE"
    if kind == AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
        return "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC"
    if kind == AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
        return "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH"
    if kind == AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
        return "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1"
    raise ValueError(f"unknown AMDGPU vector-memory cache-policy encoding '{kind}'")


def _parse_isa_xml_argument(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition(":")
    if not separator or not key or not path:
        raise ValueError("AMDGPU target-info --isa-xml entries must be key:path pairs")
    return key, Path(path)


def _parse_isa_xml_arguments(
    values: Sequence[str],
) -> dict[str, AmdgpuIsaFactSource]:
    specs: dict[str, AmdgpuIsaFactSource] = {}
    for value in values:
        key, path = _parse_isa_xml_argument(value)
        if key in specs:
            raise ValueError(f"AMDGPU target-info ISA XML key '{key}' is duplicate")
        specs[key] = parse_amdgpu_isa_xml_path(path)
    return specs


def _sopp_opcode(spec: AmdgpuIsaFactSource, instruction_name: str) -> int:
    summaries = tuple(
        summary for summary in spec.instruction_encoding_summaries((instruction_name,), include_aliases=False) if summary.encoding_name == "ENC_SOPP" and summary.condition_name == "default"
    )
    if len(summaries) != 1:
        raise ValueError(f"{spec.source_name}: expected one default ENC_SOPP encoding for {instruction_name}, found {len(summaries)}")
    return summaries[0].opcode


def _materialize_descriptor_set_rows(
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
) -> tuple[_AmdgpuDescriptorSetRow, ...]:
    rows: list[_AmdgpuDescriptorSetRow] = []
    for info in descriptor_sets:
        spec = isa_specs.get(info.isa_xml_key)
        if spec is None:
            raise ValueError(f"AMDGPU descriptor set {info.key} references missing ISA XML key '{info.isa_xml_key}'")
        validate_amdgpu_descriptor_set_isa_xml(info, spec)
        rows.append(
            _AmdgpuDescriptorSetRow(
                info=info,
                s_nop_opcode=_sopp_opcode(spec, "S_NOP"),
                s_endpgm_opcode=_sopp_opcode(spec, "S_ENDPGM"),
                s_branch_opcode=_sopp_opcode(spec, "S_BRANCH"),
                s_cbranch_scc0_opcode=_sopp_opcode(spec, "S_CBRANCH_SCC0"),
                s_cbranch_scc1_opcode=_sopp_opcode(spec, "S_CBRANCH_SCC1"),
            )
        )
    return tuple(rows)


def _validate_descriptor_sets(descriptor_sets: Sequence[AmdgpuDescriptorSetInfo]) -> None:
    keys = [info.key for info in descriptor_sets]
    if keys != sorted(keys):
        raise ValueError("AMDGPU descriptor-set target-info keys must be sorted")
    if len(keys) != len(set(keys)):
        raise ValueError("AMDGPU descriptor-set target-info keys must be unique")
    if len(keys) >= AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE:
        raise ValueError("AMDGPU descriptor-set ordinals must fit uint16_t")
    generator_targets = [info.generator_target for info in descriptor_sets]
    if len(generator_targets) != len(set(generator_targets)):
        raise ValueError("AMDGPU descriptor generator targets must be unique")
    infos_by_generator_target = {info.generator_target: info for info in descriptor_sets}
    for info in descriptor_sets:
        if not info.generator_target:
            raise ValueError("AMDGPU descriptor generator target is required")
        if not info.key:
            raise ValueError("AMDGPU descriptor-set key is required")
        if not info.isa_xml_key:
            raise ValueError(f"AMDGPU ISA XML key is required for {info.key}")
        if not info.isa_architecture_name:
            raise ValueError(f"AMDGPU ISA XML architecture name is required for {info.key}")
        if info.isa_architecture_id <= 0:
            raise ValueError(f"AMDGPU ISA XML architecture id is required for {info.key}")
        if info.storage_generator_target is not None:
            if not info.storage_generator_target:
                raise ValueError(f"AMDGPU storage generator target is required for {info.key}")
            if info.storage_generator_target == info.generator_target:
                raise ValueError(f"AMDGPU descriptor set {info.key} cannot store itself as a view")
            storage_info = infos_by_generator_target.get(info.storage_generator_target)
            if storage_info is None:
                raise ValueError(f"AMDGPU descriptor set {info.key} references unknown storage generator target '{info.storage_generator_target}'")
            if storage_info.storage_generator_target is not None:
                raise ValueError(f"AMDGPU descriptor set {info.key} uses view-only target '{storage_info.generator_target}' as storage")
            if storage_info.isa_xml_key != info.isa_xml_key:
                raise ValueError(f"AMDGPU descriptor set {info.key} storage target '{storage_info.generator_target}' uses ISA XML key '{storage_info.isa_xml_key}', expected '{info.isa_xml_key}'")
        _buffer_resource_cache_swizzle_expr(info.buffer_resource_cache_swizzle)
        _vector_memory_cache_policy_encoding_expr(info.vector_memory_cache_policy_encoding)


def _validate_descriptor_set_rows(rows: Sequence[_AmdgpuDescriptorSetRow]) -> None:
    for row in rows:
        if row.s_nop_opcode < 0 or row.s_nop_opcode > 0xFFFF:
            raise ValueError(f"AMDGPU s_nop opcode for {row.info.key} must fit u16")
        if row.s_endpgm_opcode < 0 or row.s_endpgm_opcode > 0xFFFF:
            raise ValueError(f"AMDGPU s_endpgm opcode for {row.info.key} must fit u16")
        if row.s_branch_opcode < 0 or row.s_branch_opcode > 0xFFFF:
            raise ValueError(f"AMDGPU s_branch opcode for {row.info.key} must fit u16")
        if row.s_cbranch_scc0_opcode < 0 or row.s_cbranch_scc0_opcode > 0xFFFF:
            raise ValueError(f"AMDGPU s_cbranch_scc0 opcode for {row.info.key} must fit u16")
        if row.s_cbranch_scc1_opcode < 0 or row.s_cbranch_scc1_opcode > 0xFFFF:
            raise ValueError(f"AMDGPU s_cbranch_scc1 opcode for {row.info.key} must fit u16")


def _validate_processors(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> None:
    processor_names = [info.processor for info in processors]
    if processor_names != sorted(processor_names):
        raise ValueError("AMDGPU processor target-info keys must be sorted")
    if len(processor_names) != len(set(processor_names)):
        raise ValueError("AMDGPU processor target-info keys must be unique")
    descriptor_set_keys = {info.key for info in descriptor_sets}
    for info in processors:
        if not info.processor:
            raise ValueError("AMDGPU processor is required")
        if info.descriptor_set_key and info.descriptor_set_key not in descriptor_set_keys:
            raise ValueError(f"AMDGPU processor {info.processor} references unknown descriptor set {info.descriptor_set_key}")
        if info.elf_machine_flags < 0 or info.elf_machine_flags > 0x0FF:
            raise ValueError(f"AMDGPU ELF machine flags for {info.processor} must fit EF_AMDGPU_MACH")
        if info.elf_feature_flags < 0 or info.elf_feature_flags > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU ELF feature flags for {info.processor} must fit u32")
        if info.elf_feature_flags & 0x0FF:
            raise ValueError(f"AMDGPU ELF feature flags for {info.processor} must not overlap EF_AMDGPU_MACH")
        if info.default_wavefront_size not in (32, 64):
            raise ValueError(f"AMDGPU default wavefront size for {info.processor} must be 32 or 64")
        if info.kernel_descriptor_profile != AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE and not kernel_descriptor_profile_supports_wavefront_size(
            info.kernel_descriptor_profile, info.default_wavefront_size
        ):
            raise ValueError(f"AMDGPU default wavefront size for {info.processor} is not supported by its kernel descriptor profile")
        _matrix_feature_profile_expr(info.matrix_feature_profile)
        if info.kernel_descriptor_profile != AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE and (
            info.kernel_descriptor_vgpr_encoding_granule_wave32 == 0 or info.kernel_descriptor_vgpr_encoding_granule_wave64 == 0
        ):
            raise ValueError(f"AMDGPU processor {info.processor} has descriptor profile but no VGPR encoding granules")
        if info.kernel_descriptor_profile != AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE and info.elf_machine_flags == 0:
            raise ValueError(f"AMDGPU processor {info.processor} has a kernel descriptor profile but no ELF machine flags")
        if info.scheduling_bits < 0 or info.scheduling_bits > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU scheduling bits for {info.processor} must fit u32")
        unknown_scheduling_bits = info.scheduling_bits & ~AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS
        if unknown_scheduling_bits != 0:
            raise ValueError(f"AMDGPU processor {info.processor} has unknown scheduling bits 0x{unknown_scheduling_bits:x}")


def _emit_header(descriptor_sets: Sequence[AmdgpuDescriptorSetInfo]) -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_H_"
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "loom/target/arch/amdgpu/target_info_defs.h"',
        "",
        "// Generated dense descriptor-set ordinals.",
    ]
    lines.extend(f"#define {_descriptor_set_ordinal_constant_name(info.key)} {_u16_expr(amdgpu_descriptor_set_ordinal(info.key))}" for info in descriptor_sets)
    lines.extend(
        [
            f"#define LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT {_u16_expr(len(descriptor_sets))}",
            "",
        ]
    )
    lines.extend(
        [
            f"#endif  // {guard}",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_tables_header() -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_TABLES_H_"
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "loom/target/arch/amdgpu/target_info_defs.h"',
        "",
        "extern const iree_string_view_t",
        "    loom_amdgpu_target_info_amdhsa_target_id_prefix;",
        "",
        "extern const loom_amdgpu_descriptor_set_info_t",
        "    loom_amdgpu_target_info_descriptor_set_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_descriptor_set_info_count;",
        "",
        "extern const loom_amdgpu_processor_info_t",
        "    loom_amdgpu_target_info_processor_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_processor_info_count;",
        "",
        f"#endif  // {guard}",
    ]
    return "\n".join(lines) + "\n"


def _emit_descriptor_set_rows(rows: Sequence[_AmdgpuDescriptorSetRow]) -> list[str]:
    key_width = max(len(_c_string_arg(row.info.key)) for row in rows)
    ordinal_width = len("UINT16_C(65535)")
    opcode_width = len("0x000")
    packet_encoding_width = len("packet_encoding")
    cache_swizzle_width = len("LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT")
    lines = [
        "#define LOOM_AMDGPU_DESCRIPTOR_SET_INFO(ordinal_, descriptor_set_key_, s_nop_opcode_, s_endpgm_opcode_, s_branch_opcode_, s_cbranch_scc0_opcode_, s_cbranch_scc1_opcode_, supports_descriptor_packet_encoding_, buffer_resource_cache_swizzle_, vector_memory_cache_policy_encoding_) \\",
        "  { \\",
        "    .descriptor_set_ordinal = ordinal_, \\",
        "    .descriptor_set_key = IREE_SVL(descriptor_set_key_), \\",
        "    .s_nop_opcode = UINT16_C(s_nop_opcode_), \\",
        "    .s_endpgm_opcode = UINT16_C(s_endpgm_opcode_), \\",
        "    .s_branch_opcode = UINT16_C(s_branch_opcode_), \\",
        "    .s_cbranch_scc0_opcode = UINT16_C(s_cbranch_scc0_opcode_), \\",
        "    .s_cbranch_scc1_opcode = UINT16_C(s_cbranch_scc1_opcode_), \\",
        "    .supports_descriptor_packet_encoding = supports_descriptor_packet_encoding_, \\",
        "    .buffer_resource_cache_swizzle = buffer_resource_cache_swizzle_, \\",
        "    .vector_memory_cache_policy_encoding = vector_memory_cache_policy_encoding_, \\",
        "  }",
        "",
        "const loom_amdgpu_descriptor_set_info_t loom_amdgpu_target_info_descriptor_set_infos[] = {",
        "  // ordinal         descriptor_set_key     s_nop s_endpgm s_branch s_cbranch_scc0 s_cbranch_scc1 packet_encoding cache_swizzle cache_policy",
    ]
    lines.extend(
        (
            "  LOOM_AMDGPU_DESCRIPTOR_SET_INFO("
            f"{_padded_arg(_u16_expr(amdgpu_descriptor_set_ordinal(info.key)), ordinal_width)}"
            f"{_padded_arg(_c_string_arg(info.key), key_width)}"
            f"{_padded_arg(f'0x{row.s_nop_opcode:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.s_endpgm_opcode:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.s_branch_opcode:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.s_cbranch_scc0_opcode:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.s_cbranch_scc1_opcode:03x}', opcode_width)}"
            f"{_padded_arg(_bool_literal(info.supports_descriptor_packet_encoding), packet_encoding_width)}"
            f"{_padded_arg(_buffer_resource_cache_swizzle_expr(info.buffer_resource_cache_swizzle), cache_swizzle_width)}"
            f"{_vector_memory_cache_policy_encoding_expr(info.vector_memory_cache_policy_encoding)}"
            "),"
        )
        for row in rows
        for info in (row.info,)
    )
    lines.extend(["};", "", "#undef LOOM_AMDGPU_DESCRIPTOR_SET_INFO", ""])
    return lines


def _emit_processor_rows(processors: Sequence[AmdgpuProcessorInfo]) -> list[str]:
    processor_width = max(len(_c_string_arg(info.processor)) for info in processors)
    descriptor_set_width = max(len(_c_string_arg(info.descriptor_set_key)) for info in processors)
    ordinal_width = len("UINT16_C(65535)")
    machine_flags_width = len("0x000")
    feature_flags_width = len("0x0")
    wavefront_width = 2
    kernel_profile_width = max(len(_kernel_descriptor_profile_expr(info.kernel_descriptor_profile)) for info in processors)
    matrix_profile_width = max(len(_matrix_feature_profile_expr(info.matrix_feature_profile)) for info in processors)
    scheduling_width = max(len(f"0x{info.scheduling_bits:03x}") for info in processors)
    register_granule_width = 1
    bool_width = len("false")
    lines = [
        "#define LOOM_AMDGPU_PROCESSOR_INFO(processor_, descriptor_set_key_, descriptor_set_ordinal_, elf_machine_flags_, elf_feature_flags_, default_wavefront_size_, kernel_descriptor_profile_, matrix_feature_profile_, scheduling_bits_, vgpr_granule_wave32_, vgpr_granule_wave64_, has_flat_scratch_, uses_gfx10_sgpr_, has_accum_offset_, has_dx10_ieee_, has_packed_tid_) \\",
        "  { \\",
        "    .processor = IREE_SVL(processor_), \\",
        "    .descriptor_set_key = IREE_SVL(descriptor_set_key_), \\",
        "    .descriptor_set_ordinal = descriptor_set_ordinal_, \\",
        "    .elf_machine_flags = UINT32_C(elf_machine_flags_), \\",
        "    .elf_feature_flags = UINT32_C(elf_feature_flags_), \\",
        "    .default_wavefront_size = default_wavefront_size_, \\",
        "    .kernel_descriptor_profile = kernel_descriptor_profile_, \\",
        "    .matrix_feature_profile = matrix_feature_profile_, \\",
        "    .scheduling_bits = UINT32_C(scheduling_bits_), \\",
        "    .kernel_descriptor_vgpr_encoding_granule_wave32 = vgpr_granule_wave32_, \\",
        "    .kernel_descriptor_vgpr_encoding_granule_wave64 = vgpr_granule_wave64_, \\",
        "    .kernel_descriptor_has_architected_flat_scratch = has_flat_scratch_, \\",
        "    .kernel_descriptor_uses_gfx10_sgpr_encoding = uses_gfx10_sgpr_, \\",
        "    .kernel_descriptor_has_accum_offset = has_accum_offset_, \\",
        "    .kernel_descriptor_has_dx10_clamp_and_ieee_mode = has_dx10_ieee_, \\",
        "    .kernel_descriptor_has_packed_workitem_id = has_packed_tid_, \\",
        "  }",
        "",
        "const loom_amdgpu_processor_info_t loom_amdgpu_target_info_processor_infos[] = {",
        "  // processor descriptor_set_key    ordinal         mach  feat wave kernel_profile                              matrix_profile                             sched vgpr32 vgpr64 flat_scratch gfx10_sgpr accum_offset dx10_ieee packed_tid",
    ]
    lines.extend(
        (
            "  LOOM_AMDGPU_PROCESSOR_INFO("
            f"{_padded_arg(_c_string_arg(info.processor), processor_width)}"
            f"{_padded_arg(_c_string_arg(info.descriptor_set_key), descriptor_set_width)}"
            f"{_padded_arg(_u16_expr(amdgpu_descriptor_set_ordinal(info.descriptor_set_key)) if info.descriptor_set_key else 'LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE', ordinal_width)}"
            f"{_padded_arg(f'0x{info.elf_machine_flags:03x}', machine_flags_width)}"
            f"{_padded_arg(f'0x{info.elf_feature_flags:x}', feature_flags_width)}"
            f"{_padded_arg(str(info.default_wavefront_size), wavefront_width)}"
            f"{_padded_arg(_kernel_descriptor_profile_expr(info.kernel_descriptor_profile), kernel_profile_width)}"
            f"{_padded_arg(_matrix_feature_profile_expr(info.matrix_feature_profile), matrix_profile_width)}"
            f"{_padded_arg(f'0x{info.scheduling_bits:03x}', scheduling_width)}"
            f"{_padded_arg(str(info.kernel_descriptor_vgpr_encoding_granule_wave32), register_granule_width)}"
            f"{_padded_arg(str(info.kernel_descriptor_vgpr_encoding_granule_wave64), register_granule_width)}"
            f"{_padded_arg(_bool_literal(info.kernel_descriptor_has_architected_flat_scratch), bool_width)}"
            f"{_padded_arg(_bool_literal(info.kernel_descriptor_uses_gfx10_sgpr_encoding), bool_width)}"
            f"{_padded_arg(_bool_literal(info.kernel_descriptor_has_accum_offset), bool_width)}"
            f"{_padded_arg(_bool_literal(info.kernel_descriptor_has_dx10_clamp_and_ieee_mode), bool_width)}"
            f"{_bool_literal(info.kernel_descriptor_has_packed_workitem_id)}),"
        )
        for info in processors
    )
    lines.extend(["};", "", "#undef LOOM_AMDGPU_PROCESSOR_INFO", ""])
    return lines


def _emit_tables_source(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_set_rows: Sequence[_AmdgpuDescriptorSetRow],
) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
        '#include "loom/target/arch/amdgpu/target_info_tables.h"',
        "",
        "#include <stdint.h>",
        "",
        f'const iree_string_view_t loom_amdgpu_target_info_amdhsa_target_id_prefix = IREE_SVL("{_c_string_literal(AMDGPU_AMDHSA_TARGET_TRIPLE)}--");',
        "",
        "// clang-format off",
    ]
    lines.extend(_emit_descriptor_set_rows(descriptor_set_rows))
    lines.extend(_emit_processor_rows(processors))
    lines.append("// clang-format on")
    lines.append("")
    lines.extend(
        [
            "const iree_host_size_t",
            "    loom_amdgpu_target_info_descriptor_set_info_count =",
            "        IREE_ARRAYSIZE(loom_amdgpu_target_info_descriptor_set_infos);",
            "",
            "const iree_host_size_t",
            "    loom_amdgpu_target_info_processor_info_count =",
            "        IREE_ARRAYSIZE(loom_amdgpu_target_info_processor_infos);",
        ]
    )
    return "\n".join(lines) + "\n"


def write_target_info_to_paths(
    header_path: Path,
    source_path: Path,
    tables_header_path: Path,
    isa_xml_arguments: Sequence[str],
) -> None:
    descriptor_sets = sorted_descriptor_set_infos()
    processors = sorted_processor_infos()
    isa_specs = _parse_isa_xml_arguments(isa_xml_arguments)
    descriptor_set_rows = _materialize_descriptor_set_rows(descriptor_sets, isa_specs)
    _validate_descriptor_sets(descriptor_sets)
    _validate_descriptor_set_rows(descriptor_set_rows)
    _validate_processors(processors, descriptor_sets)
    header = _emit_header(descriptor_sets)
    tables_header = _emit_tables_header()
    source = _emit_tables_source(processors, descriptor_set_rows)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    tables_header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    tables_header_path.write_text(tables_header, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU target-info C tables from Loom overlay data.")
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated target-info header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated target-info source path.",
    )
    parser.add_argument(
        "--tables-header",
        required=True,
        type=Path,
        help="Generated target-info private table header path.",
    )
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as key:path.",
    )
    args = parser.parse_args(argv)

    write_target_info_to_paths(
        header_path=args.header,
        source_path=args.source,
        tables_header_path=args.tables_header,
        isa_xml_arguments=args.isa_xml,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
