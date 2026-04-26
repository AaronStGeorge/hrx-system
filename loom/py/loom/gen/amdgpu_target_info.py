# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU target-info overlay -> compact C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[2]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_AMDHSA_TARGET_TRIPLE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE,
    AmdgpuDescriptorSetInfo,
    AmdgpuProcessorInfo,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
)
from loom.target.low_descriptors import descriptor_stable_id  # noqa: E402


def _c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _c_string_arg(value: str) -> str:
    return f'"{_c_string_literal(value)}"'


def _bool_literal(value: bool) -> str:
    return "true" if value else "false"


def _u64_expr(value: int) -> str:
    return f"UINT64_C(0x{value:016x})"


def _padded_arg(value: str, width: int) -> str:
    return f"{value},{' ' * (width - len(value) + 1)}"


def _kernel_descriptor_profile_expr(profile: str) -> str:
    if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
        return "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE"
    if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11:
        return "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11"
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


def _validate_descriptor_sets(descriptor_sets: Sequence[AmdgpuDescriptorSetInfo]) -> None:
    keys = [info.key for info in descriptor_sets]
    if keys != sorted(keys):
        raise ValueError("AMDGPU descriptor-set target-info keys must be sorted")
    if len(keys) != len(set(keys)):
        raise ValueError("AMDGPU descriptor-set target-info keys must be unique")
    stable_ids = [descriptor_stable_id(key) for key in keys]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("AMDGPU descriptor-set target-info stable IDs must be unique")
    for info in descriptor_sets:
        if not info.key:
            raise ValueError("AMDGPU descriptor-set key is required")
        if not info.low_preset_key:
            raise ValueError(f"AMDGPU low preset key is required for {info.key}")
        if info.s_endpgm_opcode < 0 or info.s_endpgm_opcode > 0xFFFF:
            raise ValueError(f"AMDGPU s_endpgm opcode for {info.key} must fit u16")
        _buffer_resource_cache_swizzle_expr(info.buffer_resource_cache_swizzle)
        _vector_memory_cache_policy_encoding_expr(info.vector_memory_cache_policy_encoding)


def _validate_processors(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> None:
    target_cpus = [info.target_cpu for info in processors]
    if target_cpus != sorted(target_cpus):
        raise ValueError("AMDGPU processor target-info keys must be sorted")
    if len(target_cpus) != len(set(target_cpus)):
        raise ValueError("AMDGPU processor target-info keys must be unique")
    descriptor_set_keys = {info.key for info in descriptor_sets}
    for info in processors:
        if not info.target_cpu:
            raise ValueError("AMDGPU target CPU is required")
        if info.descriptor_set_key:
            if info.descriptor_set_key not in descriptor_set_keys:
                raise ValueError(f"AMDGPU processor {info.target_cpu} references unknown descriptor set {info.descriptor_set_key}")
            if not info.low_preset_key:
                raise ValueError(f"AMDGPU low preset key is required for {info.target_cpu}")
        elif info.low_preset_key:
            raise ValueError(f"AMDGPU processor {info.target_cpu} has a low preset key but no descriptor set")
        if info.elf_machine_flags < 0 or info.elf_machine_flags > 0x0FF:
            raise ValueError(f"AMDGPU ELF machine flags for {info.target_cpu} must fit EF_AMDGPU_MACH")
        if info.default_wavefront_size not in (32, 64):
            raise ValueError(f"AMDGPU default wavefront size for {info.target_cpu} must be 32 or 64")
        _matrix_feature_profile_expr(info.matrix_feature_profile)
        if info.kernel_descriptor_profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11 and (
            info.kernel_descriptor_vgpr_encoding_granule_wave32 == 0 or info.kernel_descriptor_vgpr_encoding_granule_wave64 == 0
        ):
            raise ValueError(f"AMDGPU processor {info.target_cpu} has descriptor profile but no VGPR encoding granules")
        if info.kernel_descriptor_profile != AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE and info.elf_machine_flags == 0:
            raise ValueError(f"AMDGPU processor {info.target_cpu} has a kernel descriptor profile but no ELF machine flags")


def _emit_header() -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_H_"
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.amdgpu_target_info"),
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "loom/target/arch/amdgpu/target_info_defs.h"',
        "",
        f"#endif  // {guard}",
    ]
    return "\n".join(lines) + "\n"


def _emit_descriptor_set_rows(descriptor_sets: Sequence[AmdgpuDescriptorSetInfo]) -> list[str]:
    key_width = max(len(_c_string_arg(info.key)) for info in descriptor_sets)
    preset_width = max(len(_c_string_arg(info.low_preset_key)) for info in descriptor_sets)
    opcode_width = len("0x000")
    packet_encoding_width = len("packet_encoding")
    cache_swizzle_width = len("LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT")
    lines = [
        "#define LOOM_AMDGPU_DESCRIPTOR_SET_INFO(stable_id_, descriptor_set_key_, low_preset_key_, s_endpgm_opcode_, s_branch_opcode_, supports_descriptor_packet_encoding_, buffer_resource_cache_swizzle_, vector_memory_cache_policy_encoding_) \\",
        "  { \\",
        "    .descriptor_set_stable_id = stable_id_, \\",
        "    .descriptor_set_key = IREE_SVL(descriptor_set_key_), \\",
        "    .low_preset_key = IREE_SVL(low_preset_key_), \\",
        "    .s_endpgm_opcode = UINT16_C(s_endpgm_opcode_), \\",
        "    .s_branch_opcode = UINT16_C(s_branch_opcode_), \\",
        "    .supports_descriptor_packet_encoding = supports_descriptor_packet_encoding_, \\",
        "    .buffer_resource_cache_swizzle = buffer_resource_cache_swizzle_, \\",
        "    .vector_memory_cache_policy_encoding = vector_memory_cache_policy_encoding_, \\",
        "  }",
        "",
        "static const loom_amdgpu_descriptor_set_info_t kAmdgpuDescriptorSetInfos[] = {",
        "  // stable_id            descriptor_set_key     low_preset_key   s_endpgm s_branch packet_encoding cache_swizzle cache_policy",
    ]
    lines.extend(
        (
            "  LOOM_AMDGPU_DESCRIPTOR_SET_INFO("
            f"{_padded_arg(_u64_expr(descriptor_stable_id(info.key)), len('UINT64_C(0xffffffffffffffff)'))}"
            f"{_padded_arg(_c_string_arg(info.key), key_width)}"
            f"{_padded_arg(_c_string_arg(info.low_preset_key), preset_width)}"
            f"{_padded_arg(f'0x{info.s_endpgm_opcode:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{info.s_branch_opcode:03x}', opcode_width)}"
            f"{_padded_arg(_bool_literal(info.supports_descriptor_packet_encoding), packet_encoding_width)}"
            f"{_padded_arg(_buffer_resource_cache_swizzle_expr(info.buffer_resource_cache_swizzle), cache_swizzle_width)}"
            f"{_vector_memory_cache_policy_encoding_expr(info.vector_memory_cache_policy_encoding)}"
            "),"
        )
        for info in descriptor_sets
    )
    lines.extend(["};", "", "#undef LOOM_AMDGPU_DESCRIPTOR_SET_INFO", ""])
    return lines


def _emit_processor_rows(processors: Sequence[AmdgpuProcessorInfo]) -> list[str]:
    target_cpu_width = max(len(_c_string_arg(info.target_cpu)) for info in processors)
    descriptor_set_width = max(len(_c_string_arg(info.descriptor_set_key)) for info in processors)
    preset_width = max(len(_c_string_arg(info.low_preset_key)) for info in processors)
    machine_flags_width = len("0x000")
    feature_flags_width = len("0x0")
    wavefront_width = 2
    kernel_profile_width = max(len(_kernel_descriptor_profile_expr(info.kernel_descriptor_profile)) for info in processors)
    matrix_profile_width = max(len(_matrix_feature_profile_expr(info.matrix_feature_profile)) for info in processors)
    register_granule_width = 1
    bool_width = len("false")
    lines = [
        "#define LOOM_AMDGPU_PROCESSOR_INFO(target_cpu_, descriptor_set_key_, descriptor_set_stable_id_, low_preset_key_, elf_machine_flags_, elf_feature_flags_, default_wavefront_size_, kernel_descriptor_profile_, matrix_feature_profile_, vgpr_granule_wave32_, vgpr_granule_wave64_, has_flat_scratch_, uses_gfx10_sgpr_, has_dx10_ieee_, has_packed_tid_) \\",
        "  { \\",
        "    .target_cpu = IREE_SVL(target_cpu_), \\",
        "    .descriptor_set_key = IREE_SVL(descriptor_set_key_), \\",
        "    .descriptor_set_stable_id = descriptor_set_stable_id_, \\",
        "    .low_preset_key = IREE_SVL(low_preset_key_), \\",
        "    .elf_machine_flags = UINT32_C(elf_machine_flags_), \\",
        "    .elf_feature_flags = UINT32_C(elf_feature_flags_), \\",
        "    .default_wavefront_size = default_wavefront_size_, \\",
        "    .kernel_descriptor_profile = kernel_descriptor_profile_, \\",
        "    .matrix_feature_profile = matrix_feature_profile_, \\",
        "    .kernel_descriptor_vgpr_encoding_granule_wave32 = vgpr_granule_wave32_, \\",
        "    .kernel_descriptor_vgpr_encoding_granule_wave64 = vgpr_granule_wave64_, \\",
        "    .kernel_descriptor_has_architected_flat_scratch = has_flat_scratch_, \\",
        "    .kernel_descriptor_uses_gfx10_sgpr_encoding = uses_gfx10_sgpr_, \\",
        "    .kernel_descriptor_has_dx10_clamp_and_ieee_mode = has_dx10_ieee_, \\",
        "    .kernel_descriptor_has_packed_workitem_id = has_packed_tid_, \\",
        "  }",
        "",
        "static const loom_amdgpu_processor_info_t kAmdgpuProcessorInfos[] = {",
        "  // target_cpu descriptor_set_key    stable_id            low_preset_key  mach  feat wave kernel_profile                             matrix_profile                             vgpr32 vgpr64 flat_scratch gfx10_sgpr dx10_ieee packed_tid",
    ]
    lines.extend(
        (
            "  LOOM_AMDGPU_PROCESSOR_INFO("
            f"{_padded_arg(_c_string_arg(info.target_cpu), target_cpu_width)}"
            f"{_padded_arg(_c_string_arg(info.descriptor_set_key), descriptor_set_width)}"
            f"{_padded_arg(_u64_expr(descriptor_stable_id(info.descriptor_set_key)) if info.descriptor_set_key else _u64_expr(0), len('UINT64_C(0xffffffffffffffff)'))}"
            f"{_padded_arg(_c_string_arg(info.low_preset_key), preset_width)}"
            f"{_padded_arg(f'0x{info.elf_machine_flags:03x}', machine_flags_width)}"
            f"{_padded_arg(f'0x{info.elf_feature_flags:x}', feature_flags_width)}"
            f"{_padded_arg(str(info.default_wavefront_size), wavefront_width)}"
            f"{_padded_arg(_kernel_descriptor_profile_expr(info.kernel_descriptor_profile), kernel_profile_width)}"
            f"{_padded_arg(_matrix_feature_profile_expr(info.matrix_feature_profile), matrix_profile_width)}"
            f"{_padded_arg(str(info.kernel_descriptor_vgpr_encoding_granule_wave32), register_granule_width)}"
            f"{_padded_arg(str(info.kernel_descriptor_vgpr_encoding_granule_wave64), register_granule_width)}"
            f"{_padded_arg(_bool_literal(info.kernel_descriptor_has_architected_flat_scratch), bool_width)}"
            f"{_padded_arg(_bool_literal(info.kernel_descriptor_uses_gfx10_sgpr_encoding), bool_width)}"
            f"{_padded_arg(_bool_literal(info.kernel_descriptor_has_dx10_clamp_and_ieee_mode), bool_width)}"
            f"{_bool_literal(info.kernel_descriptor_has_packed_workitem_id)}),"
        )
        for info in processors
    )
    lines.extend(["};", "", "#undef LOOM_AMDGPU_PROCESSOR_INFO", ""])
    return lines


def _emit_source(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.amdgpu_target_info"),
        "",
        '#include "loom/target/arch/amdgpu/target_info.h"',
        "",
        "#include <inttypes.h>",
        "#include <stdint.h>",
        "",
        f'static const iree_string_view_t kAmdgpuAmdhsaTargetIdPrefix = IREE_SVL("{_c_string_literal(AMDGPU_AMDHSA_TARGET_TRIPLE)}--");',
        "",
        "// clang-format off",
    ]
    lines.extend(_emit_descriptor_set_rows(descriptor_sets))
    lines.extend(_emit_processor_rows(processors))
    lines.append("// clang-format on")
    lines.append("")
    lines.extend(
        [
            "iree_host_size_t loom_amdgpu_target_info_processor_count(void) {",
            "  return IREE_ARRAYSIZE(kAmdgpuProcessorInfos);",
            "}",
            "",
            "const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_processor_at(",
            "    iree_host_size_t index) {",
            "  if (index >= IREE_ARRAYSIZE(kAmdgpuProcessorInfos)) {",
            "    return NULL;",
            "  }",
            "  return &kAmdgpuProcessorInfos[index];",
            "}",
            "",
            "iree_status_t loom_amdgpu_target_info_lookup_processor(",
            "    iree_string_view_t target_cpu,",
            "    const loom_amdgpu_processor_info_t** out_processor) {",
            "  IREE_ASSERT_ARGUMENT(out_processor);",
            "  *out_processor = NULL;",
            "  if (iree_string_view_is_empty(target_cpu)) {",
            "    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,",
            '                            "AMDGPU target CPU is required");',
            "  }",
            "  iree_host_size_t low = 0;",
            "  iree_host_size_t high = IREE_ARRAYSIZE(kAmdgpuProcessorInfos);",
            "  while (low < high) {",
            "    const iree_host_size_t mid = low + (high - low) / 2;",
            "    const loom_amdgpu_processor_info_t* processor =",
            "        &kAmdgpuProcessorInfos[mid];",
            "    const int comparison =",
            "        iree_string_view_compare(processor->target_cpu, target_cpu);",
            "    if (comparison == 0) {",
            "      *out_processor = processor;",
            "      return iree_ok_status();",
            "    }",
            "    if (comparison < 0) {",
            "      low = mid + 1;",
            "    } else {",
            "      high = mid;",
            "    }",
            "  }",
            "  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,",
            "                          \"AMDGPU target CPU '%.*s' is not supported\",",
            "                          (int)target_cpu.size, target_cpu.data);",
            "}",
            "",
            "iree_status_t loom_amdgpu_target_info_lookup_descriptor_set(",
            "    iree_string_view_t descriptor_set_key,",
            "    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set) {",
            "  IREE_ASSERT_ARGUMENT(out_descriptor_set);",
            "  *out_descriptor_set = NULL;",
            "  if (iree_string_view_is_empty(descriptor_set_key)) {",
            "    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,",
            '                            "AMDGPU descriptor set key is required");',
            "  }",
            "  iree_host_size_t low = 0;",
            "  iree_host_size_t high = IREE_ARRAYSIZE(kAmdgpuDescriptorSetInfos);",
            "  while (low < high) {",
            "    const iree_host_size_t mid = low + (high - low) / 2;",
            "    const loom_amdgpu_descriptor_set_info_t* descriptor_set =",
            "        &kAmdgpuDescriptorSetInfos[mid];",
            "    const int comparison = iree_string_view_compare(",
            "        descriptor_set->descriptor_set_key, descriptor_set_key);",
            "    if (comparison == 0) {",
            "      *out_descriptor_set = descriptor_set;",
            "      return iree_ok_status();",
            "    }",
            "    if (comparison < 0) {",
            "      low = mid + 1;",
            "    } else {",
            "      high = mid;",
            "    }",
            "  }",
            "  return iree_make_status(",
            "      IREE_STATUS_UNIMPLEMENTED,",
            "      \"AMDGPU descriptor set '%.*s' is not supported by native emission\",",
            "      (int)descriptor_set_key.size, descriptor_set_key.data);",
            "}",
            "",
            "const loom_amdgpu_descriptor_set_info_t*",
            "loom_amdgpu_target_info_descriptor_set_by_id(",
            "    uint64_t descriptor_set_stable_id) {",
            "  switch (descriptor_set_stable_id) {",
        ]
    )
    for index, info in enumerate(descriptor_sets):
        lines.extend(
            [
                f"    case {_u64_expr(descriptor_stable_id(info.key))}:",
                f"      return &kAmdgpuDescriptorSetInfos[{index}];",
            ]
        )
    lines.extend(
        [
            "    default:",
            "      break;",
            "  }",
            "  return NULL;",
            "}",
            "",
            "iree_status_t loom_amdgpu_target_info_lookup_descriptor_set_by_id(",
            "    uint64_t descriptor_set_stable_id,",
            "    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set) {",
            "  IREE_ASSERT_ARGUMENT(out_descriptor_set);",
            "  *out_descriptor_set = NULL;",
            "  if (descriptor_set_stable_id == 0) {",
            "    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,",
            '                            "AMDGPU descriptor set stable ID is required");',
            "  }",
            "  const loom_amdgpu_descriptor_set_info_t* descriptor_set =",
            "      loom_amdgpu_target_info_descriptor_set_by_id(",
            "          descriptor_set_stable_id);",
            "  if (descriptor_set != NULL) {",
            "    *out_descriptor_set = descriptor_set;",
            "    return iree_ok_status();",
            "  }",
            "  return iree_make_status(",
            "      IREE_STATUS_UNIMPLEMENTED,",
            '      "AMDGPU descriptor set stable ID 0x%016" PRIx64',
            '      " is not supported by native emission",',
            "      descriptor_set_stable_id);",
            "}",
            "",
            "static iree_status_t loom_amdgpu_target_info_validate_target_id_chars(",
            "    iree_string_view_t target_id) {",
            "  for (iree_host_size_t i = 0; i < target_id.size; ++i) {",
            "    const unsigned char c = (unsigned char)target_id.data[i];",
            "    if (c <= ' ' || c == '\"' || c == '\\\\') {",
            "      return iree_make_status(",
            "          IREE_STATUS_INVALID_ARGUMENT,",
            '          "AMDGPU AMDHSA target id contains an unsupported character");',
            "    }",
            "  }",
            "  return iree_ok_status();",
            "}",
            "",
            "iree_status_t loom_amdgpu_target_info_parse_amdhsa_target_id(",
            "    iree_string_view_t target_id,",
            "    loom_amdgpu_amdhsa_target_id_t* out_target_id) {",
            "  IREE_ASSERT_ARGUMENT(out_target_id);",
            "  *out_target_id = (loom_amdgpu_amdhsa_target_id_t){0};",
            "  if (iree_string_view_is_empty(target_id)) {",
            "    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,",
            '                            "AMDGPU AMDHSA target id is required");',
            "  }",
            "  IREE_RETURN_IF_ERROR(",
            "      loom_amdgpu_target_info_validate_target_id_chars(target_id));",
            "  iree_string_view_t processor_and_features = target_id;",
            "  if (!iree_string_view_consume_prefix(",
            "          &processor_and_features, kAmdgpuAmdhsaTargetIdPrefix)) {",
            "    return iree_make_status(",
            "        IREE_STATUS_INVALID_ARGUMENT,",
            "        \"AMDGPU AMDHSA target id '%.*s' does not start with '%.*s'\",",
            "        (int)target_id.size, target_id.data,",
            "        (int)kAmdgpuAmdhsaTargetIdPrefix.size,",
            "        kAmdgpuAmdhsaTargetIdPrefix.data);",
            "  }",
            "  iree_string_view_t target_cpu = iree_string_view_empty();",
            "  iree_string_view_t feature_suffix = iree_string_view_empty();",
            "  const intptr_t split = iree_string_view_split(",
            "      processor_and_features, ':', &target_cpu, &feature_suffix);",
            "  if (split == -1) {",
            "    target_cpu = processor_and_features;",
            "  } else if (iree_string_view_is_empty(feature_suffix)) {",
            "    return iree_make_status(",
            "        IREE_STATUS_INVALID_ARGUMENT,",
            "        \"AMDGPU AMDHSA target id '%.*s' has an empty feature suffix\",",
            "        (int)target_id.size, target_id.data);",
            "  }",
            "  const loom_amdgpu_processor_info_t* processor = NULL;",
            "  IREE_RETURN_IF_ERROR(",
            "      loom_amdgpu_target_info_lookup_processor(target_cpu, &processor));",
            "  *out_target_id = (loom_amdgpu_amdhsa_target_id_t){",
            "      .processor = processor,",
            "      .feature_suffix = feature_suffix,",
            "  };",
            "  return iree_ok_status();",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_target_info_to_paths(header_path: Path, source_path: Path) -> None:
    descriptor_sets = sorted_descriptor_set_infos()
    processors = sorted_processor_infos()
    _validate_descriptor_sets(descriptor_sets)
    _validate_processors(processors, descriptor_sets)
    header = _emit_header()
    source = _emit_source(processors, descriptor_sets)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")


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
    args = parser.parse_args(argv)

    write_target_info_to_paths(header_path=args.header, source_path=args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
