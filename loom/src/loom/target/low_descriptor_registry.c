// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"
#include "loom/target/arch/amdgpu/gfx12_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"
#include "loom/target/arch/wasm/descriptors.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

#define LOOM_TARGET_LOW_X86_64_TRIPLE IREE_SVL("x86_64-unknown-linux-gnu")
#define LOOM_TARGET_LOW_X86_64_DATA_LAYOUT          \
  IREE_SVL(                                         \
      "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:" \
      "64-i128:128-f80:128-n8:16:32:64-S128")
#define LOOM_TARGET_LOW_X86_64_PACKED_DOT_FEATURES \
  IREE_SVL("+avx512bf16,+avx512vl,+avxvnni,+avxvnniint8")

#define LOOM_TARGET_LOW_AMDGPU_TRIPLE IREE_SVL("amdgcn-amd-amdhsa")
#define LOOM_TARGET_LOW_AMDGPU_DATA_LAYOUT                       \
  IREE_SVL(                                                      \
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-"  \
      "p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:" \
      "256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"  \
      "256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-"  \
      "A5-G1-ni:7:8:9")

static const loom_target_snapshot_t kIreeVmSnapshot = {
    .name = IREE_SVL("iree-vm"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_VM,
    .target_triple = IREE_SVL("iree-vm"),
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = 0,
            .constant = 0,
            .private_memory = 0,
            .host = 0,
            .descriptor = 0,
        },
};

static const loom_target_export_plan_t kIreeVmExportPlan = {
    .name = IREE_SVL("iree-vm-function"),
    .abi_kind = LOOM_TARGET_ABI_VM_MODULE_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kIreeVmConfig = {
    .name = IREE_SVL("iree.vm.core"),
    .contract_set_key = IREE_SVL("iree.vm.core"),
};

static const loom_target_bundle_t kIreeVmBundle = {
    .name = IREE_SVL("iree-vm"),
    .snapshot = &kIreeVmSnapshot,
    .export_plan = &kIreeVmExportPlan,
    .config = &kIreeVmConfig,
};

static const loom_target_snapshot_t kWasmSimd128Snapshot = {
    .name = IREE_SVL("wasm32-simd128"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_WASM,
    .target_triple = IREE_SVL("wasm32-unknown-unknown"),
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY,
    .target_features = IREE_SVL("+simd128"),
    .default_pointer_bitwidth = 32,
    .index_bitwidth = 32,
    .offset_bitwidth = 32,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = UINT32_MAX,
            .constant = 0,
            .private_memory = UINT32_MAX,
            .host = UINT32_MAX,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kWasmFunctionExportPlan = {
    .name = IREE_SVL("wasm-function"),
    .abi_kind = LOOM_TARGET_ABI_WASM_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kWasmSimd128Config = {
    .name = IREE_SVL("wasm.core.simd128"),
    .contract_set_key = IREE_SVL("wasm.core.simd128"),
};

static const loom_target_bundle_t kWasmSimd128Bundle = {
    .name = IREE_SVL("wasm32-simd128"),
    .snapshot = &kWasmSimd128Snapshot,
    .export_plan = &kWasmFunctionExportPlan,
    .config = &kWasmSimd128Config,
};

static const loom_target_snapshot_t kX86Avx512Snapshot = {
    .name = IREE_SVL("x86_64-avx512"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .target_triple = LOOM_TARGET_LOW_X86_64_TRIPLE,
    .data_layout = LOOM_TARGET_LOW_X86_64_DATA_LAYOUT,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .target_cpu = IREE_SVL("x86-64-v4"),
    .target_features = IREE_SVL("+avx512f,+avx512bw,+avx512dq,+avx512vl"),
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = UINT32_MAX,
            .constant = 0,
            .private_memory = 0,
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kX86ObjectExportPlan = {
    .name = IREE_SVL("x86_64-object"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kX86Avx512Config = {
    .name = IREE_SVL("x86.avx512.core"),
    .contract_set_key = IREE_SVL("x86.avx512.core"),
};

static const loom_target_bundle_t kX86Avx512Bundle = {
    .name = IREE_SVL("x86_64-avx512-object"),
    .snapshot = &kX86Avx512Snapshot,
    .export_plan = &kX86ObjectExportPlan,
    .config = &kX86Avx512Config,
};

static const loom_target_snapshot_t kX86PackedDotSnapshot = {
    .name = IREE_SVL("x86_64-packed-dot"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .target_triple = LOOM_TARGET_LOW_X86_64_TRIPLE,
    .data_layout = LOOM_TARGET_LOW_X86_64_DATA_LAYOUT,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .target_features = LOOM_TARGET_LOW_X86_64_PACKED_DOT_FEATURES,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = UINT32_MAX,
            .constant = 0,
            .private_memory = 0,
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_config_t kX86PackedDotConfig = {
    .name = IREE_SVL("x86.packed_dot.core"),
    .contract_set_key = IREE_SVL("x86.packed_dot.core"),
    .contract_feature_bits =
        LOOM_X86_FEATURE_AVX512_BF16 | LOOM_X86_FEATURE_AVX512_VL |
        LOOM_X86_FEATURE_AVX_VNNI | LOOM_X86_FEATURE_AVX_VNNI_INT8,
};

static const loom_target_bundle_t kX86PackedDotBundle = {
    .name = IREE_SVL("x86_64-packed-dot-object"),
    .snapshot = &kX86PackedDotSnapshot,
    .export_plan = &kX86ObjectExportPlan,
    .config = &kX86PackedDotConfig,
};

static const loom_target_export_plan_t kAmdgpuHalExportPlan = {
    .name = IREE_SVL("amdgpu-hal"),
    .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
    .hal_kernel =
        {
            .binding_alignment = 16,
            .required_workgroup_size = {.x = 64, .y = 1, .z = 1},
            .flat_workgroup_size_min = 64,
            .flat_workgroup_size_max = 64,
            .buffer_resource_flags = 159744,
        },
};

#define LOOM_TARGET_LOW_AMDGPU_SNAPSHOT(name_value, cpu_value, features_value) \
  {                                                                            \
      .name = IREE_SVL(name_value),                                            \
      .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,                 \
      .target_triple = LOOM_TARGET_LOW_AMDGPU_TRIPLE,                          \
      .data_layout = LOOM_TARGET_LOW_AMDGPU_DATA_LAYOUT,                       \
      .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,                      \
      .target_cpu = IREE_SVL(cpu_value),                                       \
      .target_features = IREE_SVL(features_value),                             \
      .default_pointer_bitwidth = 64,                                          \
      .index_bitwidth = 32,                                                    \
      .offset_bitwidth = 64,                                                   \
      .memory_spaces =                                                         \
          {                                                                    \
              .generic = 0,                                                    \
              .global = 1,                                                     \
              .workgroup = 3,                                                  \
              .constant = 4,                                                   \
              .private_memory = 5,                                             \
              .host = UINT32_MAX,                                              \
              .descriptor = 7,                                                 \
          },                                                                   \
  }

static const loom_target_snapshot_t kAmdgpuGfx950Snapshot =
    LOOM_TARGET_LOW_AMDGPU_SNAPSHOT("amdgpu-gfx950", "gfx950",
                                    "+wavefrontsize64");
static const loom_target_snapshot_t kAmdgpuGfx11Snapshot =
    LOOM_TARGET_LOW_AMDGPU_SNAPSHOT("amdgpu-gfx11", "gfx1100",
                                    "+wavefrontsize32");
static const loom_target_snapshot_t kAmdgpuGfx12Snapshot =
    LOOM_TARGET_LOW_AMDGPU_SNAPSHOT("amdgpu-gfx12", "gfx1200",
                                    "+wavefrontsize32");
static const loom_target_snapshot_t kAmdgpuGfx1250Snapshot =
    LOOM_TARGET_LOW_AMDGPU_SNAPSHOT("amdgpu-gfx1250", "gfx1250",
                                    "+wavefrontsize32");

static const loom_target_config_t kAmdgpuGfx950Config = {
    .name = IREE_SVL("amdgpu.gfx950.core"),
    .contract_set_key = IREE_SVL("amdgpu.gfx950.core"),
};
static const loom_target_config_t kAmdgpuGfx11Config = {
    .name = IREE_SVL("amdgpu.gfx11.core"),
    .contract_set_key = IREE_SVL("amdgpu.gfx11.core"),
};
static const loom_target_config_t kAmdgpuGfx12Config = {
    .name = IREE_SVL("amdgpu.gfx12.core"),
    .contract_set_key = IREE_SVL("amdgpu.gfx12.core"),
};
static const loom_target_config_t kAmdgpuGfx1250Config = {
    .name = IREE_SVL("amdgpu.gfx1250.core"),
    .contract_set_key = IREE_SVL("amdgpu.gfx1250.core"),
};

static const loom_target_bundle_t kAmdgpuGfx950Bundle = {
    .name = IREE_SVL("amdgpu-gfx950-hal"),
    .snapshot = &kAmdgpuGfx950Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx950Config,
};
static const loom_target_bundle_t kAmdgpuGfx11Bundle = {
    .name = IREE_SVL("amdgpu-gfx11-hal"),
    .snapshot = &kAmdgpuGfx11Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx11Config,
};
static const loom_target_bundle_t kAmdgpuGfx12Bundle = {
    .name = IREE_SVL("amdgpu-gfx12-hal"),
    .snapshot = &kAmdgpuGfx12Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx12Config,
};
static const loom_target_bundle_t kAmdgpuGfx1250Bundle = {
    .name = IREE_SVL("amdgpu-gfx1250-hal"),
    .snapshot = &kAmdgpuGfx1250Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx1250Config,
};

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_ireevm_core_descriptor_set,
    loom_wasm_core_simd128_descriptor_set,
    loom_x86_avx512_core_descriptor_set,
    loom_x86_packed_dot_core_descriptor_set,
    loom_amdgpu_gfx950_core_descriptor_set,
    loom_amdgpu_gfx11_core_descriptor_set,
    loom_amdgpu_gfx12_core_descriptor_set,
    loom_amdgpu_gfx1250_core_descriptor_set,
};

static const loom_target_bundle_t* const kLowTargetBundles[] = {
    &kIreeVmBundle,       &kWasmSimd128Bundle,   &kX86Avx512Bundle,
    &kX86PackedDotBundle, &kAmdgpuGfx950Bundle,  &kAmdgpuGfx11Bundle,
    &kAmdgpuGfx12Bundle,  &kAmdgpuGfx1250Bundle,
};

void loom_target_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_target_low_descriptor_registry_t){
      .descriptor_set_providers = kLowDescriptorSetProviders,
      .descriptor_set_provider_count =
          IREE_ARRAYSIZE(kLowDescriptorSetProviders),
      .target_bundles = kLowTargetBundles,
      .target_bundle_count = IREE_ARRAYSIZE(kLowTargetBundles),
      .registry =
          {
              .descriptor_set_providers = kLowDescriptorSetProviders,
              .descriptor_set_provider_count =
                  IREE_ARRAYSIZE(kLowDescriptorSetProviders),
          },
  };
}

static iree_status_t loom_target_low_verify_bundle_record(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_target_bundle_t* bundle,
    loom_low_descriptor_requirement_flags_t requirements,
    iree_host_size_t row) {
  if (bundle == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry bundle row %" PRIhsz " is null", row);
  }
  if (iree_string_view_is_empty(iree_string_view_trim(bundle->name))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry bundle row %" PRIhsz " has no name", row);
  }
  if (bundle->snapshot == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low bundle '%.*s' has no target snapshot",
                            (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->export_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low bundle '%.*s' has no export plan",
                            (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low bundle '%.*s' has no config",
                            (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low bundle '%.*s' snapshot has unknown codegen format",
        (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->snapshot->artifact_format ==
      LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low bundle '%.*s' snapshot has unknown artifact format",
        (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low bundle '%.*s' export plan has unknown ABI",
        (int)bundle->name.size, bundle->name.data);
  }

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  return loom_target_low_descriptor_set_select_for_bundle(
      descriptor_registry, bundle, requirements, &descriptor_set);
}

iree_status_t loom_target_low_descriptor_registry_verify(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry is required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor-set providers are required");
  }
  if (registry->target_bundle_count != 0 && registry->target_bundles == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry bundles are required");
  }
  if (registry->registry.descriptor_set_count != 0 ||
      registry->registry.descriptor_sets != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor view must use provider tables");
  }
  if (registry->registry.descriptor_set_providers !=
          registry->descriptor_set_providers ||
      registry->registry.descriptor_set_provider_count !=
          registry->descriptor_set_provider_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor view does not match provider table");
  }

  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_registry_verify(&registry->registry));

  for (iree_host_size_t i = 0; i < registry->target_bundle_count; ++i) {
    const loom_target_bundle_t* bundle = registry->target_bundles[i];
    IREE_RETURN_IF_ERROR(loom_target_low_verify_bundle_record(
        &registry->registry, bundle, requirements, i));

    for (iree_host_size_t j = i + 1; j < registry->target_bundle_count; ++j) {
      const loom_target_bundle_t* other_bundle = registry->target_bundles[j];
      if (other_bundle == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "target-low registry bundle row %" PRIhsz " is null", j);
      }
      if (iree_string_view_equal(bundle->name, other_bundle->name)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "target-low registry has duplicate bundle key '%.*s'",
            (int)bundle->name.size, bundle->name.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_low_manifest_write_string_field(
    loom_output_stream_t* stream, const char* field_name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\":"));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_target_low_manifest_write_memory_spaces(
    loom_output_stream_t* stream,
    const loom_target_memory_space_map_t* memory_spaces) {
  return loom_output_stream_write_format(
      stream,
      "\"memory_spaces\":{\"generic\":%" PRIu32 ",\"global\":%" PRIu32
      ",\"workgroup\":%" PRIu32 ",\"constant\":%" PRIu32
      ",\"private_memory\":%" PRIu32 ",\"host\":%" PRIu32
      ",\"descriptor\":%" PRIu32 "}",
      memory_spaces->generic, memory_spaces->global, memory_spaces->workgroup,
      memory_spaces->constant, memory_spaces->private_memory,
      memory_spaces->host, memory_spaces->descriptor);
}

static iree_status_t loom_target_low_manifest_write_descriptor_set_summary(
    loom_output_stream_t* stream,
    const loom_low_descriptor_set_t* descriptor_set) {
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &key));
  iree_string_view_t target = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->target_key_string_offset, &target));
  iree_string_view_t feature_namespace = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->feature_key_string_offset,
      &feature_namespace));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  IREE_RETURN_IF_ERROR(
      loom_target_low_manifest_write_string_field(stream, "key", key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(
      loom_target_low_manifest_write_string_field(stream, "target", target));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "feature_namespace", feature_namespace));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"abi_version\":%" PRIu32 ",\"generator_version\":%" PRIu32
      ",\"table_counts\":{\"descriptors\":%" PRIu32
      ",\"descriptor_refs\":%" PRIu32 ",\"operands\":%" PRIu32
      ",\"immediates\":%" PRIu32 ",\"enum_domains\":%" PRIu32
      ",\"enum_values\":%" PRIu32 ",\"effects\":%" PRIu32
      ",\"constraints\":%" PRIu32 ",\"reg_classes\":%" PRIu32
      ",\"reg_class_alts\":%" PRIu32 ",\"schedule_classes\":%" PRIu32
      ",\"issue_uses\":%" PRIu32 ",\"resources\":%" PRIu32
      ",\"hazards\":%" PRIu32 ",\"pressure_deltas\":%" PRIu32
      ",\"feature_mask_words\":%" PRIu32 "}}",
      descriptor_set->abi_version, descriptor_set->generator_version,
      descriptor_set->descriptor_count, descriptor_set->descriptor_ref_count,
      descriptor_set->operand_count, descriptor_set->immediate_count,
      descriptor_set->enum_domain_count, descriptor_set->enum_value_count,
      descriptor_set->effect_count, descriptor_set->constraint_count,
      descriptor_set->reg_class_count, descriptor_set->reg_class_alt_count,
      descriptor_set->schedule_class_count, descriptor_set->issue_use_count,
      descriptor_set->resource_count, descriptor_set->hazard_count,
      descriptor_set->pressure_delta_count,
      descriptor_set->feature_mask_word_count));
  return iree_ok_status();
}

static iree_status_t loom_target_low_manifest_write_bundle_summary(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_target_bundle_t* bundle,
    loom_low_descriptor_requirement_flags_t requirements,
    loom_output_stream_t* stream) {
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      descriptor_registry, bundle, requirements, &descriptor_set));
  iree_string_view_t descriptor_set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &descriptor_set_key));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  IREE_RETURN_IF_ERROR(
      loom_target_low_manifest_write_string_field(stream, "key", bundle->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"snapshot\":{"));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "name", bundle->snapshot->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"codegen_format\":%u,",
      (unsigned)bundle->snapshot->codegen_format));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "target_triple", bundle->snapshot->target_triple));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "data_layout", bundle->snapshot->data_layout));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"artifact_format\":%u,",
      (unsigned)bundle->snapshot->artifact_format));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "target_cpu", bundle->snapshot->target_cpu));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "target_features", bundle->snapshot->target_features));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"default_pointer_bitwidth\":%" PRIu32 ",\"index_bitwidth\":%" PRIu32
      ",\"offset_bitwidth\":%" PRIu32 ",",
      bundle->snapshot->default_pointer_bitwidth,
      bundle->snapshot->index_bitwidth, bundle->snapshot->offset_bitwidth));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_memory_spaces(
      stream, &bundle->snapshot->memory_spaces));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "},\"export\":{"));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "name", bundle->export_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "source_symbol", bundle->export_plan->source_symbol));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "export_symbol", bundle->export_plan->export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, ",\"abi_kind\":%u,\"linkage\":%u",
                                      (unsigned)bundle->export_plan->abi_kind,
                                      (unsigned)bundle->export_plan->linkage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"hal_kernel\":{\"binding_alignment\":%" PRIu32
      ",\"required_workgroup_size\":{\"x\":%" PRIu32 ",\"y\":%" PRIu32
      ",\"z\":%" PRIu32 "},\"flat_workgroup_size_min\":%" PRIu32
      ",\"flat_workgroup_size_max\":%" PRIu32
      ",\"buffer_resource_flags\":%" PRIu32 "}",
      bundle->export_plan->hal_kernel.binding_alignment,
      bundle->export_plan->hal_kernel.required_workgroup_size.x,
      bundle->export_plan->hal_kernel.required_workgroup_size.y,
      bundle->export_plan->hal_kernel.required_workgroup_size.z,
      bundle->export_plan->hal_kernel.flat_workgroup_size_min,
      bundle->export_plan->hal_kernel.flat_workgroup_size_max,
      bundle->export_plan->hal_kernel.buffer_resource_flags));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "},\"config\":{"));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "name", bundle->config->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "descriptor_set", descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"contract_feature_bits\":%" PRIu64 "}}",
      bundle->config->contract_feature_bits));
  return iree_ok_status();
}

iree_status_t loom_target_low_descriptor_registry_format_manifest_json(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements,
    iree_string_builder_t* builder) {
  if (builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry manifest builder is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_target_low_descriptor_registry_verify(registry, requirements));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(&registry->registry);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      "{\"descriptor_set_count\":%" PRIhsz ",\"bundle_count\":%" PRIhsz
      ",\"descriptor_sets\":[",
      descriptor_set_count, registry->target_bundle_count));
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(&registry->registry, i);
    IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_descriptor_set_summary(
        &stream, descriptor_set));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "],\"bundles\":["));
  for (iree_host_size_t i = 0; i < registry->target_bundle_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_bundle_summary(
        &registry->registry, registry->target_bundles[i], requirements,
        &stream));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}

iree_status_t loom_target_low_descriptor_set_lookup(
    iree_string_view_t key,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  loom_target_low_descriptor_registry_t registry;
  loom_target_low_descriptor_registry_initialize(&registry);
  return loom_low_descriptor_registry_lookup(&registry.registry, key,
                                             out_descriptor_set);
}

iree_status_t loom_target_low_descriptor_registry_lookup_bundle(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_view_t key, const loom_target_bundle_t** out_bundle) {
  if (registry == NULL) {
    if (out_bundle != NULL) *out_bundle = NULL;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry is required");
  }
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(registry);
  return loom_target_preset_registry_lookup_bundle(&preset_registry, key,
                                                   out_bundle);
}

iree_status_t loom_target_low_bundle_lookup(
    iree_string_view_t key, const loom_target_bundle_t** out_bundle) {
  loom_target_low_descriptor_registry_t registry;
  loom_target_low_descriptor_registry_initialize(&registry);
  return loom_target_low_descriptor_registry_lookup_bundle(&registry, key,
                                                           out_bundle);
}

iree_status_t loom_target_low_descriptor_set_select_for_bundle(
    const loom_low_descriptor_registry_t* registry,
    const loom_target_bundle_t* bundle,
    loom_low_descriptor_requirement_flags_t requirements,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  if (out_descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set output is required");
  }
  *out_descriptor_set = NULL;
  if (registry == NULL || bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry and target bundle are "
                            "required");
  }
  if (bundle->config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle '%.*s' has no config",
                            (int)bundle->name.size, bundle->name.data);
  }
  iree_string_view_t descriptor_set_key =
      iree_string_view_trim(bundle->config->contract_set_key);
  if (iree_string_view_is_empty(descriptor_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' config '%.*s' does not name a low descriptor set",
        (int)bundle->name.size, bundle->name.data,
        (int)bundle->config->name.size, bundle->config->name.data);
  }

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_registry_lookup(
      registry, descriptor_set_key, &descriptor_set));
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target bundle '%.*s' selected low descriptor set '%.*s' that is not "
        "linked",
        (int)bundle->name.size, bundle->name.data, (int)descriptor_set_key.size,
        descriptor_set_key.data);
  }
  if (requirements != 0) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_verify_requirements(
        descriptor_set, requirements));
  }
  *out_descriptor_set = descriptor_set;
  return iree_ok_status();
}
