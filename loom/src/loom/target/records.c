// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/records.h"

#define LOOM_TARGET_FINGERPRINT_OFFSET UINT64_C(1469598103934665603)
#define LOOM_TARGET_FINGERPRINT_PRIME UINT64_C(1099511628211)

static uint64_t loom_target_fingerprint_byte(uint64_t hash, uint8_t value) {
  hash ^= value;
  return hash * LOOM_TARGET_FINGERPRINT_PRIME;
}

static uint64_t loom_target_fingerprint_u32(uint64_t hash, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    hash = loom_target_fingerprint_byte(hash, (uint8_t)(value & 0xFFu));
    value >>= 8;
  }
  return hash;
}

static uint64_t loom_target_fingerprint_u64(uint64_t hash, uint64_t value) {
  for (uint32_t i = 0; i < 8; ++i) {
    hash = loom_target_fingerprint_byte(hash, (uint8_t)(value & 0xFFu));
    value >>= 8;
  }
  return hash;
}

static uint64_t loom_target_fingerprint_string(uint64_t hash,
                                               iree_string_view_t value) {
  hash = loom_target_fingerprint_u64(hash, value.size);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    hash = loom_target_fingerprint_byte(hash, (uint8_t)value.data[i]);
  }
  return hash;
}

static uint64_t loom_target_fingerprint_memory_spaces(
    uint64_t hash, loom_target_memory_space_map_t memory_spaces) {
  hash = loom_target_fingerprint_u32(hash, memory_spaces.generic);
  hash = loom_target_fingerprint_u32(hash, memory_spaces.global);
  hash = loom_target_fingerprint_u32(hash, memory_spaces.workgroup);
  hash = loom_target_fingerprint_u32(hash, memory_spaces.constant);
  hash = loom_target_fingerprint_u32(hash, memory_spaces.private_memory);
  hash = loom_target_fingerprint_u32(hash, memory_spaces.host);
  return loom_target_fingerprint_u32(hash, memory_spaces.descriptor);
}

static uint64_t loom_target_fingerprint_workgroup_size(
    uint64_t hash, loom_target_workgroup_size_t workgroup_size) {
  hash = loom_target_fingerprint_u32(hash, workgroup_size.x);
  hash = loom_target_fingerprint_u32(hash, workgroup_size.y);
  return loom_target_fingerprint_u32(hash, workgroup_size.z);
}

static uint64_t loom_target_fingerprint_hal_kernel_abi(
    uint64_t hash, loom_target_hal_kernel_abi_t hal_kernel) {
  hash = loom_target_fingerprint_u32(hash, hal_kernel.binding_alignment);
  hash = loom_target_fingerprint_workgroup_size(
      hash, hal_kernel.required_workgroup_size);
  hash = loom_target_fingerprint_u32(hash, hal_kernel.flat_workgroup_size_min);
  hash = loom_target_fingerprint_u32(hash, hal_kernel.flat_workgroup_size_max);
  return loom_target_fingerprint_u32(hash, hal_kernel.buffer_resource_flags);
}

iree_status_t loom_target_snapshot_fingerprint(
    const loom_target_snapshot_t* snapshot, uint64_t* out_fingerprint) {
  if (out_fingerprint == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target snapshot fingerprint output is required");
  }
  *out_fingerprint = 0;
  if (snapshot == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target snapshot is required");
  }
  uint64_t hash = LOOM_TARGET_FINGERPRINT_OFFSET;
  hash =
      loom_target_fingerprint_string(hash, IREE_SV("loom.target.snapshot.v0"));
  hash = loom_target_fingerprint_string(hash, snapshot->name);
  hash = loom_target_fingerprint_u32(hash, (uint32_t)snapshot->codegen_format);
  hash = loom_target_fingerprint_string(hash, snapshot->target_triple);
  hash = loom_target_fingerprint_string(hash, snapshot->data_layout);
  hash = loom_target_fingerprint_u32(hash, (uint32_t)snapshot->artifact_format);
  hash = loom_target_fingerprint_string(hash, snapshot->target_cpu);
  hash = loom_target_fingerprint_string(hash, snapshot->target_features);
  hash = loom_target_fingerprint_u32(hash, snapshot->default_pointer_bitwidth);
  hash = loom_target_fingerprint_u32(hash, snapshot->index_bitwidth);
  hash = loom_target_fingerprint_u32(hash, snapshot->offset_bitwidth);
  hash = loom_target_fingerprint_memory_spaces(hash, snapshot->memory_spaces);
  *out_fingerprint = hash;
  return iree_ok_status();
}

iree_status_t loom_target_export_plan_fingerprint(
    const loom_target_export_plan_t* export_plan, uint64_t* out_fingerprint) {
  if (out_fingerprint == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target export-plan fingerprint output is required");
  }
  *out_fingerprint = 0;
  if (export_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target export plan is required");
  }
  uint64_t hash = LOOM_TARGET_FINGERPRINT_OFFSET;
  hash = loom_target_fingerprint_string(hash, IREE_SV("loom.target.export.v0"));
  hash = loom_target_fingerprint_string(hash, export_plan->name);
  hash = loom_target_fingerprint_string(hash, export_plan->source_symbol);
  hash = loom_target_fingerprint_string(hash, export_plan->export_symbol);
  hash = loom_target_fingerprint_u32(hash, (uint32_t)export_plan->abi_kind);
  hash = loom_target_fingerprint_u32(hash, (uint32_t)export_plan->linkage);
  hash = loom_target_fingerprint_hal_kernel_abi(hash, export_plan->hal_kernel);
  *out_fingerprint = hash;
  return iree_ok_status();
}

iree_status_t loom_target_config_fingerprint(const loom_target_config_t* config,
                                             uint64_t* out_fingerprint) {
  if (out_fingerprint == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target config fingerprint output is required");
  }
  *out_fingerprint = 0;
  if (config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target config is required");
  }
  uint64_t hash = LOOM_TARGET_FINGERPRINT_OFFSET;
  hash = loom_target_fingerprint_string(hash, IREE_SV("loom.target.config.v0"));
  hash = loom_target_fingerprint_string(hash, config->name);
  hash = loom_target_fingerprint_string(hash, config->contract_set_key);
  hash = loom_target_fingerprint_u64(hash, config->contract_feature_bits);
  *out_fingerprint = hash;
  return iree_ok_status();
}

iree_status_t loom_target_bundle_fingerprint(const loom_target_bundle_t* bundle,
                                             uint64_t* out_fingerprint) {
  if (out_fingerprint == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle fingerprint output is required");
  }
  *out_fingerprint = 0;
  if (bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle is required");
  }
  uint64_t snapshot_fingerprint = 0;
  IREE_RETURN_IF_ERROR(loom_target_snapshot_fingerprint(bundle->snapshot,
                                                        &snapshot_fingerprint));
  uint64_t export_plan_fingerprint = 0;
  IREE_RETURN_IF_ERROR(loom_target_export_plan_fingerprint(
      bundle->export_plan, &export_plan_fingerprint));
  uint64_t config_fingerprint = 0;
  IREE_RETURN_IF_ERROR(
      loom_target_config_fingerprint(bundle->config, &config_fingerprint));

  uint64_t hash = LOOM_TARGET_FINGERPRINT_OFFSET;
  hash = loom_target_fingerprint_string(hash, IREE_SV("loom.target.bundle.v0"));
  hash = loom_target_fingerprint_string(hash, bundle->name);
  hash = loom_target_fingerprint_u64(hash, snapshot_fingerprint);
  hash = loom_target_fingerprint_u64(hash, export_plan_fingerprint);
  hash = loom_target_fingerprint_u64(hash, config_fingerprint);
  *out_fingerprint = hash;
  return iree_ok_status();
}
