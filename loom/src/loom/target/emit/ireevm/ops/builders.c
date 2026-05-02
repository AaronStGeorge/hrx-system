// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.c_tables.
// clang-format off

#include "loom/target/emit/ireevm/ops/ops.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/builder_macros.h"

iree_status_t loom_ireevm_target_build(
    loom_builder_t* builder,
    loom_ireevm_target_build_flags_t build_flags,
    loom_ireevm_target_kind_t kind,
    loom_symbol_ref_t symbol,
    loom_optional uint8_t codegen_format,
    loom_optional loom_string_id_t target_triple,
    loom_optional loom_string_id_t data_layout,
    loom_optional uint8_t artifact_format,
    loom_optional loom_string_id_t target_cpu,
    loom_optional loom_string_id_t target_features,
    loom_optional int64_t default_pointer_bitwidth,
    loom_optional int64_t index_bitwidth,
    loom_optional int64_t offset_bitwidth,
    loom_optional int64_t max_workgroup_size_x,
    loom_optional int64_t max_workgroup_size_y,
    loom_optional int64_t max_workgroup_size_z,
    loom_optional int64_t max_flat_workgroup_size,
    loom_optional int64_t subgroup_size,
    loom_optional int64_t max_workgroup_count_x,
    loom_optional int64_t max_workgroup_count_y,
    loom_optional int64_t max_workgroup_count_z,
    loom_optional int64_t memory_space_generic,
    loom_optional int64_t memory_space_global,
    loom_optional int64_t memory_space_workgroup,
    loom_optional int64_t memory_space_constant,
    loom_optional int64_t memory_space_private,
    loom_optional int64_t memory_space_host,
    loom_optional int64_t memory_space_descriptor,
    loom_optional uint8_t abi,
    loom_optional loom_string_id_t export_symbol,
    loom_optional uint8_t linkage,
    loom_optional int64_t hal_binding_alignment,
    loom_optional int64_t hal_buffer_resource_flags,
    loom_optional loom_string_id_t contract_set_key,
    loom_optional int64_t contract_feature_bits,
    loom_location_id_t location,
    loom_op_t** out_op) {
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, LOOM_OP_IREEVM_TARGET, 0,
      0, 0, 0,
      33, location, out_op));
  loom_op_attrs(*out_op)[1] = loom_attr_enum(kind);
  loom_op_attrs(*out_op)[0] = loom_attr_symbol(symbol);
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT)) {
    loom_op_attrs(*out_op)[2] = loom_attr_enum(codegen_format);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_TARGET_TRIPLE)) {
    loom_op_attrs(*out_op)[3] = loom_attr_string(target_triple);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_DATA_LAYOUT)) {
    loom_op_attrs(*out_op)[4] = loom_attr_string(data_layout);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_ARTIFACT_FORMAT)) {
    loom_op_attrs(*out_op)[5] = loom_attr_enum(artifact_format);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_TARGET_CPU)) {
    loom_op_attrs(*out_op)[6] = loom_attr_string(target_cpu);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_TARGET_FEATURES)) {
    loom_op_attrs(*out_op)[7] = loom_attr_string(target_features);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH)) {
    loom_op_attrs(*out_op)[8] = loom_attr_i64(default_pointer_bitwidth);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_INDEX_BITWIDTH)) {
    loom_op_attrs(*out_op)[9] = loom_attr_i64(index_bitwidth);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_OFFSET_BITWIDTH)) {
    loom_op_attrs(*out_op)[10] = loom_attr_i64(offset_bitwidth);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X)) {
    loom_op_attrs(*out_op)[11] = loom_attr_i64(max_workgroup_size_x);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y)) {
    loom_op_attrs(*out_op)[12] = loom_attr_i64(max_workgroup_size_y);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z)) {
    loom_op_attrs(*out_op)[13] = loom_attr_i64(max_workgroup_size_z);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE)) {
    loom_op_attrs(*out_op)[14] = loom_attr_i64(max_flat_workgroup_size);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE)) {
    loom_op_attrs(*out_op)[15] = loom_attr_i64(subgroup_size);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X)) {
    loom_op_attrs(*out_op)[16] = loom_attr_i64(max_workgroup_count_x);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y)) {
    loom_op_attrs(*out_op)[17] = loom_attr_i64(max_workgroup_count_y);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z)) {
    loom_op_attrs(*out_op)[18] = loom_attr_i64(max_workgroup_count_z);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC)) {
    loom_op_attrs(*out_op)[19] = loom_attr_i64(memory_space_generic);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL)) {
    loom_op_attrs(*out_op)[20] = loom_attr_i64(memory_space_global);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP)) {
    loom_op_attrs(*out_op)[21] = loom_attr_i64(memory_space_workgroup);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT)) {
    loom_op_attrs(*out_op)[22] = loom_attr_i64(memory_space_constant);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE)) {
    loom_op_attrs(*out_op)[23] = loom_attr_i64(memory_space_private);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_HOST)) {
    loom_op_attrs(*out_op)[24] = loom_attr_i64(memory_space_host);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR)) {
    loom_op_attrs(*out_op)[25] = loom_attr_i64(memory_space_descriptor);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_ABI)) {
    loom_op_attrs(*out_op)[26] = loom_attr_enum(abi);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    loom_op_attrs(*out_op)[27] = loom_attr_string(export_symbol);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_LINKAGE)) {
    loom_op_attrs(*out_op)[28] = loom_attr_enum(linkage);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_HAL_BINDING_ALIGNMENT)) {
    loom_op_attrs(*out_op)[29] = loom_attr_i64(hal_binding_alignment);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_HAL_BUFFER_RESOURCE_FLAGS)) {
    loom_op_attrs(*out_op)[30] = loom_attr_i64(hal_buffer_resource_flags);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    loom_op_attrs(*out_op)[31] = loom_attr_string(contract_set_key);
  }
  if (iree_any_bit_set(build_flags, LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS)) {
    loom_op_attrs(*out_op)[32] = loom_attr_i64(contract_feature_bits);
  }
  return loom_builder_finalize_op(builder, *out_op);
}
