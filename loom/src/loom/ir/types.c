// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include <string.h>

iree_status_t loom_type_function_build(const loom_type_t* arg_types,
                                       uint16_t arg_count,
                                       const loom_type_t* result_types,
                                       uint16_t result_count,
                                       iree_allocator_t allocator,
                                       loom_type_t* out_type) {
  // Use IREE_STRUCT_LAYOUT to overflow-check the total allocation size.
  // The two STRUCT_FIELD entries validate arg_count * sizeof(loom_type_t)
  // and result_count * sizeof(loom_type_t) independently, rejecting
  // overflow from untrusted bytecode counts before any allocation.
  iree_host_size_t alloc_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_func_type_data_t), &alloc_size,
      IREE_STRUCT_FIELD_FAM(arg_count, loom_type_t),
      IREE_STRUCT_FIELD(result_count, loom_type_t, /*out_offset=*/NULL)));
  loom_func_type_data_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, alloc_size, (void**)&data));
  data->arg_count = arg_count;
  data->result_count = result_count;
  data->reserved = 0;
  memcpy(data->types, arg_types, arg_count * sizeof(loom_type_t));
  memcpy(data->types + arg_count, result_types,
         result_count * sizeof(loom_type_t));
  *out_type = loom_type_function(data);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type equality
//===----------------------------------------------------------------------===//

static bool loom_type_sequence_equal(const loom_type_t* a_types,
                                     const loom_type_t* b_types,
                                     uint16_t type_count) {
  if (type_count == 0) return true;
  if (!a_types || !b_types) return a_types == b_types;
  for (uint16_t i = 0; i < type_count; ++i) {
    if (!loom_type_equal(a_types[i], b_types[i])) return false;
  }
  return true;
}

bool loom_type_equal(loom_type_t a, loom_type_t b) {
  if (a.header != b.header || a.encoding_id != b.encoding_id ||
      a.encoding_flags != b.encoding_flags) {
    return false;
  }
  loom_type_kind_t kind = loom_type_kind(a);
  if (!loom_type_kind_is_valid(kind)) {
    return a.dims[0] == b.dims[0] && a.dims[1] == b.dims[1];
  }
  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* a_data = loom_type_func_data(a);
      const loom_func_type_data_t* b_data = loom_type_func_data(b);
      if (!a_data || !b_data) return a_data == b_data;
      return a_data->arg_count == b_data->arg_count &&
             a_data->result_count == b_data->result_count &&
             loom_type_sequence_equal(
                 a_data->types, b_data->types,
                 (uint16_t)(a_data->arg_count + a_data->result_count));
    }
    case LOOM_TYPE_DIALECT: {
      uint16_t param_count = loom_type_dialect_param_count(a);
      return loom_type_dialect_name_id(a) == loom_type_dialect_name_id(b) &&
             loom_type_sequence_equal(loom_type_dialect_params(a),
                                      loom_type_dialect_params(b), param_count);
    }
    default:
      break;
  }
  if (loom_type_has_inline_dims(a)) {
    return a.dims[0] == b.dims[0] && a.dims[1] == b.dims[1];
  }
  // Overflow: compare each dim via the overflow pointer.
  uint8_t rank = loom_type_rank(a);
  const loom_overflow_dim_t* a_dims =
      (const loom_overflow_dim_t*)(uintptr_t)a.dims[0];
  const loom_overflow_dim_t* b_dims =
      (const loom_overflow_dim_t*)(uintptr_t)b.dims[0];
  if (rank == 0 || !a_dims || !b_dims) return a_dims == b_dims;
  for (uint8_t i = 0; i < rank; ++i) {
    if (a_dims[i] != b_dims[i]) return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Type hashing
//===----------------------------------------------------------------------===//

static uint32_t loom_type_hash_mix_bytes(uint32_t hash, const void* data,
                                         iree_host_size_t length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_type_hash_mix_u16(uint32_t hash, uint16_t value) {
  return loom_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_type_hash_mix_u32(uint32_t hash, uint32_t value) {
  return loom_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_type_hash_mix_u64(uint32_t hash, uint64_t value) {
  return loom_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_type_hash_mix_sequence(uint32_t hash,
                                            const loom_type_t* types,
                                            uint16_t type_count) {
  hash = loom_type_hash_mix_u16(hash, type_count);
  if (!types) return hash;
  for (uint16_t i = 0; i < type_count; ++i) {
    uint32_t element_hash = loom_type_hash(types[i]);
    hash = loom_type_hash_mix_u32(hash, element_hash);
  }
  return hash;
}

uint32_t loom_type_hash(loom_type_t type) {
  uint32_t hash = 2166136261u;
  hash = loom_type_hash_mix_u32(hash, type.header);
  hash = loom_type_hash_mix_u16(hash, type.encoding_id);
  hash = loom_type_hash_mix_u16(hash, type.encoding_flags);

  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) {
    hash = loom_type_hash_mix_u64(hash, type.dims[0]);
    return loom_type_hash_mix_u64(hash, type.dims[1]);
  }
  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return hash;
      hash = loom_type_hash_mix_u16(hash, data->arg_count);
      hash = loom_type_hash_mix_u16(hash, data->result_count);
      return loom_type_hash_mix_sequence(
          hash, data->types, (uint16_t)(data->arg_count + data->result_count));
    }
    case LOOM_TYPE_DIALECT:
      hash = loom_type_hash_mix_u32(hash, loom_type_dialect_name_id(type));
      return loom_type_hash_mix_sequence(hash, loom_type_dialect_params(type),
                                         loom_type_dialect_param_count(type));
    default:
      break;
  }

  if (loom_type_has_inline_dims(type)) {
    hash = loom_type_hash_mix_u64(hash, type.dims[0]);
    return loom_type_hash_mix_u64(hash, type.dims[1]);
  }

  uint8_t rank = loom_type_rank(type);
  const loom_overflow_dim_t* dims =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  if (rank == 0 || !dims) return hash;
  for (uint8_t i = 0; i < rank; ++i) {
    hash = loom_type_hash_mix_u64(hash, dims[i]);
  }
  return hash;
}
