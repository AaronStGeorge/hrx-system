// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Scalar type names
//===----------------------------------------------------------------------===//

static const char* const loom_scalar_type_names[] = {
    [LOOM_SCALAR_TYPE_INDEX] = "index",   [LOOM_SCALAR_TYPE_OFFSET] = "offset",
    [LOOM_SCALAR_TYPE_I1] = "i1",         [LOOM_SCALAR_TYPE_I8] = "i8",
    [LOOM_SCALAR_TYPE_I16] = "i16",       [LOOM_SCALAR_TYPE_I32] = "i32",
    [LOOM_SCALAR_TYPE_I64] = "i64",       [LOOM_SCALAR_TYPE_F8E4M3] = "f8E4M3",
    [LOOM_SCALAR_TYPE_F8E5M2] = "f8E5M2", [LOOM_SCALAR_TYPE_F16] = "f16",
    [LOOM_SCALAR_TYPE_BF16] = "bf16",     [LOOM_SCALAR_TYPE_F32] = "f32",
    [LOOM_SCALAR_TYPE_F64] = "f64",
};

static const int32_t loom_scalar_type_bitwidths[] = {
    [LOOM_SCALAR_TYPE_INDEX] = 64, [LOOM_SCALAR_TYPE_OFFSET] = 64,
    [LOOM_SCALAR_TYPE_I1] = 1,     [LOOM_SCALAR_TYPE_I8] = 8,
    [LOOM_SCALAR_TYPE_I16] = 16,   [LOOM_SCALAR_TYPE_I32] = 32,
    [LOOM_SCALAR_TYPE_I64] = 64,   [LOOM_SCALAR_TYPE_F8E4M3] = 8,
    [LOOM_SCALAR_TYPE_F8E5M2] = 8, [LOOM_SCALAR_TYPE_F16] = 16,
    [LOOM_SCALAR_TYPE_BF16] = 16,  [LOOM_SCALAR_TYPE_F32] = 32,
    [LOOM_SCALAR_TYPE_F64] = 64,
};

static_assert(IREE_ARRAYSIZE(loom_scalar_type_names) == LOOM_SCALAR_TYPE_COUNT_,
              "loom_scalar_type_names out of sync with enum");
static_assert(IREE_ARRAYSIZE(loom_scalar_type_bitwidths) ==
                  LOOM_SCALAR_TYPE_COUNT_,
              "loom_scalar_type_bitwidths out of sync with enum");

const char* loom_scalar_type_name(loom_scalar_type_t type) {
  if (type >= 0 && type < (int)LOOM_SCALAR_TYPE_COUNT_) {
    return loom_scalar_type_names[type];
  }
  return NULL;
}

int32_t loom_scalar_type_bitwidth(loom_scalar_type_t type) {
  if (type >= 0 && type < (int)LOOM_SCALAR_TYPE_COUNT_) {
    return loom_scalar_type_bitwidths[type];
  }
  return 0;
}

bool loom_scalar_type_parse(iree_string_view_t name,
                            loom_scalar_type_t* out_type) {
  for (int i = 0; i < (int)LOOM_SCALAR_TYPE_COUNT_; ++i) {
    if (loom_scalar_type_names[i] &&
        iree_string_view_equal(
            name, iree_make_cstring_view(loom_scalar_type_names[i]))) {
      *out_type = (loom_scalar_type_t)i;
      return true;
    }
  }
  return false;
}

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

bool loom_type_equal(loom_type_t a, loom_type_t b) {
  if (a.header != b.header || a.encoding_id != b.encoding_id ||
      a.encoding_flags != b.encoding_flags) {
    return false;
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
  for (uint8_t i = 0; i < rank; ++i) {
    if (a_dims[i] != b_dims[i]) return false;
  }
  return true;
}
