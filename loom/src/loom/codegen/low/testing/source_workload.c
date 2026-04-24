// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/source_workload.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/verify/verify.h"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

static loom_type_t loom_low_source_workload_i32_type(void) {
  return loom_type_scalar(LOOM_SCALAR_TYPE_I32);
}

static loom_type_t loom_low_source_workload_index_type(void) {
  return loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
}

static loom_type_t loom_low_source_workload_vector4xi32_type(void) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                             loom_dim_pack_static(4), 0);
}

static loom_type_t loom_low_source_workload_vector16xi8_type(void) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                             loom_dim_pack_static(16), 0);
}

static loom_type_t loom_low_source_workload_view4xi32_type(
    loom_value_id_t layout_id) {
  loom_type_t type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);
  type.encoding_id = (uint16_t)layout_id;
  type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  return type;
}

//===----------------------------------------------------------------------===//
// Config
//===----------------------------------------------------------------------===//

loom_low_source_workload_config_t loom_low_source_workload_config_make(
    iree_string_view_t target_preset, uint32_t scale) {
  if (scale == 0) {
    scale = 1;
  }
  uint32_t op_count = scale * 64;
  if (op_count > UINT16_MAX) {
    op_count = UINT16_MAX;
  }
  return (loom_low_source_workload_config_t){
      .module_name = IREE_SV("source_low_workload"),
      .target_symbol_name = IREE_SV("target"),
      .target_preset = target_preset,
      .function_symbol_name = IREE_SV("generated"),
      .op_count = (uint16_t)op_count,
  };
}

static iree_string_view_t loom_low_source_workload_default_string(
    iree_string_view_t value, iree_string_view_t default_value) {
  return iree_string_view_is_empty(value) ? default_value : value;
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

typedef const loom_op_vtable_t* const* (
    *loom_low_source_workload_dialect_vtables_fn_t)(iree_host_size_t* count);

static iree_status_t loom_low_source_workload_register_dialect(
    loom_context_t* context, uint8_t dialect_id,
    loom_low_source_workload_dialect_vtables_fn_t dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

iree_status_t loom_low_source_workload_register_dialects(
    loom_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(loom_low_source_workload_register_dialect(
      context, LOOM_DIALECT_TARGET, loom_target_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_low_source_workload_register_dialect(
      context, LOOM_DIALECT_FUNC, loom_func_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_low_source_workload_register_dialect(
      context, LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_low_source_workload_register_dialect(
      context, LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_low_source_workload_register_dialect(
      context, LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_low_source_workload_register_dialect(
      context, LOOM_DIALECT_INDEX, loom_index_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_low_source_workload_register_dialect(
      context, LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables));
  return loom_low_source_workload_register_dialect(context, LOOM_DIALECT_LOW,
                                                   loom_low_dialect_vtables);
}

//===----------------------------------------------------------------------===//
// Symbol helpers
//===----------------------------------------------------------------------===//

static iree_status_t loom_low_source_workload_add_symbol(
    loom_builder_t* builder, iree_string_view_t name,
    loom_symbol_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(out_ref);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(builder->module, name_id, &symbol_id));
  *out_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Source op hooks
//===----------------------------------------------------------------------===//

static loom_value_id_t loom_low_source_workload_pick_latest_exact_type(
    const loom_test_gen_values_t* values, loom_type_t type) {
  for (uint16_t i = values->count; i > 0; --i) {
    uint16_t index = (uint16_t)(i - 1);
    if (loom_type_equal(values->types[index], type)) {
      return values->entries[index];
    }
  }
  return LOOM_VALUE_ID_INVALID;
}

static iree_status_t loom_low_source_workload_gen_scalar_i32_constant(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      context->builder,
      loom_attr_i64((int64_t)loom_test_gen_next_range(context->gen, 256)),
      loom_low_source_workload_i32_type(), LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_scalar_constant_result(op),
                           loom_low_source_workload_i32_type());
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_scalar_i32_binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t lhs = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_I32);
  loom_value_id_t rhs = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_I32);
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  switch (loom_test_gen_next_range(context->gen, 3)) {
    case 0: {
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
          context->builder, 0, lhs, rhs, loom_low_source_workload_i32_type(),
          LOOM_LOCATION_UNKNOWN, &op));
      break;
    }
    case 1: {
      IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
          context->builder, 0, lhs, rhs, loom_low_source_workload_i32_type(),
          LOOM_LOCATION_UNKNOWN, &op));
      break;
    }
    default: {
      IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
          context->builder, 0, lhs, rhs, loom_low_source_workload_i32_type(),
          LOOM_LOCATION_UNKNOWN, &op));
      break;
    }
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                           loom_low_source_workload_i32_type());
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_vector4xi32_binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t vector_type = loom_low_source_workload_vector4xi32_type();
  loom_value_id_t lhs = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  loom_value_id_t rhs = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  switch (loom_test_gen_next_range(context->gen, 3)) {
    case 0: {
      IREE_RETURN_IF_ERROR(loom_vector_addi_build(context->builder, 0, lhs, rhs,
                                                  vector_type,
                                                  LOOM_LOCATION_UNKNOWN, &op));
      break;
    }
    case 1: {
      IREE_RETURN_IF_ERROR(loom_vector_subi_build(context->builder, 0, lhs, rhs,
                                                  vector_type,
                                                  LOOM_LOCATION_UNKNOWN, &op));
      break;
    }
    default: {
      IREE_RETURN_IF_ERROR(loom_vector_muli_build(context->builder, 0, lhs, rhs,
                                                  vector_type,
                                                  LOOM_LOCATION_UNKNOWN, &op));
      break;
    }
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                           vector_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_vector4xi32_extract(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t vector_type = loom_low_source_workload_vector4xi32_type();
  loom_value_id_t source = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  if (source == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->builder->arena, 1,
                                                 sizeof(*static_indices),
                                                 (void**)&static_indices));
  static_indices[0] =
      (int64_t)loom_test_gen_next_range(context->gen, /*upper_exclusive=*/4);

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      context->builder, source, NULL, 0, static_indices, 1,
      loom_low_source_workload_i32_type(), LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_vector_extract_result(op),
                           loom_low_source_workload_i32_type());
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_vector4xi32_reduce_addi(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t vector_type = loom_low_source_workload_vector4xi32_type();
  loom_type_t scalar_type = loom_low_source_workload_i32_type();
  loom_value_id_t input = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  loom_value_id_t init = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, scalar_type);
  if (input == LOOM_VALUE_ID_INVALID || init == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
      context->builder, LOOM_VECTOR_REDUCE_KIND_ADDI, input, init, scalar_type,
      LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_vector_reduce_result(op),
                           scalar_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_vector_dot4i_s8s8(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t lhs_type = loom_low_source_workload_vector16xi8_type();
  loom_type_t result_type = loom_low_source_workload_vector4xi32_type();
  loom_value_id_t lhs = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, lhs_type);
  loom_value_id_t rhs = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, lhs_type);
  loom_value_id_t acc = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, result_type);
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID ||
      acc == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_dot4i_build(
      context->builder, LOOM_VECTOR_DOT4I_KIND_S8S8, lhs, rhs, acc, result_type,
      LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_vector_dot4i_result(op),
                           result_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_vector4xi32_shuffle(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t vector_type = loom_low_source_workload_vector4xi32_type();
  loom_value_id_t source = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  if (source == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  int64_t* source_lanes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->builder->arena, 4,
                                                 sizeof(*source_lanes),
                                                 (void**)&source_lanes));
  for (iree_host_size_t i = 0; i < 4; ++i) {
    source_lanes[i] =
        (int64_t)loom_test_gen_next_range(context->gen, /*upper_exclusive=*/4);
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_shuffle_build(context->builder, source_lanes,
                                                 4, source, vector_type,
                                                 LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_vector_shuffle_result(op),
                           vector_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

typedef struct loom_low_source_workload_memory_state_t {
  // Source view used by generated loads.
  loom_value_id_t load_view;
  // Source view used by generated stores.
  loom_value_id_t store_view;
} loom_low_source_workload_memory_state_t;

static iree_status_t loom_low_source_workload_allocate_static_zero_index(
    loom_builder_t* builder, int64_t** out_static_indices) {
  IREE_ASSERT_ARGUMENT(out_static_indices);
  *out_static_indices = NULL;
  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, 1, sizeof(*static_indices), (void**)&static_indices));
  static_indices[0] = 0;
  *out_static_indices = static_indices;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_vector4xi32_load(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  const loom_low_source_workload_memory_state_t* state =
      (const loom_low_source_workload_memory_state_t*)user_data;
  loom_type_t vector_type = loom_low_source_workload_vector4xi32_type();
  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_workload_allocate_static_zero_index(
      context->builder, &static_indices));

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      context->builder, 0, state->load_view, NULL, 0, static_indices, 1, 0, 0,
      vector_type, LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_vector_load_result(op),
                           vector_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_vector4xi32_store(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  const loom_low_source_workload_memory_state_t* state =
      (const loom_low_source_workload_memory_state_t*)user_data;
  loom_type_t vector_type = loom_low_source_workload_vector4xi32_type();
  loom_value_id_t value = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  if (value == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_workload_allocate_static_zero_index(
      context->builder, &static_indices));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_store_build(
      context->builder, 0, value, state->store_view, NULL, 0, static_indices, 1,
      0, 0, LOOM_LOCATION_UNKNOWN, &op));
  (void)op;
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_low_source_workload_gen_index_madd(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t a = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t b = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t c = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  if (a == LOOM_VALUE_ID_INVALID || b == LOOM_VALUE_ID_INVALID ||
      c == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_madd_build(
      context->builder, a, b, c, loom_low_source_workload_index_type(),
      LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_index_madd_result(op),
                           loom_low_source_workload_index_type());
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static const loom_test_gen_op_hook_t kLoomLowSourceWorkloadHooks[] = {
    {1, loom_low_source_workload_gen_scalar_i32_constant, NULL, NULL},
    {4, loom_low_source_workload_gen_scalar_i32_binary, NULL, NULL},
    {4, loom_low_source_workload_gen_vector4xi32_binary, NULL, NULL},
    {2, loom_low_source_workload_gen_vector4xi32_reduce_addi, NULL, NULL},
    {2, loom_low_source_workload_gen_vector_dot4i_s8s8, NULL, NULL},
    {2, loom_low_source_workload_gen_vector4xi32_extract, NULL, NULL},
    {2, loom_low_source_workload_gen_vector4xi32_shuffle, NULL, NULL},
    {2, loom_low_source_workload_gen_vector4xi32_load, NULL, NULL},
    {2, loom_low_source_workload_gen_vector4xi32_store, NULL, NULL},
    {3, loom_low_source_workload_gen_index_madd, NULL, NULL},
};

//===----------------------------------------------------------------------===//
// Module generation
//===----------------------------------------------------------------------===//

static iree_status_t loom_low_source_workload_generate_module_into(
    loom_test_gen_t* gen, const loom_low_source_workload_config_t* config,
    loom_module_t* module) {
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  iree_string_view_t target_symbol_name =
      loom_low_source_workload_default_string(config->target_symbol_name,
                                              IREE_SV("target"));
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_low_source_workload_add_symbol(
      &builder, target_symbol_name, &target_ref));

  loom_string_id_t preset_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(&builder, config->target_preset, &preset_id));
  loom_op_t* target_op = NULL;
  IREE_RETURN_IF_ERROR(loom_target_profile_build(
      &builder, target_ref, preset_id, loom_named_attr_slice_empty(),
      LOOM_LOCATION_UNKNOWN, &target_op));
  (void)target_op;

  iree_string_view_t function_symbol_name =
      loom_low_source_workload_default_string(config->function_symbol_name,
                                              IREE_SV("generated"));
  loom_symbol_ref_t func_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_low_source_workload_add_symbol(
      &builder, function_symbol_name, &func_ref));
  const loom_type_t arg_types[] = {
      loom_type_buffer(),
      loom_type_buffer(),
      loom_low_source_workload_i32_type(),
      loom_low_source_workload_i32_type(),
      loom_low_source_workload_vector4xi32_type(),
      loom_low_source_workload_vector4xi32_type(),
      loom_low_source_workload_vector16xi8_type(),
      loom_low_source_workload_vector16xi8_type(),
      loom_low_source_workload_index_type(),
      loom_low_source_workload_index_type(),
      loom_low_source_workload_index_type(),
  };
  const loom_type_t result_types[] = {
      loom_low_source_workload_i32_type(),
      loom_low_source_workload_vector4xi32_type(),
      loom_low_source_workload_index_type(),
  };
  loom_op_t* func_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &builder, LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET, 0, 0, 0, target_ref, 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), func_ref, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  loom_region_t* body = loom_func_def_body(func_op);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&builder, func_op, body);

  loom_test_gen_values_t values;
  loom_test_gen_values_initialize(&values);
  loom_block_t* entry_block = loom_region_entry_block(body);
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
    loom_test_gen_values_add(&values, arg_id,
                             loom_module_value_type(module, arg_id));
  }

  loom_op_t* layout_op = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_layout_dense_build(
      &builder, loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      LOOM_LOCATION_UNKNOWN, &layout_op));
  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &builder, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      LOOM_LOCATION_UNKNOWN, &zero_op));
  loom_type_t view_type = loom_low_source_workload_view4xi32_type(
      loom_encoding_layout_dense_result(layout_op));
  loom_low_source_workload_memory_state_t memory_state = {0};
  loom_op_t* load_view_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_buffer_view_build(&builder, loom_block_arg_id(entry_block, 0),
                             loom_index_constant_result(zero_op), view_type,
                             LOOM_LOCATION_UNKNOWN, &load_view_op));
  memory_state.load_view = loom_buffer_view_result(load_view_op);
  loom_op_t* store_view_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_buffer_view_build(&builder, loom_block_arg_id(entry_block, 1),
                             loom_index_constant_result(zero_op), view_type,
                             LOOM_LOCATION_UNKNOWN, &store_view_op));
  memory_state.store_view = loom_buffer_view_result(store_view_op);

  loom_test_gen_body_config_t body_config = {0};
  body_config.op_count = config->op_count;
  body_config.max_nesting_depth = 0;
  body_config.dead_op_probability = 0;
  body_config.value_fan_out = 4;
  loom_test_gen_type_palette_default(&body_config.palette);
  memcpy(body_config.hooks, kLoomLowSourceWorkloadHooks,
         sizeof(kLoomLowSourceWorkloadHooks));
  body_config.hook_count = IREE_ARRAYSIZE(kLoomLowSourceWorkloadHooks);
  for (iree_host_size_t i = 0; i < body_config.hook_count; ++i) {
    if (body_config.hooks[i].generate ==
            loom_low_source_workload_gen_vector4xi32_load ||
        body_config.hooks[i].generate ==
            loom_low_source_workload_gen_vector4xi32_store) {
      body_config.hooks[i].user_data = &memory_state;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_test_gen_body_internal(gen, &body_config, &builder, &values, 0));

  loom_value_id_t returns[] = {
      loom_low_source_workload_pick_latest_exact_type(
          &values, loom_low_source_workload_i32_type()),
      loom_low_source_workload_pick_latest_exact_type(
          &values, loom_low_source_workload_vector4xi32_type()),
      loom_low_source_workload_pick_latest_exact_type(
          &values, loom_low_source_workload_index_type()),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(returns); ++i) {
    IREE_ASSERT(returns[i] != LOOM_VALUE_ID_INVALID);
  }
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_func_return_build(&builder, returns, IREE_ARRAYSIZE(returns),
                             LOOM_LOCATION_UNKNOWN, &return_op));
  (void)return_op;
  loom_builder_restore(&builder, saved_ip);

  return iree_ok_status();
}

iree_status_t loom_low_source_workload_generate_module(
    loom_test_gen_t* gen, const loom_low_source_workload_config_t* config,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(gen);
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  if (iree_string_view_is_empty(config->target_preset)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source workload target preset is required");
  }

  iree_string_view_t module_name = loom_low_source_workload_default_string(
      config->module_name, IREE_SV("source_low_workload"));
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(context, module_name, block_pool,
                                            NULL, context->allocator, &module));

  iree_status_t status =
      loom_low_source_workload_generate_module_into(gen, config, module);
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    loom_module_free(module);
  }
  return status;
}

iree_status_t loom_low_source_workload_generate_seeded_module(
    uint64_t seed, const loom_low_source_workload_config_t* config,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(seed, &gen);
  return loom_low_source_workload_generate_module(&gen, config, context,
                                                  block_pool, out_module);
}

//===----------------------------------------------------------------------===//
// Counting
//===----------------------------------------------------------------------===//

static void loom_low_source_workload_count_op(
    const loom_op_t* op, loom_low_source_workload_counts_t* counts) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_SCALAR_MULI:
      ++counts->scalar_integer_op_count;
      break;
    case LOOM_OP_SCALAR_CONSTANT:
      ++counts->scalar_constant_count;
      break;
    case LOOM_OP_VECTOR_ADDI:
    case LOOM_OP_VECTOR_SUBI:
    case LOOM_OP_VECTOR_MULI:
      ++counts->vector_integer_op_count;
      break;
    case LOOM_OP_VECTOR_REDUCE:
      ++counts->vector_reduce_op_count;
      break;
    case LOOM_OP_VECTOR_DOT4I:
      ++counts->vector_dot_op_count;
      break;
    case LOOM_OP_VECTOR_EXTRACT:
      ++counts->vector_extract_op_count;
      break;
    case LOOM_OP_VECTOR_SHUFFLE:
      ++counts->vector_shuffle_op_count;
      break;
    case LOOM_OP_VECTOR_LOAD:
      ++counts->vector_load_op_count;
      break;
    case LOOM_OP_VECTOR_STORE:
      ++counts->vector_store_op_count;
      break;
    case LOOM_OP_INDEX_MADD:
      ++counts->index_madd_op_count;
      break;
    default:
      break;
  }
}

void loom_low_source_workload_count_func_ops(
    const loom_op_t* func_op, loom_low_source_workload_counts_t* out_counts) {
  IREE_ASSERT_ARGUMENT(func_op);
  IREE_ASSERT_ARGUMENT(out_counts);
  memset(out_counts, 0, sizeof(*out_counts));
  const loom_region_t* body = loom_func_def_body(func_op);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_low_source_workload_count_op(op, out_counts);
    }
  }
}

void loom_low_source_workload_counts_accumulate(
    loom_low_source_workload_counts_t* target_counts,
    const loom_low_source_workload_counts_t* source_counts) {
  IREE_ASSERT_ARGUMENT(target_counts);
  IREE_ASSERT_ARGUMENT(source_counts);
  target_counts->scalar_integer_op_count +=
      source_counts->scalar_integer_op_count;
  target_counts->scalar_constant_count += source_counts->scalar_constant_count;
  target_counts->vector_integer_op_count +=
      source_counts->vector_integer_op_count;
  target_counts->vector_reduce_op_count +=
      source_counts->vector_reduce_op_count;
  target_counts->vector_dot_op_count += source_counts->vector_dot_op_count;
  target_counts->vector_extract_op_count +=
      source_counts->vector_extract_op_count;
  target_counts->vector_shuffle_op_count +=
      source_counts->vector_shuffle_op_count;
  target_counts->vector_load_op_count += source_counts->vector_load_op_count;
  target_counts->vector_store_op_count += source_counts->vector_store_op_count;
  target_counts->index_madd_op_count += source_counts->index_madd_op_count;
}

uint64_t loom_low_source_workload_counts_total(
    const loom_low_source_workload_counts_t* counts) {
  IREE_ASSERT_ARGUMENT(counts);
  return counts->scalar_integer_op_count + counts->scalar_constant_count +
         counts->vector_integer_op_count + counts->vector_reduce_op_count +
         counts->vector_dot_op_count + counts->vector_extract_op_count +
         counts->vector_shuffle_op_count + counts->vector_load_op_count +
         counts->vector_store_op_count + counts->index_madd_op_count;
}

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

static iree_status_t loom_low_source_workload_verify_source_module(
    const loom_module_t* module) {
  loom_verify_options_t options = {0};
  loom_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_verify_module(module, &options, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated source failed verification");
  }
  return iree_ok_status();
}

static void loom_low_source_workload_count_module_source_ops(
    const loom_module_t* module,
    loom_low_source_workload_counts_t* out_counts) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_counts);
  memset(out_counts, 0, sizeof(*out_counts));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* op = module->symbols.entries[i].defining_op;
    if (!loom_func_def_isa(op)) {
      continue;
    }
    loom_low_source_workload_counts_t func_counts;
    loom_low_source_workload_count_func_ops(op, &func_counts);
    loom_low_source_workload_counts_accumulate(out_counts, &func_counts);
  }
}

static iree_status_t loom_low_source_workload_verify_low_module(
    const loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options) {
  const loom_low_verify_options_t verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = options->descriptor_registry,
      .descriptor_requirements = options->descriptor_requirements,
      .max_errors = 20,
  };
  loom_low_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &verify_options, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated low function failed verification");
  }
  return iree_ok_status();
}

static uint32_t loom_low_source_workload_count_low_descriptor_ops(
    const loom_op_t* low_func_op) {
  uint32_t count = 0;
  const loom_region_t* body = loom_low_func_def_body(low_func_op);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_op_isa(op) || loom_low_const_isa(op)) {
        ++count;
      }
    }
  }
  return count;
}

iree_status_t loom_low_source_workload_run_pipeline(
    loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_low_source_workload_pipeline_counters_t* out_counters) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->descriptor_registry);
  IREE_ASSERT_ARGUMENT(options->policy_registry);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_counters);
  memset(out_counters, 0, sizeof(*out_counters));

  loom_low_source_workload_count_module_source_ops(
      module, &out_counters->source_counts);
  iree_status_t status = loom_low_source_workload_verify_source_module(module);

  iree_arena_allocator_t lowering_arena;
  bool lowering_arena_initialized = false;
  loom_low_lower_result_t lower_result = {0};
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &lowering_arena);
    lowering_arena_initialized = true;
    const loom_low_source_selection_options_t selection_options = {
        .descriptor_registry = options->descriptor_registry,
        .policy_registry = options->policy_registry,
    };
    loom_low_source_selection_list_t selection_list = {0};
    status = loom_low_select_source_funcs(module, &selection_options,
                                          &lowering_arena, &selection_list);
    loom_op_t** lowered_funcs = NULL;
    if (iree_status_is_ok(status) && selection_list.count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "generated workload has no compatible source functions");
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(&lowering_arena, selection_list.count,
                                         sizeof(*lowered_funcs),
                                         (void**)&lowered_funcs);
    }
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      const loom_low_source_selection_t* selection = &selection_list.values[i];
      const loom_low_lower_options_t lower_options = {
          .target_ref = selection->target_ref,
          .bundle = selection->target_bundle,
          .descriptor_registry = options->descriptor_registry,
          .descriptor_requirements = options->descriptor_requirements,
          .policy = selection->policy,
          .max_errors = 20,
      };
      loom_low_lower_result_t func_lower_result = {0};
      status = loom_low_lower_function(module, selection->func, &lower_options,
                                       &func_lower_result);
      lower_result.error_count += func_lower_result.error_count;
      lower_result.remark_count += func_lower_result.remark_count;
      if (iree_status_is_ok(status) && func_lower_result.error_count != 0) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "source lowering produced errors");
      } else if (iree_status_is_ok(status) &&
                 func_lower_result.low_func_op == NULL) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "source lowering did not emit low.func.def");
      }
      if (iree_status_is_ok(status)) {
        lowered_funcs[i] = func_lower_result.low_func_op;
      }
    }
    if (iree_status_is_ok(status)) {
      out_counters->lower_error_count = lower_result.error_count;
      out_counters->lower_remark_count = lower_result.remark_count;
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_source_workload_verify_low_module(module, options);
    }
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      out_counters->low_descriptor_op_count +=
          loom_low_source_workload_count_low_descriptor_ops(lowered_funcs[i]);
    }

    iree_arena_allocator_t packet_arena;
    bool packet_arena_initialized = false;
    if (iree_status_is_ok(status)) {
      iree_arena_initialize(block_pool, &packet_arena);
      packet_arena_initialized = true;
    }
    const loom_low_packetization_options_t packet_options = {
        .descriptor_registry = options->descriptor_registry,
        .schedule_strategy = options->schedule_strategy,
    };
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      loom_low_packetization_t packetization = {0};
      status =
          loom_low_packetize_function(module, lowered_funcs[i], &packet_options,
                                      &packet_arena, &packetization);
      if (iree_status_is_ok(status)) {
        out_counters->schedule_node_count +=
            packetization.schedule.scheduled_node_count;
        out_counters->schedule_dependency_count +=
            packetization.schedule.dependency_count;
        out_counters->schedule_resource_use_count +=
            packetization.schedule.resource_use_count;
        out_counters->schedule_hazard_gap_count +=
            packetization.schedule.hazard_gap_count;
        out_counters->allocation_assignment_count +=
            packetization.allocation.assignment_count;
        out_counters->allocation_spill_count +=
            packetization.allocation.spill_count;
        out_counters->allocation_coalesced_copy_count +=
            packetization.allocation.coalesced_copy_count;
        out_counters->allocation_materialized_copy_count +=
            packetization.allocation.materialized_copy_count;
      }
    }
    if (packet_arena_initialized) {
      out_counters->packet_arena_used_bytes = packet_arena.used_allocation_size;
      iree_arena_deinitialize(&packet_arena);
    }
  }

  out_counters->module_arena_used_bytes = module->arena.used_allocation_size;
  out_counters->module_arena_allocated_bytes =
      module->arena.total_allocation_size;
  if (lowering_arena_initialized) {
    out_counters->lowering_arena_used_bytes =
        lowering_arena.used_allocation_size;
  }
  if (lowering_arena_initialized) {
    iree_arena_deinitialize(&lowering_arena);
  }
  return status;
}
