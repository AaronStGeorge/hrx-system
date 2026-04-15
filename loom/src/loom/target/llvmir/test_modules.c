// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/test_modules.h"

#include "loom/target/llvmir/llvmir.h"

#define LOOM_LLVMIR_TEST_MODULE_SCENARIO_COUNT 11

static loom_llvmir_attr_t loom_llvmir_test_attr(loom_llvmir_attr_kind_t kind) {
  return (loom_llvmir_attr_t){
      .kind = kind,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
}

static loom_llvmir_attr_t loom_llvmir_test_align_attr(uint64_t value) {
  return (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_ALIGN,
      .value = value,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
}

static loom_llvmir_attr_t loom_llvmir_test_range_attr(
    loom_llvmir_type_id_t type_id, uint64_t lower, uint64_t upper) {
  return (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_RANGE,
      .value = lower,
      .value2 = upper,
      .type_id = type_id,
  };
}

static loom_llvmir_attr_t loom_llvmir_test_string_key_attr(const char* key) {
  return (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_STRING_KEY,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
      .key = IREE_SV(key),
  };
}

static loom_llvmir_attr_t loom_llvmir_test_string_key_value_attr(
    const char* key, const char* value) {
  return (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_STRING_KEY_VALUE,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
      .key = IREE_SV(key),
      .string_value = IREE_SV(value),
  };
}

static iree_status_t loom_llvmir_test_populate_object_vadd4(
    loom_llvmir_module_t* module) {
  const loom_llvmir_target_env_t* target_env =
      loom_llvmir_target_env_x86_64_unknown_linux_gnu();

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v4f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, target_env->address_spaces.generic, &ptr_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_vector_type(module, 4, f32_type, &v4f32_type));

  loom_llvmir_function_t* function = NULL;
  loom_llvmir_function_desc_t function_desc = {
      .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
      .name = IREE_SV("vadd4_object"),
      .return_type = void_type,
      .linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
      .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_add_function(module, &function_desc, &function));

  loom_llvmir_attr_t readonly_attrs[] = {
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOALIAS),
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOUNDEF),
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_READONLY),
  };
  loom_llvmir_attr_t writeonly_attrs[] = {
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOALIAS),
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOUNDEF),
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_WRITEONLY),
  };
  loom_llvmir_attr_t noundef_attrs[] = {
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOUNDEF),
  };
  loom_llvmir_value_id_t x = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t z = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t base = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = ptr_type,
          .name = IREE_SV("x"),
          .attrs = readonly_attrs,
          .attr_count = IREE_ARRAYSIZE(readonly_attrs),
      },
      &x));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = ptr_type,
          .name = IREE_SV("y"),
          .attrs = readonly_attrs,
          .attr_count = IREE_ARRAYSIZE(readonly_attrs),
      },
      &y));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = ptr_type,
          .name = IREE_SV("z"),
          .attrs = writeonly_attrs,
          .attr_count = IREE_ARRAYSIZE(writeonly_attrs),
      },
      &z));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = i64_type,
          .name = IREE_SV("base"),
          .attrs = noundef_attrs,
          .attr_count = IREE_ARRAYSIZE(noundef_attrs),
      },
      &base));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t x_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t z_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_gep(entry,
                            &(loom_llvmir_gep_desc_t){
                                .result_name = IREE_SV("x.ptr"),
                                .result_type = ptr_type,
                                .element_type = f32_type,
                                .base = x,
                                .indices = &base,
                                .index_count = 1,
                            },
                            &x_ptr));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_gep(entry,
                            &(loom_llvmir_gep_desc_t){
                                .result_name = IREE_SV("y.ptr"),
                                .result_type = ptr_type,
                                .element_type = f32_type,
                                .base = y,
                                .indices = &base,
                                .index_count = 1,
                            },
                            &y_ptr));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_gep(entry,
                            &(loom_llvmir_gep_desc_t){
                                .result_name = IREE_SV("z.ptr"),
                                .result_type = ptr_type,
                                .element_type = f32_type,
                                .base = z,
                                .indices = &base,
                                .index_count = 1,
                            },
                            &z_ptr));
  loom_llvmir_value_id_t x_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t sum = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_load(entry,
                                              &(loom_llvmir_load_desc_t){
                                                  .result_name = IREE_SV("x.v"),
                                                  .result_type = v4f32_type,
                                                  .pointer = x_ptr,
                                                  .alignment = 4,
                                              },
                                              &x_value));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_load(entry,
                                              &(loom_llvmir_load_desc_t){
                                                  .result_name = IREE_SV("y.v"),
                                                  .result_type = v4f32_type,
                                                  .pointer = y_ptr,
                                                  .alignment = 4,
                                              },
                                              &y_value));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("z.v"),
                                  .result_type = v4f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FADD,
                                  .lhs = x_value,
                                  .rhs = y_value,
                              },
                              &sum));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_store(entry, &(loom_llvmir_store_desc_t){
                                         .value = sum,
                                         .pointer = z_ptr,
                                         .alignment = 4,
                                     }));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret_void(entry));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_call_constants(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));

  loom_llvmir_function_t* callee = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
          .name = IREE_SV("callee_i32"),
          .return_type = i32_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &callee));
  loom_llvmir_value_id_t callee_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(callee,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("value"),
                                         },
                                         &callee_value));

  loom_llvmir_function_t* sink = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
          .name = IREE_SV("sink_i32"),
          .return_type = void_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &sink));
  loom_llvmir_value_id_t sink_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(sink,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("value"),
                                         },
                                         &sink_value));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("call_i32"),
          .return_type = i32_type,
          .linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));
  loom_llvmir_attr_t noundef_attrs[] = {
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOUNDEF),
  };
  loom_llvmir_value_id_t x = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = i32_type,
          .name = IREE_SV("x"),
          .attrs = noundef_attrs,
          .attr_count = IREE_ARRAYSIZE(noundef_attrs),
      },
      &x));

  loom_llvmir_value_id_t constant_42 = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
      module, i32_type, 42, &constant_42));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t y = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_call(entry,
                             &(loom_llvmir_call_desc_t){
                                 .result_name = IREE_SV("y"),
                                 .callee = loom_llvmir_function_id(callee),
                                 .args = &constant_42,
                                 .arg_count = 1,
                             },
                             &y));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_call(entry,
                             &(loom_llvmir_call_desc_t){
                                 .callee = loom_llvmir_function_id(sink),
                                 .args = &x,
                                 .arg_count = 1,
                             },
                             NULL));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret(entry, y));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_builtin_intrinsics(
    loom_llvmir_module_t* module) {
  const loom_llvmir_target_env_t* target_env =
      loom_llvmir_target_env_x86_64_unknown_linux_gnu();

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i1_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i8_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, target_env->address_spaces.generic, &ptr_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 1, &i1_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 8, &i8_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));

  loom_llvmir_function_t* memcpy_function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_declare_memcpy(
      module, target_env->address_spaces.generic,
      target_env->address_spaces.generic, 64, &memcpy_function));
  loom_llvmir_function_t* memset_function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_declare_memset(
      module, target_env->address_spaces.generic, 64, &memset_function));
  loom_llvmir_function_t* lifetime_start = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_declare_lifetime_start(
      module, target_env->address_spaces.generic, &lifetime_start));
  loom_llvmir_function_t* lifetime_end = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_declare_lifetime_end(
      module, target_env->address_spaces.generic, &lifetime_end));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("memory_ops"),
          .return_type = void_type,
          .linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));
  loom_llvmir_value_id_t target = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = ptr_type,
                                             .name = IREE_SV("target"),
                                         },
                                         &target));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = ptr_type,
                                             .name = IREE_SV("source"),
                                         },
                                         &source));

  loom_llvmir_value_id_t size = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t zero_byte = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t false_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_add_integer_constant(module, i64_type, 16, &size));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_add_integer_constant(module, i8_type, 0, &zero_byte));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
      module, i1_type, 0, &false_value));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t lifetime_args[] = {size, target};
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .callee = loom_llvmir_function_id(lifetime_start),
          .args = lifetime_args,
          .arg_count = IREE_ARRAYSIZE(lifetime_args),
      },
      NULL));
  loom_llvmir_value_id_t memset_args[] = {target, zero_byte, size, false_value};
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .callee = loom_llvmir_function_id(memset_function),
          .args = memset_args,
          .arg_count = IREE_ARRAYSIZE(memset_args),
      },
      NULL));
  loom_llvmir_value_id_t memcpy_args[] = {target, source, size, false_value};
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .callee = loom_llvmir_function_id(memcpy_function),
          .args = memcpy_args,
          .arg_count = IREE_ARRAYSIZE(memcpy_args),
      },
      NULL));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .callee = loom_llvmir_function_id(lifetime_end),
          .args = lifetime_args,
          .arg_count = IREE_ARRAYSIZE(lifetime_args),
      },
      NULL));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret_void(entry));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_cfg_phi(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t i1_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 1, &i1_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("select_i32"),
          .return_type = i32_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));
  loom_llvmir_value_id_t condition = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i1_type,
                                             .name = IREE_SV("c"),
                                         },
                                         &condition));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("a"),
                                         },
                                         &lhs));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("b"),
                                         },
                                         &rhs));

  loom_llvmir_block_t* entry = NULL;
  loom_llvmir_block_t* then_block = NULL;
  loom_llvmir_block_t* else_block = NULL;
  loom_llvmir_block_t* join_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("then"), &then_block));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("else"), &else_block));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("join"), &join_block));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cond_br(
      entry, condition, loom_llvmir_block_id(then_block),
      loom_llvmir_block_id(else_block)));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_br(then_block, loom_llvmir_block_id(join_block)));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_br(else_block, loom_llvmir_block_id(join_block)));
  loom_llvmir_phi_incoming_t incoming[] = {
      {.value = lhs, .predecessor = loom_llvmir_block_id(then_block)},
      {.value = rhs, .predecessor = loom_llvmir_block_id(else_block)},
  };
  loom_llvmir_value_id_t selected = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_phi(join_block,
                            &(loom_llvmir_phi_desc_t){
                                .result_name = IREE_SV("x"),
                                .result_type = i32_type,
                                .incoming = incoming,
                                .incoming_count = IREE_ARRAYSIZE(incoming),
                            },
                            &selected));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret(join_block, selected));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_scalar_binop(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("binops"),
          .return_type = i32_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));

  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t xf = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t yf = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("lhs"),
                                         },
                                         &lhs));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("rhs"),
                                         },
                                         &rhs));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = f32_type,
                                             .name = IREE_SV("xf"),
                                         },
                                         &xf));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = f32_type,
                                             .name = IREE_SV("yf"),
                                         },
                                         &yf));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t sum = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_binop(
      entry,
      &(loom_llvmir_binop_desc_t){
          .result_name = IREE_SV("sum"),
          .result_type = i32_type,
          .op = LOOM_LLVMIR_BINOP_ADD,
          .lhs = lhs,
          .rhs = rhs,
          .integer_flags = LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP |
                           LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP,
      },
      &sum));
  loom_llvmir_value_id_t difference = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("difference"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_SUB,
                                  .lhs = sum,
                                  .rhs = rhs,
                              },
                              &difference));
  loom_llvmir_value_id_t product = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("product"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_MUL,
                                  .lhs = difference,
                                  .rhs = lhs,
                              },
                              &product));
  loom_llvmir_value_id_t unsigned_quotient = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("unsigned_quotient"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_UDIV,
                                  .lhs = product,
                                  .rhs = rhs,
                              },
                              &unsigned_quotient));
  loom_llvmir_value_id_t signed_quotient = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_binop(
      entry,
      &(loom_llvmir_binop_desc_t){
          .result_name = IREE_SV("signed_quotient"),
          .result_type = i32_type,
          .op = LOOM_LLVMIR_BINOP_SDIV,
          .lhs = product,
          .rhs = rhs,
          .integer_flags = LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT,
      },
      &signed_quotient));
  loom_llvmir_value_id_t unsigned_remainder = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("unsigned_remainder"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_UREM,
                                  .lhs = unsigned_quotient,
                                  .rhs = rhs,
                              },
                              &unsigned_remainder));
  loom_llvmir_value_id_t signed_remainder = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("signed_remainder"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_SREM,
                                  .lhs = signed_quotient,
                                  .rhs = rhs,
                              },
                              &signed_remainder));
  loom_llvmir_value_id_t bits = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("bits"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_AND,
                                  .lhs = unsigned_remainder,
                                  .rhs = signed_remainder,
                              },
                              &bits));
  loom_llvmir_value_id_t either = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("either"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_OR,
                                  .lhs = bits,
                                  .rhs = lhs,
                              },
                              &either));
  loom_llvmir_value_id_t toggled = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("toggled"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_XOR,
                                  .lhs = either,
                                  .rhs = rhs,
                              },
                              &toggled));
  loom_llvmir_value_id_t shifted = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("shifted"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_SHL,
                                  .lhs = toggled,
                                  .rhs = rhs,
                              },
                              &shifted));
  loom_llvmir_value_id_t logical_shifted = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("logical_shifted"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_LSHR,
                                  .lhs = shifted,
                                  .rhs = rhs,
                              },
                              &logical_shifted));
  loom_llvmir_value_id_t arithmetic_shifted = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("arithmetic_shifted"),
                                  .result_type = i32_type,
                                  .op = LOOM_LLVMIR_BINOP_ASHR,
                                  .lhs = logical_shifted,
                                  .rhs = rhs,
                              },
                              &arithmetic_shifted));
  loom_llvmir_value_id_t float_sum = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("float_sum"),
                                  .result_type = f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FADD,
                                  .lhs = xf,
                                  .rhs = yf,
                                  .fast_math_flags = LOOM_LLVMIR_FAST_MATH_FAST,
                              },
                              &float_sum));
  loom_llvmir_value_id_t float_difference = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("float_difference"),
                                  .result_type = f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FSUB,
                                  .lhs = float_sum,
                                  .rhs = yf,
                              },
                              &float_difference));
  loom_llvmir_value_id_t float_product = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("float_product"),
                                  .result_type = f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FMUL,
                                  .lhs = float_difference,
                                  .rhs = xf,
                              },
                              &float_product));
  loom_llvmir_value_id_t float_quotient = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("float_quotient"),
                                  .result_type = f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FDIV,
                                  .lhs = float_product,
                                  .rhs = yf,
                              },
                              &float_quotient));
  loom_llvmir_value_id_t float_remainder = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("float_remainder"),
                                  .result_type = f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FREM,
                                  .lhs = float_quotient,
                                  .rhs = yf,
                              },
                              &float_remainder));
  loom_llvmir_value_id_t float_negated = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_unop(
      entry,
      &(loom_llvmir_unop_desc_t){
          .result_name = IREE_SV("float_negated"),
          .result_type = f32_type,
          .op = LOOM_LLVMIR_UNOP_FNEG,
          .value = float_remainder,
          .fast_math_flags = LOOM_LLVMIR_FAST_MATH_NO_SIGNED_ZEROS,
      },
      &float_negated));
  (void)float_negated;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret(entry, arithmetic_shifted));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_vector_elements(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v4f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_vector_type(module, 4, f32_type, &v4f32_type));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("vector_elements"),
          .return_type = v4f32_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));

  loom_llvmir_value_id_t vector = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t index = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = v4f32_type,
                                             .name = IREE_SV("vector"),
                                         },
                                         &vector));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = f32_type,
                                             .name = IREE_SV("value"),
                                         },
                                         &value));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("index"),
                                         },
                                         &index));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t lane = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_extract_element(entry,
                                        &(loom_llvmir_extract_element_desc_t){
                                            .result_name = IREE_SV("lane"),
                                            .result_type = f32_type,
                                            .vector = vector,
                                            .index = index,
                                        },
                                        &lane));
  loom_llvmir_value_id_t sum = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("sum"),
                                  .result_type = f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FADD,
                                  .lhs = lane,
                                  .rhs = value,
                              },
                              &sum));
  loom_llvmir_value_id_t updated = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_insert_element(entry,
                                       &(loom_llvmir_insert_element_desc_t){
                                           .result_name = IREE_SV("updated"),
                                           .result_type = v4f32_type,
                                           .vector = vector,
                                           .element = sum,
                                           .index = index,
                                       },
                                       &updated));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret(entry, updated));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_shuffle_vector(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v4i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v4f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_vector_type(module, 4, i32_type, &v4i32_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_vector_type(module, 4, f32_type, &v4f32_type));

  uint64_t mask_values[] = {0, 4, 1, 5};
  loom_llvmir_value_id_t mask = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_vector_constant(
      module, v4i32_type, mask_values, IREE_ARRAYSIZE(mask_values), &mask));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("shuffle_vector"),
          .return_type = v4f32_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));

  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = v4f32_type,
                                             .name = IREE_SV("lhs"),
                                         },
                                         &lhs));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = v4f32_type,
                                             .name = IREE_SV("rhs"),
                                         },
                                         &rhs));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t interleaved = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_shuffle_vector(
      entry,
      &(loom_llvmir_shuffle_vector_desc_t){
          .result_name = IREE_SV("interleaved"),
          .result_type = v4f32_type,
          .lhs = lhs,
          .rhs = rhs,
          .mask = mask,
      },
      &interleaved));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret(entry, interleaved));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_compare_select(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t i1_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 1, &i1_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("compare_select"),
          .return_type = i32_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));

  loom_llvmir_value_id_t x = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t lower = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t upper = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t xf = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t yf = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("x"),
                                         },
                                         &x));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("lower"),
                                         },
                                         &lower));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("upper"),
                                         },
                                         &upper));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = f32_type,
                                             .name = IREE_SV("xf"),
                                         },
                                         &xf));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = f32_type,
                                             .name = IREE_SV("yf"),
                                         },
                                         &yf));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t below = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_icmp(entry,
                             &(loom_llvmir_icmp_desc_t){
                                 .result_name = IREE_SV("below"),
                                 .result_type = i1_type,
                                 .predicate = LOOM_LLVMIR_ICMP_SLT,
                                 .lhs = x,
                                 .rhs = lower,
                             },
                             &below));
  loom_llvmir_value_id_t above = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_icmp(entry,
                             &(loom_llvmir_icmp_desc_t){
                                 .result_name = IREE_SV("above"),
                                 .result_type = i1_type,
                                 .predicate = LOOM_LLVMIR_ICMP_SGT,
                                 .lhs = x,
                                 .rhs = upper,
                             },
                             &above));
  loom_llvmir_value_id_t ordered = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_fcmp(entry,
                             &(loom_llvmir_fcmp_desc_t){
                                 .result_name = IREE_SV("ordered"),
                                 .result_type = i1_type,
                                 .predicate = LOOM_LLVMIR_FCMP_OLT,
                                 .lhs = xf,
                                 .rhs = yf,
                             },
                             &ordered));
  loom_llvmir_value_id_t at_least_lower = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_select(entry,
                               &(loom_llvmir_select_desc_t){
                                   .result_name = IREE_SV("at_least_lower"),
                                   .result_type = i32_type,
                                   .condition = below,
                                   .true_value = lower,
                                   .false_value = x,
                               },
                               &at_least_lower));
  loom_llvmir_value_id_t clamped = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_select(entry,
                               &(loom_llvmir_select_desc_t){
                                   .result_name = IREE_SV("clamped"),
                                   .result_type = i32_type,
                                   .condition = above,
                                   .true_value = upper,
                                   .false_value = at_least_lower,
                               },
                               &clamped));
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_select(entry,
                               &(loom_llvmir_select_desc_t){
                                   .result_name = IREE_SV("result"),
                                   .result_type = i32_type,
                                   .condition = ordered,
                                   .true_value = clamped,
                                   .false_value = x,
                               },
                               &result));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret(entry, result));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_casts(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i16_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_global_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 16, &i16_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F64, &f64_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_pointer_type(module, 0, &ptr_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_pointer_type(module, 1, &ptr_global_type));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("casts"),
          .return_type = void_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));

  loom_llvmir_value_id_t wide = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t narrow = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t scalar = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t wide_scalar = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i64_type,
                                             .name = IREE_SV("wide"),
                                         },
                                         &wide));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i16_type,
                                             .name = IREE_SV("narrow"),
                                         },
                                         &narrow));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = f32_type,
                                             .name = IREE_SV("scalar"),
                                         },
                                         &scalar));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = f64_type,
                                             .name = IREE_SV("wide_scalar"),
                                         },
                                         &wide_scalar));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = ptr_type,
                                             .name = IREE_SV("pointer"),
                                         },
                                         &pointer));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("truncated"),
                                 .result_type = i32_type,
                                 .op = LOOM_LLVMIR_CAST_TRUNCATE,
                                 .value = wide,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("zero_extended"),
                                 .result_type = i64_type,
                                 .op = LOOM_LLVMIR_CAST_ZERO_EXTEND,
                                 .value = narrow,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("sign_extended"),
                                 .result_type = i64_type,
                                 .op = LOOM_LLVMIR_CAST_SIGN_EXTEND,
                                 .value = narrow,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("fp_truncated"),
                                 .result_type = f32_type,
                                 .op = LOOM_LLVMIR_CAST_FP_TRUNCATE,
                                 .value = wide_scalar,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("fp_extended"),
                                 .result_type = f64_type,
                                 .op = LOOM_LLVMIR_CAST_FP_EXTEND,
                                 .value = scalar,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("unsigned_to_fp"),
                                 .result_type = f32_type,
                                 .op = LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP,
                                 .value = narrow,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("signed_to_fp"),
                                 .result_type = f32_type,
                                 .op = LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP,
                                 .value = narrow,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("fp_to_unsigned"),
                                 .result_type = i32_type,
                                 .op = LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT,
                                 .value = scalar,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("fp_to_signed"),
                                 .result_type = i32_type,
                                 .op = LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT,
                                 .value = scalar,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("ptr_to_int"),
                                 .result_type = i64_type,
                                 .op = LOOM_LLVMIR_CAST_PTR_TO_INT,
                                 .value = pointer,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("int_to_ptr"),
                                 .result_type = ptr_type,
                                 .op = LOOM_LLVMIR_CAST_INT_TO_PTR,
                                 .value = wide,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("bits"),
                                 .result_type = i32_type,
                                 .op = LOOM_LLVMIR_CAST_BITCAST,
                                 .value = scalar,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_cast(entry,
                             &(loom_llvmir_cast_desc_t){
                                 .result_name = IREE_SV("global_pointer"),
                                 .result_type = ptr_global_type,
                                 .op = LOOM_LLVMIR_CAST_ADDRESS_SPACE_CAST,
                                 .value = pointer,
                             },
                             &result));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret_void(entry));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_inline_asm(
    loom_llvmir_module_t* module) {
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("inline_asm_add"),
          .return_type = i32_type,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));
  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("lhs"),
                                         },
                                         &lhs));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("rhs"),
                                         },
                                         &rhs));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t args[] = {lhs, rhs};
  loom_llvmir_value_id_t sum = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_inline_asm(
      entry,
      &(loom_llvmir_inline_asm_desc_t){
          .result_name = IREE_SV("sum"),
          .result_type = i32_type,
          .flags = LOOM_LLVMIR_INLINE_ASM_SIDE_EFFECT,
          .asm_template = IREE_SV("addl $2, $0"),
          .constraints = IREE_SV("=r,0,r"),
          .args = args,
          .arg_count = IREE_ARRAYSIZE(args),
      },
      &sum));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret(entry, sum));
  return iree_ok_status();
}

static iree_status_t loom_llvmir_test_populate_amdgpu_intrinsics(
    loom_llvmir_module_t* module) {
  const loom_llvmir_target_env_t* target_env =
      loom_llvmir_target_env_amdgcn_amd_amdhsa();

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i16_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v1f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t global_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t fat_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 16, &i16_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_vector_type(module, 1, f32_type, &v1f32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, target_env->address_spaces.global, &global_ptr_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, target_env->address_spaces.buffer_resource, &fat_ptr_type));

  loom_llvmir_attr_t kernel_attrs[] = {
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_ALWAYSINLINE),
      loom_llvmir_test_string_key_value_attr("amdgpu-flat-work-group-size",
                                             "64,64"),
      loom_llvmir_test_string_key_attr("uniform-work-group-size"),
  };
  loom_llvmir_attr_group_id_t kernel_attr_group =
      LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_attr_group(
      module, kernel_attrs, IREE_ARRAYSIZE(kernel_attrs), &kernel_attr_group));
  int32_t workgroup_size_values[] = {64, 1, 1};
  loom_llvmir_metadata_id_t workgroup_size = LOOM_LLVMIR_METADATA_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_metadata_i32_tuple(
      module,
      &(loom_llvmir_metadata_i32_tuple_t){
          .values = workgroup_size_values,
          .value_count = IREE_ARRAYSIZE(workgroup_size_values),
      },
      &workgroup_size));

  loom_llvmir_function_t* kernel = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("add_dispatch"),
          .return_type = void_type,
          .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL,
          .attr_group_id = kernel_attr_group,
      },
      &kernel));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_metadata_attachment(
      kernel, &(loom_llvmir_metadata_attachment_t){
                  .name = IREE_SV("reqd_work_group_size"),
                  .metadata_id = workgroup_size,
              }));
  loom_llvmir_attr_t binding_attrs[] = {
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_INREG),
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOALIAS),
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NOUNDEF),
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_NONNULL),
      loom_llvmir_test_align_attr(16),
  };
  loom_llvmir_value_id_t x = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t z = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      kernel,
      &(loom_llvmir_parameter_desc_t){
          .type_id = global_ptr_type,
          .name = IREE_SV("x"),
          .attrs = binding_attrs,
          .attr_count = IREE_ARRAYSIZE(binding_attrs),
      },
      &x));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      kernel,
      &(loom_llvmir_parameter_desc_t){
          .type_id = global_ptr_type,
          .name = IREE_SV("y"),
          .attrs = binding_attrs,
          .attr_count = IREE_ARRAYSIZE(binding_attrs),
      },
      &y));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      kernel,
      &(loom_llvmir_parameter_desc_t){
          .type_id = global_ptr_type,
          .name = IREE_SV("z"),
          .attrs = binding_attrs,
          .attr_count = IREE_ARRAYSIZE(binding_attrs),
      },
      &z));

  loom_llvmir_function_t* workitem_id = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_declare_amdgcn_workitem_id_x(module, &workitem_id));
  loom_llvmir_function_t* make_buffer_resource = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_declare_amdgcn_make_buffer_rsrc(
      module, 7, 1, &make_buffer_resource));

  loom_llvmir_value_id_t stride_zero = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t records = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t flags = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
      module, i16_type, 0, &stride_zero));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_add_integer_constant(module, i32_type, 64, &records));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(module, i32_type,
                                                               159744, &flags));

  loom_llvmir_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_block(kernel, IREE_SV("entry"), &entry));
  loom_llvmir_attr_t tid_range[] = {
      loom_llvmir_test_range_attr(i32_type, 0, 64),
  };
  loom_llvmir_value_id_t tid = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_call(entry,
                             &(loom_llvmir_call_desc_t){
                                 .result_name = IREE_SV("tid"),
                                 .callee = loom_llvmir_function_id(workitem_id),
                                 .result_attrs = tid_range,
                                 .result_attr_count = IREE_ARRAYSIZE(tid_range),
                             },
                             &tid));

  loom_llvmir_value_id_t x_resource = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y_resource = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t z_resource = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t resource_args[] = {x, stride_zero, records, flags};
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .result_name = IREE_SV("x.rsrc"),
          .callee = loom_llvmir_function_id(make_buffer_resource),
          .args = resource_args,
          .arg_count = IREE_ARRAYSIZE(resource_args),
      },
      &x_resource));
  resource_args[0] = y;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .result_name = IREE_SV("y.rsrc"),
          .callee = loom_llvmir_function_id(make_buffer_resource),
          .args = resource_args,
          .arg_count = IREE_ARRAYSIZE(resource_args),
      },
      &y_resource));
  resource_args[0] = z;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .result_name = IREE_SV("z.rsrc"),
          .callee = loom_llvmir_function_id(make_buffer_resource),
          .args = resource_args,
          .arg_count = IREE_ARRAYSIZE(resource_args),
      },
      &z_resource));

  loom_llvmir_value_id_t x_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t z_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_gep(entry,
                            &(loom_llvmir_gep_desc_t){
                                .result_name = IREE_SV("x.ptr"),
                                .result_type = fat_ptr_type,
                                .element_type = f32_type,
                                .base = x_resource,
                                .indices = &tid,
                                .index_count = 1,
                            },
                            &x_ptr));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_gep(entry,
                            &(loom_llvmir_gep_desc_t){
                                .result_name = IREE_SV("y.ptr"),
                                .result_type = fat_ptr_type,
                                .element_type = f32_type,
                                .base = y_resource,
                                .indices = &tid,
                                .index_count = 1,
                            },
                            &y_ptr));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_gep(entry,
                            &(loom_llvmir_gep_desc_t){
                                .result_name = IREE_SV("z.ptr"),
                                .result_type = fat_ptr_type,
                                .element_type = f32_type,
                                .base = z_resource,
                                .indices = &tid,
                                .index_count = 1,
                            },
                            &z_ptr));
  loom_llvmir_value_id_t x_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t z_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_load(entry,
                             &(loom_llvmir_load_desc_t){
                                 .result_name = IREE_SV("x.value"),
                                 .result_type = v1f32_type,
                                 .pointer = x_ptr,
                                 .alignment = 4,
                             },
                             &x_value));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_load(entry,
                             &(loom_llvmir_load_desc_t){
                                 .result_name = IREE_SV("y.value"),
                                 .result_type = v1f32_type,
                                 .pointer = y_ptr,
                                 .alignment = 4,
                             },
                             &y_value));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("z.value"),
                                  .result_type = v1f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FADD,
                                  .lhs = x_value,
                                  .rhs = y_value,
                              },
                              &z_value));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_store(entry, &(loom_llvmir_store_desc_t){
                                         .value = z_value,
                                         .pointer = z_ptr,
                                         .alignment = 4,
                                     }));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_ret_void(entry));
  return iree_ok_status();
}

static const char kObjectVadd4Text[] =
    "source_filename = \"loom-object\"\n"
    "target triple = \"x86_64-unknown-linux-gnu\"\n"
    "\n"
    "define dso_local void @vadd4_object(ptr noalias noundef readonly "
    "%x, ptr noalias noundef readonly %y, ptr noalias noundef "
    "writeonly %z, i64 noundef %base) {\n"
    "entry:\n"
    "  %x.ptr = getelementptr float, ptr %x, i64 %base\n"
    "  %y.ptr = getelementptr float, ptr %y, i64 %base\n"
    "  %z.ptr = getelementptr float, ptr %z, i64 %base\n"
    "  %x.v = load <4 x float>, ptr %x.ptr, align 4\n"
    "  %y.v = load <4 x float>, ptr %y.ptr, align 4\n"
    "  %z.v = fadd <4 x float> %x.v, %y.v\n"
    "  store <4 x float> %z.v, ptr %z.ptr, align 4\n"
    "  ret void\n"
    "}\n";

static const char kCallConstantsText[] =
    "source_filename = \"loom-call-constants\"\n"
    "target triple = \"x86_64-unknown-linux-gnu\"\n"
    "\n"
    "declare i32 @callee_i32(i32 %value)\n"
    "\n"
    "declare void @sink_i32(i32 %value)\n"
    "\n"
    "define dso_local i32 @call_i32(i32 noundef %x) {\n"
    "entry:\n"
    "  %y = call i32 @callee_i32(i32 42)\n"
    "  call void @sink_i32(i32 %x)\n"
    "  ret i32 %y\n"
    "}\n";

static const char kBuiltinIntrinsicsText[] =
    "source_filename = \"loom-builtins\"\n"
    "target triple = \"x86_64-unknown-linux-gnu\"\n"
    "\n"
    "declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, "
    "ptr noalias nocapture readonly, i64, i1 immarg)\n"
    "\n"
    "declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, "
    "i1 immarg)\n"
    "\n"
    "declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)\n"
    "\n"
    "declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)\n"
    "\n"
    "define dso_local void @memory_ops(ptr %target, ptr %source) {\n"
    "entry:\n"
    "  call void @llvm.lifetime.start.p0(i64 16, ptr %target)\n"
    "  call void @llvm.memset.p0.i64(ptr %target, i8 0, i64 16, i1 0)\n"
    "  call void @llvm.memcpy.p0.p0.i64(ptr %target, ptr %source, "
    "i64 16, i1 0)\n"
    "  call void @llvm.lifetime.end.p0(i64 16, ptr %target)\n"
    "  ret void\n"
    "}\n";

static const char kCfgPhiText[] =
    "define i32 @select_i32(i1 %c, i32 %a, i32 %b) {\n"
    "entry:\n"
    "  br i1 %c, label %then, label %else\n"
    "then:\n"
    "  br label %join\n"
    "else:\n"
    "  br label %join\n"
    "join:\n"
    "  %x = phi i32 [ %a, %then ], [ %b, %else ]\n"
    "  ret i32 %x\n"
    "}\n";

static const char kScalarBinopText[] =
    "define i32 @binops(i32 %lhs, i32 %rhs, float %xf, float %yf) {\n"
    "entry:\n"
    "  %sum = add nuw nsw i32 %lhs, %rhs\n"
    "  %difference = sub i32 %sum, %rhs\n"
    "  %product = mul i32 %difference, %lhs\n"
    "  %unsigned_quotient = udiv i32 %product, %rhs\n"
    "  %signed_quotient = sdiv exact i32 %product, %rhs\n"
    "  %unsigned_remainder = urem i32 %unsigned_quotient, %rhs\n"
    "  %signed_remainder = srem i32 %signed_quotient, %rhs\n"
    "  %bits = and i32 %unsigned_remainder, %signed_remainder\n"
    "  %either = or i32 %bits, %lhs\n"
    "  %toggled = xor i32 %either, %rhs\n"
    "  %shifted = shl i32 %toggled, %rhs\n"
    "  %logical_shifted = lshr i32 %shifted, %rhs\n"
    "  %arithmetic_shifted = ashr i32 %logical_shifted, %rhs\n"
    "  %float_sum = fadd fast float %xf, %yf\n"
    "  %float_difference = fsub float %float_sum, %yf\n"
    "  %float_product = fmul float %float_difference, %xf\n"
    "  %float_quotient = fdiv float %float_product, %yf\n"
    "  %float_remainder = frem float %float_quotient, %yf\n"
    "  %float_negated = fneg nsz float %float_remainder\n"
    "  ret i32 %arithmetic_shifted\n"
    "}\n";

static const char kVectorElementsText[] =
    "define <4 x float> @vector_elements(<4 x float> %vector, float %value, "
    "i32 %index) {\n"
    "entry:\n"
    "  %lane = extractelement <4 x float> %vector, i32 %index\n"
    "  %sum = fadd float %lane, %value\n"
    "  %updated = insertelement <4 x float> %vector, float %sum, i32 %index\n"
    "  ret <4 x float> %updated\n"
    "}\n";

static const char kShuffleVectorText[] =
    "define <4 x float> @shuffle_vector(<4 x float> %lhs, <4 x float> %rhs) "
    "{\n"
    "entry:\n"
    "  %interleaved = shufflevector <4 x float> %lhs, <4 x float> %rhs, "
    "<4 x i32> <i32 0, i32 4, i32 1, i32 5>\n"
    "  ret <4 x float> %interleaved\n"
    "}\n";

static const char kCompareSelectText[] =
    "define i32 @compare_select(i32 %x, i32 %lower, i32 %upper, float "
    "%xf, float %yf) {\n"
    "entry:\n"
    "  %below = icmp slt i32 %x, %lower\n"
    "  %above = icmp sgt i32 %x, %upper\n"
    "  %ordered = fcmp olt float %xf, %yf\n"
    "  %at_least_lower = select i1 %below, i32 %lower, i32 %x\n"
    "  %clamped = select i1 %above, i32 %upper, i32 %at_least_lower\n"
    "  %result = select i1 %ordered, i32 %clamped, i32 %x\n"
    "  ret i32 %result\n"
    "}\n";

static const char kCastsText[] =
    "define void @casts(i64 %wide, i16 %narrow, float %scalar, double "
    "%wide_scalar, ptr %pointer) {\n"
    "entry:\n"
    "  %truncated = trunc i64 %wide to i32\n"
    "  %zero_extended = zext i16 %narrow to i64\n"
    "  %sign_extended = sext i16 %narrow to i64\n"
    "  %fp_truncated = fptrunc double %wide_scalar to float\n"
    "  %fp_extended = fpext float %scalar to double\n"
    "  %unsigned_to_fp = uitofp i16 %narrow to float\n"
    "  %signed_to_fp = sitofp i16 %narrow to float\n"
    "  %fp_to_unsigned = fptoui float %scalar to i32\n"
    "  %fp_to_signed = fptosi float %scalar to i32\n"
    "  %ptr_to_int = ptrtoint ptr %pointer to i64\n"
    "  %int_to_ptr = inttoptr i64 %wide to ptr\n"
    "  %bits = bitcast float %scalar to i32\n"
    "  %global_pointer = addrspacecast ptr %pointer to ptr addrspace(1)\n"
    "  ret void\n"
    "}\n";

static const char kInlineAsmText[] =
    "define i32 @inline_asm_add(i32 %lhs, i32 %rhs) {\n"
    "entry:\n"
    "  %sum = call i32 asm sideeffect \"addl $2, $0\", "
    "\"=r,0,r\"(i32 %lhs, i32 %rhs)\n"
    "  ret i32 %sum\n"
    "}\n";

static const char kAmdgpuIntrinsicsText[] =
    "source_filename = \"loom-amdgpu\"\n"
    "target triple = \"amdgcn-amd-amdhsa\"\n"
    "\n"
    "define amdgpu_kernel void @add_dispatch(ptr addrspace(1) inreg "
    "noalias noundef nonnull align 16 %x, ptr addrspace(1) inreg "
    "noalias noundef nonnull align 16 %y, ptr addrspace(1) inreg "
    "noalias noundef nonnull align 16 %z) #0 !reqd_work_group_size "
    "!0 {\n"
    "entry:\n"
    "  %tid = call range(i32 0, 64) i32 "
    "@llvm.amdgcn.workitem.id.x()\n"
    "  %x.rsrc = call ptr addrspace(7) "
    "@llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) %x, i16 0, "
    "i32 64, i32 159744)\n"
    "  %y.rsrc = call ptr addrspace(7) "
    "@llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) %y, i16 0, "
    "i32 64, i32 159744)\n"
    "  %z.rsrc = call ptr addrspace(7) "
    "@llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) %z, i16 0, "
    "i32 64, i32 159744)\n"
    "  %x.ptr = getelementptr float, ptr addrspace(7) %x.rsrc, i32 "
    "%tid\n"
    "  %y.ptr = getelementptr float, ptr addrspace(7) %y.rsrc, i32 "
    "%tid\n"
    "  %z.ptr = getelementptr float, ptr addrspace(7) %z.rsrc, i32 "
    "%tid\n"
    "  %x.value = load <1 x float>, ptr addrspace(7) %x.ptr, align "
    "4\n"
    "  %y.value = load <1 x float>, ptr addrspace(7) %y.ptr, align "
    "4\n"
    "  %z.value = fadd <1 x float> %x.value, %y.value\n"
    "  store <1 x float> %z.value, ptr addrspace(7) %z.ptr, align "
    "4\n"
    "  ret void\n"
    "}\n"
    "\n"
    "declare i32 @llvm.amdgcn.workitem.id.x()\n"
    "\n"
    "declare ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1("
    "ptr addrspace(1) readnone, i16, i32, i32)\n"
    "\n"
    "attributes #0 = { alwaysinline "
    "\"amdgpu-flat-work-group-size\"=\"64,64\" "
    "\"uniform-work-group-size\" }\n"
    "\n"
    "!0 = !{i32 64, i32 1, i32 1}\n";

iree_host_size_t loom_llvmir_test_module_scenario_count(void) {
  return LOOM_LLVMIR_TEST_MODULE_SCENARIO_COUNT;
}

iree_string_view_t loom_llvmir_test_module_scenario_name(
    loom_llvmir_test_module_scenario_t scenario) {
  switch (scenario) {
    case LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4:
      return IREE_SV("object_vadd4");
    case LOOM_LLVMIR_TEST_MODULE_CALL_CONSTANTS:
      return IREE_SV("call_constants");
    case LOOM_LLVMIR_TEST_MODULE_CFG_PHI:
      return IREE_SV("cfg_phi");
    case LOOM_LLVMIR_TEST_MODULE_INLINE_ASM:
      return IREE_SV("inline_asm");
    case LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS:
      return IREE_SV("amdgpu_intrinsics");
    case LOOM_LLVMIR_TEST_MODULE_SCALAR_BINOP:
      return IREE_SV("scalar_binop");
    case LOOM_LLVMIR_TEST_MODULE_VECTOR_ELEMENTS:
      return IREE_SV("vector_elements");
    case LOOM_LLVMIR_TEST_MODULE_SHUFFLE_VECTOR:
      return IREE_SV("shuffle_vector");
    case LOOM_LLVMIR_TEST_MODULE_BUILTIN_INTRINSICS:
      return IREE_SV("builtin_intrinsics");
    case LOOM_LLVMIR_TEST_MODULE_COMPARE_SELECT:
      return IREE_SV("compare_select");
    case LOOM_LLVMIR_TEST_MODULE_CASTS:
      return IREE_SV("casts");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_llvmir_test_module_target_config(
    loom_llvmir_test_module_scenario_t scenario,
    loom_llvmir_target_config_t* out_target_config,
    const loom_llvmir_target_config_t** out_target_config_ptr) {
  *out_target_config_ptr = NULL;
  switch (scenario) {
    case LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4: {
      const loom_llvmir_target_env_t* target_env =
          loom_llvmir_target_env_x86_64_unknown_linux_gnu();
      IREE_RETURN_IF_ERROR(loom_llvmir_target_env_module_config(
          target_env, IREE_SV("loom-object"), out_target_config));
      *out_target_config_ptr = out_target_config;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_TEST_MODULE_CALL_CONSTANTS: {
      const loom_llvmir_target_env_t* target_env =
          loom_llvmir_target_env_x86_64_unknown_linux_gnu();
      IREE_RETURN_IF_ERROR(loom_llvmir_target_env_module_config(
          target_env, IREE_SV("loom-call-constants"), out_target_config));
      *out_target_config_ptr = out_target_config;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_TEST_MODULE_BUILTIN_INTRINSICS: {
      const loom_llvmir_target_env_t* target_env =
          loom_llvmir_target_env_x86_64_unknown_linux_gnu();
      IREE_RETURN_IF_ERROR(loom_llvmir_target_env_module_config(
          target_env, IREE_SV("loom-builtins"), out_target_config));
      *out_target_config_ptr = out_target_config;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS: {
      const loom_llvmir_target_env_t* target_env =
          loom_llvmir_target_env_amdgcn_amd_amdhsa();
      IREE_RETURN_IF_ERROR(loom_llvmir_target_env_module_config(
          target_env, IREE_SV("loom-amdgpu"), out_target_config));
      *out_target_config_ptr = out_target_config;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_TEST_MODULE_CFG_PHI:
    case LOOM_LLVMIR_TEST_MODULE_INLINE_ASM:
    case LOOM_LLVMIR_TEST_MODULE_SCALAR_BINOP:
    case LOOM_LLVMIR_TEST_MODULE_VECTOR_ELEMENTS:
    case LOOM_LLVMIR_TEST_MODULE_SHUFFLE_VECTOR:
    case LOOM_LLVMIR_TEST_MODULE_COMPARE_SELECT:
    case LOOM_LLVMIR_TEST_MODULE_CASTS:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM IR test module scenario");
  }
}

static iree_status_t loom_llvmir_test_module_populate(
    loom_llvmir_test_module_scenario_t scenario, loom_llvmir_module_t* module) {
  switch (scenario) {
    case LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4:
      return loom_llvmir_test_populate_object_vadd4(module);
    case LOOM_LLVMIR_TEST_MODULE_CALL_CONSTANTS:
      return loom_llvmir_test_populate_call_constants(module);
    case LOOM_LLVMIR_TEST_MODULE_BUILTIN_INTRINSICS:
      return loom_llvmir_test_populate_builtin_intrinsics(module);
    case LOOM_LLVMIR_TEST_MODULE_CFG_PHI:
      return loom_llvmir_test_populate_cfg_phi(module);
    case LOOM_LLVMIR_TEST_MODULE_INLINE_ASM:
      return loom_llvmir_test_populate_inline_asm(module);
    case LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS:
      return loom_llvmir_test_populate_amdgpu_intrinsics(module);
    case LOOM_LLVMIR_TEST_MODULE_SCALAR_BINOP:
      return loom_llvmir_test_populate_scalar_binop(module);
    case LOOM_LLVMIR_TEST_MODULE_VECTOR_ELEMENTS:
      return loom_llvmir_test_populate_vector_elements(module);
    case LOOM_LLVMIR_TEST_MODULE_SHUFFLE_VECTOR:
      return loom_llvmir_test_populate_shuffle_vector(module);
    case LOOM_LLVMIR_TEST_MODULE_COMPARE_SELECT:
      return loom_llvmir_test_populate_compare_select(module);
    case LOOM_LLVMIR_TEST_MODULE_CASTS:
      return loom_llvmir_test_populate_casts(module);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM IR test module scenario");
  }
}

iree_status_t loom_llvmir_test_module_build(
    loom_llvmir_test_module_scenario_t scenario, iree_allocator_t allocator,
    loom_llvmir_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  loom_llvmir_target_config_t target_config = {0};
  const loom_llvmir_target_config_t* target_config_ptr = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_test_module_target_config(
      scenario, &target_config, &target_config_ptr));

  loom_llvmir_module_t* module = NULL;
  iree_status_t status =
      loom_llvmir_module_allocate(target_config_ptr, allocator, &module);
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_test_module_populate(scenario, module);
  }
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    loom_llvmir_module_free(module);
  }
  return status;
}

iree_string_view_t loom_llvmir_test_module_expected_text(
    loom_llvmir_test_module_scenario_t scenario) {
  switch (scenario) {
    case LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4:
      return iree_make_string_view(kObjectVadd4Text,
                                   IREE_ARRAYSIZE(kObjectVadd4Text) - 1);
    case LOOM_LLVMIR_TEST_MODULE_CALL_CONSTANTS:
      return iree_make_string_view(kCallConstantsText,
                                   IREE_ARRAYSIZE(kCallConstantsText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_BUILTIN_INTRINSICS:
      return iree_make_string_view(kBuiltinIntrinsicsText,
                                   IREE_ARRAYSIZE(kBuiltinIntrinsicsText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_CFG_PHI:
      return iree_make_string_view(kCfgPhiText,
                                   IREE_ARRAYSIZE(kCfgPhiText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_INLINE_ASM:
      return iree_make_string_view(kInlineAsmText,
                                   IREE_ARRAYSIZE(kInlineAsmText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS:
      return iree_make_string_view(kAmdgpuIntrinsicsText,
                                   IREE_ARRAYSIZE(kAmdgpuIntrinsicsText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_SCALAR_BINOP:
      return iree_make_string_view(kScalarBinopText,
                                   IREE_ARRAYSIZE(kScalarBinopText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_VECTOR_ELEMENTS:
      return iree_make_string_view(kVectorElementsText,
                                   IREE_ARRAYSIZE(kVectorElementsText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_SHUFFLE_VECTOR:
      return iree_make_string_view(kShuffleVectorText,
                                   IREE_ARRAYSIZE(kShuffleVectorText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_COMPARE_SELECT:
      return iree_make_string_view(kCompareSelectText,
                                   IREE_ARRAYSIZE(kCompareSelectText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_CASTS:
      return iree_make_string_view(kCastsText, IREE_ARRAYSIZE(kCastsText) - 1);
    default:
      return iree_string_view_empty();
  }
}
