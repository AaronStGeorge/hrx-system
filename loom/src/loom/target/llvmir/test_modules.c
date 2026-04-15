// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/test_modules.h"

#include "loom/target/llvmir/llvmir.h"

#define LOOM_LLVMIR_TEST_MODULE_SCENARIO_COUNT 4

#define LOOM_LLVMIR_TEST_GOTO_IF_ERROR(expr) \
  do {                                       \
    status = (expr);                         \
    if (!iree_status_is_ok(status)) {        \
      goto cleanup;                          \
    }                                        \
  } while (0)

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

static iree_status_t loom_llvmir_test_finish_module(
    iree_status_t status, loom_llvmir_module_t* module,
    loom_llvmir_module_t** out_module) {
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    loom_llvmir_module_free(module);
  }
  return status;
}

static iree_status_t loom_llvmir_test_build_object_vadd4(
    iree_allocator_t allocator, loom_llvmir_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  loom_llvmir_target_config_t target_config = {
      .source_name = IREE_SV("loom-object"),
      .target_triple = IREE_SV("x86_64-unknown-linux-gnu"),
      .default_pointer_bitwidth = 64,
      .index_bitwidth = 64,
      .offset_bitwidth = 64,
  };
  loom_llvmir_module_t* module = NULL;
  iree_status_t status =
      loom_llvmir_module_allocate(&target_config, allocator, &module);
  if (!iree_status_is_ok(status)) goto cleanup;

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v4f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_void_type(module, &void_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_pointer_type(module, 0, &ptr_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_vector_type(module, 4, f32_type, &v4f32_type));

  loom_llvmir_function_t* function = NULL;
  loom_llvmir_function_desc_t function_desc = {
      .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
      .name = IREE_SV("vadd4_object"),
      .return_type = void_type,
      .linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
      .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
  };
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = ptr_type,
          .name = IREE_SV("x"),
          .attrs = readonly_attrs,
          .attr_count = IREE_ARRAYSIZE(readonly_attrs),
      },
      &x));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = ptr_type,
          .name = IREE_SV("y"),
          .attrs = readonly_attrs,
          .attr_count = IREE_ARRAYSIZE(readonly_attrs),
      },
      &y));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = ptr_type,
          .name = IREE_SV("z"),
          .attrs = writeonly_attrs,
          .attr_count = IREE_ARRAYSIZE(writeonly_attrs),
      },
      &z));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_parameter(
      function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = i64_type,
          .name = IREE_SV("base"),
          .attrs = noundef_attrs,
          .attr_count = IREE_ARRAYSIZE(noundef_attrs),
      },
      &base));

  loom_llvmir_block_t* entry = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t x_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t y_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t z_ptr = LOOM_LLVMIR_VALUE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_load(entry,
                             &(loom_llvmir_load_desc_t){
                                 .result_name = IREE_SV("x.v"),
                                 .result_type = v4f32_type,
                                 .pointer = x_ptr,
                                 .alignment = 4,
                             },
                             &x_value));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_load(entry,
                             &(loom_llvmir_load_desc_t){
                                 .result_name = IREE_SV("y.v"),
                                 .result_type = v4f32_type,
                                 .pointer = y_ptr,
                                 .alignment = 4,
                             },
                             &y_value));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("z.v"),
                                  .result_type = v4f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FADD,
                                  .lhs = x_value,
                                  .rhs = y_value,
                              },
                              &sum));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_store(entry, &(loom_llvmir_store_desc_t){
                                         .value = sum,
                                         .pointer = z_ptr,
                                         .alignment = 4,
                                     }));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_ret_void(entry));

cleanup:
  return loom_llvmir_test_finish_module(status, module, out_module);
}

static iree_status_t loom_llvmir_test_build_cfg_phi(
    iree_allocator_t allocator, loom_llvmir_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  loom_llvmir_module_t* module = NULL;
  iree_status_t status = loom_llvmir_module_allocate(NULL, allocator, &module);
  if (!iree_status_is_ok(status)) goto cleanup;

  loom_llvmir_type_id_t i1_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 1, &i1_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));

  loom_llvmir_function_t* function = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_add_function(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i1_type,
                                             .name = IREE_SV("c"),
                                         },
                                         &condition));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("a"),
                                         },
                                         &lhs));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("then"), &then_block));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("else"), &else_block));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("join"), &join_block));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_cond_br(
      entry, condition, loom_llvmir_block_id(then_block),
      loom_llvmir_block_id(else_block)));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_br(then_block, loom_llvmir_block_id(join_block)));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_br(else_block, loom_llvmir_block_id(join_block)));
  loom_llvmir_phi_incoming_t incoming[] = {
      {.value = lhs, .predecessor = loom_llvmir_block_id(then_block)},
      {.value = rhs, .predecessor = loom_llvmir_block_id(else_block)},
  };
  loom_llvmir_value_id_t selected = LOOM_LLVMIR_VALUE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_phi(join_block,
                            &(loom_llvmir_phi_desc_t){
                                .result_name = IREE_SV("x"),
                                .result_type = i32_type,
                                .incoming = incoming,
                                .incoming_count = IREE_ARRAYSIZE(incoming),
                            },
                            &selected));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_ret(join_block, selected));

cleanup:
  return loom_llvmir_test_finish_module(status, module, out_module);
}

static iree_status_t loom_llvmir_test_build_inline_asm(
    iree_allocator_t allocator, loom_llvmir_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  loom_llvmir_module_t* module = NULL;
  iree_status_t status = loom_llvmir_module_allocate(NULL, allocator, &module);
  if (!iree_status_is_ok(status)) goto cleanup;

  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));

  loom_llvmir_function_t* function = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_add_function(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("lhs"),
                                         },
                                         &lhs));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_parameter(function,
                                         &(loom_llvmir_parameter_desc_t){
                                             .type_id = i32_type,
                                             .name = IREE_SV("rhs"),
                                         },
                                         &rhs));

  loom_llvmir_block_t* entry = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  loom_llvmir_value_id_t args[] = {lhs, rhs};
  loom_llvmir_value_id_t sum = LOOM_LLVMIR_VALUE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_inline_asm(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_ret(entry, sum));

cleanup:
  return loom_llvmir_test_finish_module(status, module, out_module);
}

static iree_status_t loom_llvmir_test_build_amdgpu_intrinsics(
    iree_allocator_t allocator, loom_llvmir_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  loom_llvmir_target_config_t target_config = {
      .source_name = IREE_SV("loom-amdgpu"),
      .target_triple = IREE_SV("amdgcn-amd-amdhsa"),
      .default_pointer_bitwidth = 64,
      .index_bitwidth = 32,
      .offset_bitwidth = 64,
  };
  loom_llvmir_module_t* module = NULL;
  iree_status_t status =
      loom_llvmir_module_allocate(&target_config, allocator, &module);
  if (!iree_status_is_ok(status)) goto cleanup;

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i16_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v1f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t global_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t fat_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_void_type(module, &void_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 16, &i16_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_vector_type(module, 1, f32_type, &v1f32_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_pointer_type(module, 1, &global_ptr_type));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_get_pointer_type(module, 7, &fat_ptr_type));

  loom_llvmir_attr_t kernel_attrs[] = {
      loom_llvmir_test_attr(LOOM_LLVMIR_ATTR_ALWAYSINLINE),
      loom_llvmir_test_string_key_value_attr("amdgpu-flat-work-group-size",
                                             "64,64"),
      loom_llvmir_test_string_key_attr("uniform-work-group-size"),
  };
  loom_llvmir_attr_group_id_t kernel_attr_group =
      LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_add_attr_group(
      module, kernel_attrs, IREE_ARRAYSIZE(kernel_attrs), &kernel_attr_group));
  int32_t workgroup_size_values[] = {64, 1, 1};
  loom_llvmir_metadata_id_t workgroup_size = LOOM_LLVMIR_METADATA_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_add_metadata_i32_tuple(
      module,
      &(loom_llvmir_metadata_i32_tuple_t){
          .values = workgroup_size_values,
          .value_count = IREE_ARRAYSIZE(workgroup_size_values),
      },
      &workgroup_size));

  loom_llvmir_function_t* kernel = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = IREE_SV("add_dispatch"),
          .return_type = void_type,
          .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL,
          .attr_group_id = kernel_attr_group,
      },
      &kernel));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_metadata_attachment(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_parameter(
      kernel,
      &(loom_llvmir_parameter_desc_t){
          .type_id = global_ptr_type,
          .name = IREE_SV("x"),
          .attrs = binding_attrs,
          .attr_count = IREE_ARRAYSIZE(binding_attrs),
      },
      &x));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_parameter(
      kernel,
      &(loom_llvmir_parameter_desc_t){
          .type_id = global_ptr_type,
          .name = IREE_SV("y"),
          .attrs = binding_attrs,
          .attr_count = IREE_ARRAYSIZE(binding_attrs),
      },
      &y));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_function_add_parameter(
      kernel,
      &(loom_llvmir_parameter_desc_t){
          .type_id = global_ptr_type,
          .name = IREE_SV("z"),
          .attrs = binding_attrs,
          .attr_count = IREE_ARRAYSIZE(binding_attrs),
      },
      &z));

  loom_llvmir_function_t* workitem_id = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_declare_amdgcn_workitem_id_x(module, &workitem_id));
  loom_llvmir_function_t* make_buffer_resource = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_declare_amdgcn_make_buffer_rsrc(
      module, 7, 1, &make_buffer_resource));

  loom_llvmir_value_id_t stride_zero = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t records = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t flags = LOOM_LLVMIR_VALUE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_add_integer_constant(
      module, i16_type, 0, &stride_zero));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_module_add_integer_constant(module, i64_type, 64, &records));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_module_add_integer_constant(
      module, i32_type, 159744, &flags));

  loom_llvmir_block_t* entry = NULL;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_function_add_block(kernel, IREE_SV("entry"), &entry));
  loom_llvmir_attr_t tid_range[] = {
      loom_llvmir_test_range_attr(i32_type, 0, 64),
  };
  loom_llvmir_value_id_t tid = LOOM_LLVMIR_VALUE_ID_INVALID;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .result_name = IREE_SV("x.rsrc"),
          .callee = loom_llvmir_function_id(make_buffer_resource),
          .args = resource_args,
          .arg_count = IREE_ARRAYSIZE(resource_args),
      },
      &x_resource));
  resource_args[0] = y;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_call(
      entry,
      &(loom_llvmir_call_desc_t){
          .result_name = IREE_SV("y.rsrc"),
          .callee = loom_llvmir_function_id(make_buffer_resource),
          .args = resource_args,
          .arg_count = IREE_ARRAYSIZE(resource_args),
      },
      &y_resource));
  resource_args[0] = z;
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_call(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
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
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_load(entry,
                             &(loom_llvmir_load_desc_t){
                                 .result_name = IREE_SV("x.value"),
                                 .result_type = v1f32_type,
                                 .pointer = x_ptr,
                                 .alignment = 4,
                             },
                             &x_value));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_load(entry,
                             &(loom_llvmir_load_desc_t){
                                 .result_name = IREE_SV("y.value"),
                                 .result_type = v1f32_type,
                                 .pointer = y_ptr,
                                 .alignment = 4,
                             },
                             &y_value));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_binop(entry,
                              &(loom_llvmir_binop_desc_t){
                                  .result_name = IREE_SV("z.value"),
                                  .result_type = v1f32_type,
                                  .op = LOOM_LLVMIR_BINOP_FADD,
                                  .lhs = x_value,
                                  .rhs = y_value,
                              },
                              &z_value));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(
      loom_llvmir_build_store(entry, &(loom_llvmir_store_desc_t){
                                         .value = z_value,
                                         .pointer = z_ptr,
                                         .alignment = 4,
                                     }));
  LOOM_LLVMIR_TEST_GOTO_IF_ERROR(loom_llvmir_build_ret_void(entry));

cleanup:
  return loom_llvmir_test_finish_module(status, module, out_module);
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
    "i64 64, i32 159744)\n"
    "  %y.rsrc = call ptr addrspace(7) "
    "@llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) %y, i16 0, "
    "i64 64, i32 159744)\n"
    "  %z.rsrc = call ptr addrspace(7) "
    "@llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) %z, i16 0, "
    "i64 64, i32 159744)\n"
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
    "ptr addrspace(1) readnone, i16, i64, i32)\n"
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
    case LOOM_LLVMIR_TEST_MODULE_CFG_PHI:
      return IREE_SV("cfg_phi");
    case LOOM_LLVMIR_TEST_MODULE_INLINE_ASM:
      return IREE_SV("inline_asm");
    case LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS:
      return IREE_SV("amdgpu_intrinsics");
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_llvmir_test_module_build(
    loom_llvmir_test_module_scenario_t scenario, iree_allocator_t allocator,
    loom_llvmir_module_t** out_module) {
  switch (scenario) {
    case LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4:
      return loom_llvmir_test_build_object_vadd4(allocator, out_module);
    case LOOM_LLVMIR_TEST_MODULE_CFG_PHI:
      return loom_llvmir_test_build_cfg_phi(allocator, out_module);
    case LOOM_LLVMIR_TEST_MODULE_INLINE_ASM:
      return loom_llvmir_test_build_inline_asm(allocator, out_module);
    case LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS:
      return loom_llvmir_test_build_amdgpu_intrinsics(allocator, out_module);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM IR test module scenario");
  }
}

iree_string_view_t loom_llvmir_test_module_expected_text(
    loom_llvmir_test_module_scenario_t scenario) {
  switch (scenario) {
    case LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4:
      return iree_make_string_view(kObjectVadd4Text,
                                   IREE_ARRAYSIZE(kObjectVadd4Text) - 1);
    case LOOM_LLVMIR_TEST_MODULE_CFG_PHI:
      return iree_make_string_view(kCfgPhiText,
                                   IREE_ARRAYSIZE(kCfgPhiText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_INLINE_ASM:
      return iree_make_string_view(kInlineAsmText,
                                   IREE_ARRAYSIZE(kInlineAsmText) - 1);
    case LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS:
      return iree_make_string_view(kAmdgpuIntrinsicsText,
                                   IREE_ARRAYSIZE(kAmdgpuIntrinsicsText) - 1);
    default:
      return iree_string_view_empty();
  }
}

#undef LOOM_LLVMIR_TEST_GOTO_IF_ERROR
