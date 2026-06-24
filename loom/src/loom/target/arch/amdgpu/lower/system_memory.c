// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/system_memory.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_system_memory_cache_policy_encoding(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  IREE_ASSERT(descriptor_set_info != NULL);
  return descriptor_set_info->vector_memory.cache_policy_encoding;
}

iree_status_t loom_amdgpu_system_memory_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_system_memory_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  return loom_amdgpu_system_memory_build_u32_attr(builder, IREE_SV("offset"),
                                                  byte_offset, out_attr);
}

static bool loom_amdgpu_system_memory_type_is_register_class(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    uint16_t reg_class_id) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             descriptor_set->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

static void loom_amdgpu_system_memory_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint16_t reg_class_id, uint32_t unit_count) {
  IREE_ASSERT_LT(value, builder->module->values.count);
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT(loom_amdgpu_system_memory_type_is_register_class(
      descriptor_set, type, reg_class_id));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(type), unit_count);
}

static iree_status_t loom_amdgpu_system_memory_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_vgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  return loom_amdgpu_system_memory_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
      vgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_system_memory_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_system_memory_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_system_memory_build_sgpr_u32_binary(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, lhs, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, rhs, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &sgpr_type,
      /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_descriptor_implicit_resource_type(
      descriptor_set, consumer_descriptor, &m0_type));
  return loom_amdgpu_system_memory_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
}

iree_status_t loom_amdgpu_system_memory_build_saddr_byte_offset(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_location_id_t location, loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, base_address, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (byte_offset == 0) {
    *out_address = base_address;
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_value_id_t base_lo = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, base_address, /*offset=*/0,
                                            sgpr_type, location, &slice_lo_op));
  base_lo = loom_low_slice_result(slice_lo_op);
  loom_value_id_t base_hi = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, base_address, /*offset=*/1,
                                            sgpr_type, location, &slice_hi_op));
  base_hi = loom_low_slice_result(slice_hi_op);

  loom_value_id_t offset_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_const(
      builder, descriptor_set, byte_offset, location, &offset_lo));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_const(
      builder, descriptor_set, 0, location, &zero));

  loom_value_id_t sum_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_binary(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, base_lo,
      offset_lo, location, &sum_lo));
  loom_value_id_t sum_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_binary(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, base_hi,
      zero, location, &sum_hi));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  const loom_value_id_t parts[] = {sum_lo, sum_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), sgpr_x2_type,
                            location, &concat_op));
  *out_address = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_global_memory_descriptor(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id, const loom_low_asm_form_t** out_asm_form) {
  *out_descriptor = NULL;
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  *out_asm_form = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  IREE_ASSERT_LT(descriptor->canonical_asm_form_ordinal,
                 descriptor_set->asm_form_count);
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  IREE_ASSERT(asm_form->operand_index_count == 2 ||
              asm_form->operand_index_count == 3);
  *out_descriptor = descriptor;
  *out_opcode_id = opcode_id;
  *out_asm_form = asm_form;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_global_load_saddr(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t register_count,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, base_address, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));

  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  const loom_low_asm_form_t* asm_form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_global_memory_descriptor(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id,
      &asm_form));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, register_count,
      &result_type));
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, byte_offset, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_load_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[3] = {zero_vaddr, base_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  if (asm_form->operand_index_count == 3) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_m0_const_u32(
        builder, descriptor_set, descriptor, 0, location,
        &operands[operand_count++]));
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t value =
      loom_value_slice_get(loom_low_op_results(op), 0);
  if (iree_any_bit_set(flags, LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering(
        builder, descriptor_set, location));
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_readfirstlane_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, source, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &result_type));
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      &descriptor, &opcode_id));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, &source,
      /*operand_count=*/1, loom_make_named_attr_slice(NULL, 0), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_append_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  IREE_ASSERT_LT(*inout_attr_count, attr_capacity);
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_system_memory_build_u32_attr(builder, name, value, &attr));
  attrs[(*inout_attr_count)++] = attr;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_append_scope_attr(
    loom_builder_t* builder, loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_u32_attr(
      builder, IREE_SV("scope"), (uint32_t)scope, attrs, attr_capacity,
      inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_append_sc0_attr(
    loom_builder_t* builder, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_u32_attr(
      builder, IREE_SV("sc0"), 1, attrs, attr_capacity, inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_append_sc1_attr(
    loom_builder_t* builder, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_u32_attr(
      builder, IREE_SV("sc1"), 1, attrs, attr_capacity, inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_append_sc0_sc1_attrs(
    loom_builder_t* builder, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_sc0_attr(
      builder, attrs, attr_capacity, inout_attr_count));
  return loom_amdgpu_system_memory_append_sc1_attr(
      builder, attrs, attr_capacity, inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_load_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_load_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_load_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_u32_attr(
          builder, IREE_SV("glc"), 1, attrs, attr_capacity, inout_attr_count));
      return loom_amdgpu_system_memory_append_u32_attr(
          builder, IREE_SV("dlc"), 1, attrs, attr_capacity, inout_attr_count);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc0_sc1_attrs(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, scope, attrs, attr_capacity, inout_attr_count);
    default:
      IREE_ASSERT_UNREACHABLE("validated AMDGPU system-memory load policy");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_system_memory_append_release_store_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_release_store_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_release_store_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc0_sc1_attrs(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, scope, attrs, attr_capacity, inout_attr_count);
    default:
      IREE_ASSERT_UNREACHABLE(
          "validated AMDGPU system-memory release-store policy");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_no_return_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc1_attr(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, scope, attrs, attr_capacity, inout_attr_count);
    default:
      IREE_ASSERT_UNREACHABLE(
          "validated AMDGPU system-memory no-return atomic policy");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_return_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc0_sc1_attrs(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, scope, attrs, attr_capacity, inout_attr_count);
    default:
      IREE_ASSERT_UNREACHABLE(
          "validated AMDGPU system-memory returning atomic policy");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_system_memory_build_resolved_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_string_id_t opcode_id,
    loom_named_attr_slice_t attrs, loom_location_id_t location) {
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, /*operands=*/NULL,
      /*operand_count=*/0, attrs, /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &op);
}

static iree_status_t loom_amdgpu_system_memory_build_descriptor_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, key, &opcode_id));
  return loom_amdgpu_system_memory_build_resolved_packet(
      builder, descriptor_set, descriptor, opcode_id, attrs, location);
}

static iree_status_t loom_amdgpu_system_memory_build_explicit_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  return loom_amdgpu_system_memory_build_resolved_packet(
      builder, descriptor_set, descriptor, opcode_id, attrs, location);
}

static iree_status_t loom_amdgpu_system_memory_build_wait_counter_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t counter_mask, uint16_t target_count, loom_location_id_t location) {
  loom_amdgpu_wait_packet_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_select_counter_mask(
      descriptor_set, counter_mask, target_count, &selection));
  loom_named_attr_t
      attrs[LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < selection.immediate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
        builder, selection.immediates[i].name, selection.immediates[i].value,
        &attrs[i]));
  }
  return loom_amdgpu_system_memory_build_descriptor_packet(
      builder, descriptor_set, selection.descriptor,
      loom_make_named_attr_slice(attrs, selection.immediate_count), location);
}

static iree_status_t loom_amdgpu_system_memory_build_scoped_buffer_wbl2(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_sc0_sc1_attrs(
      builder, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2,
      loom_make_named_attr_slice(attrs, attr_count), location);
}

static iree_status_t loom_amdgpu_system_memory_build_scoped_buffer_inv(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_sc0_sc1_attrs(
      builder, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV,
      loom_make_named_attr_slice(attrs, attr_count), location);
}

static iree_status_t
loom_amdgpu_system_memory_build_scoped_global_cache_control(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_cache_scope_t scope,
    loom_location_id_t location) {
  loom_named_attr_t attr = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_scope_attr(
      builder, scope, &attr, 1, &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, descriptor_ref,
      loom_make_named_attr_slice(&attr, attr_count), location);
}

iree_status_t loom_amdgpu_system_memory_build_release_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  return loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location);
}

iree_status_t loom_amdgpu_system_memory_build_release_ordering_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_location_id_t location) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*target_count=*/0, location));
      return loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
          /*target_count=*/0, location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_build_scoped_buffer_wbl2(
          builder, descriptor_set, location);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_system_memory_build_scoped_global_cache_control(
              builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB,
              scope, location));
      return loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
          /*target_count=*/0, location);
    }
    default:
      IREE_ASSERT_UNREACHABLE(
          "validated AMDGPU system-memory release ordering");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_system_memory_build_load_wait(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*target_count=*/0, location);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*target_count=*/0, location);
    default:
      IREE_ASSERT_UNREACHABLE("validated AMDGPU system-memory load wait");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_system_memory_build_acquire_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  return loom_amdgpu_system_memory_build_acquire_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location);
}

iree_status_t loom_amdgpu_system_memory_build_acquire_ordering_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_location_id_t location) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*target_count=*/0, location));
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV,
          loom_named_attr_slice_empty(), location));
      return loom_amdgpu_system_memory_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV,
          loom_named_attr_slice_empty(), location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*target_count=*/0, location));
      return loom_amdgpu_system_memory_build_scoped_buffer_inv(
          builder, descriptor_set, location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
          /*target_count=*/0, location));
      return loom_amdgpu_system_memory_build_scoped_global_cache_control(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV, scope,
          location);
    }
    default:
      IREE_ASSERT_UNREACHABLE(
          "validated AMDGPU system-memory acquire ordering");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_system_memory_build_uniform_load_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vector_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_global_load_saddr(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
      /*register_count=*/1, base_address, byte_offset, flags, location,
      &vector_value));
  return loom_amdgpu_system_memory_build_readfirstlane_b32(
      builder, descriptor_set, vector_value, location, out_value);
}

iree_status_t loom_amdgpu_system_memory_build_uniform_load_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vector_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_global_load_saddr(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
      /*register_count=*/2, base_address, byte_offset, flags, location,
      &vector_value));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_value_id_t vector_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(vector_lanes); ++i) {
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, vector_value, i,
                                              vgpr_type, location, &slice_op));
    vector_lanes[i] = loom_low_slice_result(slice_op);
  }

  loom_value_id_t scalar_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(scalar_lanes); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_readfirstlane_b32(
        builder, descriptor_set, vector_lanes[i], location, &scalar_lanes[i]));
  }

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, scalar_lanes, IREE_ARRAYSIZE(scalar_lanes),
                            sgpr_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}
