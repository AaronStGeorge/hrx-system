// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_storage.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors.h"
#include "loom/target/arch/spirv/isa.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/registers.h"

struct loom_spirv_module_workgroup_storage_entry_t {
  // True after a low.storage.reserve defines this storage value.
  bool reserved;
  // Reserved byte length from low.storage.reserve.
  int64_t byte_length;
  // Reserved byte alignment from low.storage.reserve.
  int64_t byte_alignment;
  // Scalar type selected by the first typed low.storage.address use.
  loom_spirv_scalar_type_t scalar_type;
  // OpVariable result ID, or zero before materialization.
  uint32_t variable_id;
  // OpTypeArray ID for the typed Workgroup storage object.
  uint32_t array_type_id;
  // OpTypePointer Workgroup array type ID for variable_id.
  uint32_t array_pointer_type_id;
};

iree_status_t loom_spirv_module_workgroup_storage_initialize(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_workgroup_storage_state_t* out_state,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT_ARGUMENT(out_state);
  IREE_ASSERT_ARGUMENT(scratch_arena);

  *out_state = (loom_spirv_module_workgroup_storage_state_t){
      .entry_count = value_domain->value_count,
  };
  if (value_domain->value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, value_domain->value_count, sizeof(*out_state->entries),
      (void**)&out_state->entries));
  memset(out_state->entries, 0,
         value_domain->value_count * sizeof(*out_state->entries));
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_workgroup_storage_lookup_entry(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_workgroup_storage_state_t* state,
    loom_value_id_t value_id,
    loom_spirv_module_workgroup_storage_entry_t** out_entry) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= state->entry_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage value %" PRIu32
                            " is outside the emitted function domain",
                            value_id);
  }
  *out_entry = &state->entries[value_ordinal];
  return iree_ok_status();
}

iree_status_t loom_spirv_module_workgroup_storage_emit_reserve(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_workgroup_storage_state_t* state, const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(op);

  loom_spirv_module_workgroup_storage_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_lookup_entry(
      value_domain, state, loom_low_storage_reserve_storage(op), &entry));
  if (entry->reserved) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage reserve was already "
                            "materialized");
  }
  entry->reserved = true;
  entry->byte_length = loom_low_storage_reserve_byte_length(op);
  entry->byte_alignment = loom_low_storage_reserve_byte_alignment(op);
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_workgroup_storage_element_byte_count(
    loom_spirv_scalar_type_t scalar_type, uint32_t* out_element_byte_count) {
  *out_element_byte_count = 0;
  const loom_spirv_scalar_type_descriptor_t* scalar_descriptor =
      loom_spirv_scalar_type_descriptor(scalar_type);
  if (scalar_descriptor == NULL || scalar_descriptor->bit_width == 0 ||
      (scalar_descriptor->bit_width % 8) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage address selected an "
                            "unsupported scalar type");
  }
  *out_element_byte_count = scalar_descriptor->bit_width / 8;
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_workgroup_storage_emit_type(
    const loom_spirv_module_workgroup_storage_entry_t* entry,
    loom_spirv_type_context_t* type_context, uint32_t* out_array_type_id,
    uint32_t* out_array_pointer_type_id) {
  uint32_t element_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_element_byte_count(
      entry->scalar_type, &element_byte_count));
  if (entry->byte_length <= 0 ||
      entry->byte_length % (int64_t)element_byte_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage byte length does not "
                            "match the selected scalar element type");
  }
  if (entry->byte_alignment < (int64_t)element_byte_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage byte alignment does not "
                            "cover the selected scalar element type");
  }
  const uint64_t element_count =
      (uint64_t)entry->byte_length / element_byte_count;
  if (element_count == 0 || element_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SPIR-V Workgroup storage element count exceeds "
                            "32-bit OpTypeArray length");
  }

  uint32_t element_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_scalar(
      type_context, entry->scalar_type, &element_type_id));
  uint32_t length_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      type_context, (uint32_t)element_count, &length_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_array(type_context, element_type_id,
                                                  length_id, element_byte_count,
                                                  out_array_type_id));
  return loom_spirv_emit_type_pointer(
      type_context, LOOM_SPIRV_STORAGE_CLASS_WORKGROUP, *out_array_type_id,
      /*pointer_array_stride=*/0, out_array_pointer_type_id);
}

static iree_status_t loom_spirv_module_workgroup_storage_emit_variable(
    loom_spirv_module_workgroup_storage_entry_t* entry,
    loom_spirv_type_context_t* type_context,
    loom_spirv_module_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_emit_type(
      entry, type_context, &entry->array_type_id,
      &entry->array_pointer_type_id));
  entry->variable_id = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t operands[] = {
      entry->array_pointer_type_id,
      entry->variable_id,
      LOOM_SPIRV_STORAGE_CLASS_WORKGROUP,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(builder,
                                        LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, operands, IREE_ARRAYSIZE(operands)));
  return iree_ok_status();
}

iree_status_t loom_spirv_module_workgroup_storage_emit_address(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_workgroup_storage_state_t* state, const loom_op_t* op,
    loom_spirv_type_context_t* type_context,
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_value_ref_t* out_value_ref) {
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(type_context);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_value_ref);

  *out_value_ref = (loom_spirv_module_value_ref_t){0};
  if (loom_low_storage_address_offset(op) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage addresses must use a "
                            "zero byte offset");
  }
  loom_spirv_module_workgroup_storage_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_lookup_entry(
      value_domain, state, loom_low_storage_address_storage(op), &entry));
  if (!entry->reserved) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage address references "
                            "storage without a reserve");
  }

  loom_spirv_value_type_t value_type = {0};
  const loom_type_t result_type = loom_module_value_type(
      value_domain->module, loom_low_storage_address_result(op));
  if (!loom_low_type_is_register(result_type) ||
      loom_low_register_type_descriptor_set_stable_id(result_type) !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID ||
      !loom_spirv_value_type_from_reg_class_id(
          loom_low_register_type_class_id(result_type), &value_type) ||
      value_type.value_class != LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage address result must be "
                            "a typed Workgroup array pointer register");
  }
  if (entry->variable_id != 0 && entry->scalar_type != value_type.scalar_type) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V Workgroup storage is addressed with "
                            "multiple scalar element types");
  }
  entry->scalar_type = value_type.scalar_type;
  if (entry->variable_id == 0) {
    IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_emit_variable(
        entry, type_context, builder));
  }
  *out_value_ref = (loom_spirv_module_value_ref_t){
      .id = entry->variable_id,
      .type_id = entry->array_pointer_type_id,
      .value_type = value_type,
  };
  return iree_ok_status();
}
