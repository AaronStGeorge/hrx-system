// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_access.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/registers.h"

typedef struct loom_amdgpu_sanitizer_lower_state_t {
  // True once the runtime config symbols have been looked up or created.
  bool has_config_symbols;
  // Module-local feedback channel configuration symbol.
  loom_symbol_ref_t feedback_config_symbol;
  // Module-local ASAN shadow configuration symbol.
  loom_symbol_ref_t asan_config_symbol;
  // True once read access failures have a shared cold report/trap island.
  bool has_read_island;
  // Shared cold island for read access failures.
  loom_amdgpu_sanitizer_access_report_trap_island_t read_island;
  // True once write access failures have a shared cold report/trap island.
  bool has_write_island;
  // Shared cold island for write access failures.
  loom_amdgpu_sanitizer_access_report_trap_island_t write_island;
  // True once atomic access failures have a shared cold report/trap island.
  bool has_atomic_island;
  // Shared cold island for atomic access failures.
  loom_amdgpu_sanitizer_access_report_trap_island_t atomic_island;
} loom_amdgpu_sanitizer_lower_state_t;

static int loom_amdgpu_sanitizer_lower_state_key;

static iree_status_t loom_amdgpu_sanitizer_get_or_create_symbol(
    loom_module_t* module, iree_string_view_t name,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
  }
  *out_symbol_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_lower_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_sanitizer_lower_state_key, sizeof(*state),
      (void**)&state));
  if (!state->has_config_symbols) {
    loom_module_t* module = loom_low_lower_context_module(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV("iree_feedback_config"),
        &state->feedback_config_symbol));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV("iree_asan_config"), &state->asan_config_symbol));
    state->has_config_symbols = true;
  }
  *out_state = state;
  return iree_ok_status();
}

static bool loom_amdgpu_sanitizer_assert_access_kinds(
    loom_sanitizer_assert_access_kind_t kind,
    loom_low_source_memory_operation_kind_t* out_source_kind,
    loom_amdgpu_memory_operation_kind_t* out_memory_kind,
    loom_amdgpu_sanitizer_access_kind_t* out_report_kind) {
  switch (kind) {
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      *out_memory_kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      *out_memory_kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      *out_memory_kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_sanitizer_access_payload_type(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_type_t* out_vector_type) {
  *out_vector_type = loom_type_none();
  if (view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  *out_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  return true;
}

iree_status_t loom_amdgpu_select_sanitizer_assert_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_access_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_access_plan_t){0};
  *out_selected = false;
  if (!loom_sanitizer_assert_access_isa(source_op)) {
    return iree_ok_status();
  }

  loom_low_source_memory_operation_kind_t source_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  loom_amdgpu_memory_operation_kind_t memory_kind =
      LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
  loom_amdgpu_sanitizer_access_kind_t report_kind =
      LOOM_AMDGPU_SANITIZER_ACCESS_KIND_UNKNOWN;
  if (!loom_amdgpu_sanitizer_assert_access_kinds(
          loom_sanitizer_assert_access_kind(source_op), &source_kind,
          &memory_kind, &report_kind)) {
    return iree_ok_status();
  }

  loom_type_t vector_type = loom_type_none();
  const loom_value_id_t view_value_id =
      loom_sanitizer_assert_access_view(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_sanitizer_access_payload_type(module, view_value_id,
                                                 &vector_type)) {
    return iree_ok_status();
  }

  loom_low_source_memory_access_plan_t source = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_low_source_memory_access_plan_build_indexed(
          module, loom_low_lower_context_fact_table(context), source_kind,
          view_value_id, loom_sanitizer_assert_access_indices(source_op),
          loom_sanitizer_assert_access_static_indices(source_op), vector_type,
          cache_policy, &source, &source_diagnostic)) {
    return iree_ok_status();
  }
  const uint64_t access_size =
      (uint64_t)source.element_byte_count * source.vector_lane_count;
  if (access_size == 0 || access_size > 8) {
    return iree_ok_status();
  }

  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_memory_access_select_flat_global_address(
          module, loom_low_lower_context_descriptor_set(context), memory_kind,
          &source, vector_type, &out_plan->address, &diagnostic)) {
    return iree_ok_status();
  }
  out_plan->report_access_kind = report_kind;
  out_plan->access_size = (uint32_t)access_size;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sgpr_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, value, &sgpr_value));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(context, source_op,
                                                        sgpr_value, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_emit_sgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
                                    sgpr_type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_emit_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_value);
}

static loom_amdgpu_sanitizer_access_report_trap_island_t*
loom_amdgpu_sanitizer_island_for_kind(
    loom_amdgpu_sanitizer_lower_state_t* state,
    loom_amdgpu_sanitizer_access_kind_t access_kind, bool** out_has_island) {
  switch (access_kind) {
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ:
      *out_has_island = &state->has_read_island;
      return &state->read_island;
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE:
      *out_has_island = &state->has_write_island;
      return &state->write_island;
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC:
      *out_has_island = &state->has_atomic_island;
      return &state->atomic_island;
    default:
      *out_has_island = NULL;
      return NULL;
  }
}

static iree_status_t loom_amdgpu_sanitizer_get_access_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_sanitizer_lower_state_t* state,
    loom_amdgpu_sanitizer_access_kind_t access_kind,
    loom_location_id_t location,
    const loom_amdgpu_sanitizer_access_report_trap_island_t** out_island) {
  *out_island = NULL;
  bool* has_island = NULL;
  loom_amdgpu_sanitizer_access_report_trap_island_t* island =
      loom_amdgpu_sanitizer_island_for_kind(state, access_kind, &has_island);
  if (island == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unsupported AMDGPU sanitizer access kind");
  }
  if (!*has_island) {
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_trap_island(
        builder, descriptor_set, saved_ip.block, state->feedback_config_symbol,
        access_kind, LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE, location, island));
    loom_builder_restore(builder, saved_ip);
    *has_island = true;
  }
  *out_island = island;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_sanitizer_assert_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_access_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_sanitizer_assert_access_view(source_op), &low_resource));
  loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
      context, source_op, &plan->address, low_resource, &fault_address));

  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  loom_amdgpu_sanitizer_access_check_t check = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_check(
      builder, descriptor_set, state->asan_config_symbol, fault_address,
      plan->access_size, source_op->location, &check));

  loom_amdgpu_sanitizer_report_source_t source = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, 0, &source.dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_sgpr_u32_constant(
      context, source_op, 0, &source.workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u32_constant(
      context, source_op, 0, &source.workitem_id_x));

  loom_amdgpu_sanitizer_access_report_t report = {
      .access_kind = plan->report_access_kind,
      .flags = LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE,
      .fault_address = fault_address,
      .shadow_address = check.shadow_address,
      .shadow_value = check.shadow_value,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
      context, source_op, plan->access_size, &report.access_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
      context, source_op, source_op->location, &report.site_id));

  const loom_amdgpu_sanitizer_access_report_trap_island_t* island = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_access_island(
      builder, descriptor_set, state, plan->report_access_kind,
      source_op->location, &island));
  loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
  return loom_amdgpu_build_sanitizer_access_report_failure_mask_branch(
      builder, descriptor_set, island, check.failure_mask, &source, &report,
      source_op->location, &branch);
}
