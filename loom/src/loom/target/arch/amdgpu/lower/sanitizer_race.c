// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race.h"

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"

static iree_string_view_t
loom_amdgpu_sanitizer_race_source_memory_rejection_key(
    const loom_low_source_memory_access_diagnostic_t* diagnostic) {
  if (diagnostic->rejection_bits != 0) {
    return loom_low_source_memory_access_rejection_key(
        diagnostic->rejection_bits);
  }
  return IREE_SV("target_contract.sanitizer_race.source_memory_access");
}

static bool loom_amdgpu_sanitizer_race_access_source_kind(
    loom_sanitizer_race_access_kind_t kind,
    loom_low_source_memory_operation_kind_t* out_source_kind) {
  switch (kind) {
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_sanitizer_race_access_payload_type(
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

static iree_status_t loom_amdgpu_sanitizer_race_reject_memory_selection(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_access_rejection_diagnostic(
      context, op, source, diagnostic, &handled));
  if (handled) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(
      context, op,
      loom_amdgpu_memory_access_rejection_key(diagnostic->rejection_bits));
}

static iree_status_t loom_amdgpu_sanitizer_race_verify_access_address(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t view_value_id = loom_sanitizer_race_access_view(op);

  loom_low_source_memory_operation_kind_t source_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (!loom_amdgpu_sanitizer_race_access_source_kind(
          loom_sanitizer_race_access_kind(op), &source_kind)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("target_contract.sanitizer_race.access_kind"));
  }

  loom_type_t vector_type = loom_type_none();
  if (!loom_amdgpu_sanitizer_race_access_payload_type(module, view_value_id,
                                                      &vector_type)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("target_contract.sanitizer_race.payload_type"));
  }

  loom_low_source_memory_access_plan_t source = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_low_source_memory_access_plan_build_indexed(
          module, loom_target_low_legality_fact_table(context), source_kind,
          view_value_id, loom_sanitizer_race_access_indices(op),
          loom_sanitizer_race_access_static_indices(op), vector_type,
          cache_policy, &source, &source_diagnostic)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_sanitizer_race_source_memory_rejection_key(
            &source_diagnostic));
  }

  if (source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.workgroup_memory_required"));
  }

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  loom_amdgpu_memory_access_t access = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_memory_access_select_u32_vaddr_byte_offset(
          module, loom_target_low_legality_fact_table(context), view_regions,
          loom_target_low_legality_function(context), &source, &access,
          &diagnostic)) {
    return loom_amdgpu_sanitizer_race_reject_memory_selection(
        context, op, &source, &diagnostic);
  }

  return loom_amdgpu_low_legality_reject(
      context, op,
      IREE_SV("target_contract.sanitizer_race.runtime_config_required"));
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_sanitizer_race_access_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (loom_sanitizer_race_access_atomic(op)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.atomic_access_unsupported"));
  }
  return loom_amdgpu_sanitizer_race_verify_access_address(context, op);
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_sync(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_sanitizer_race_sync_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (loom_sanitizer_race_sync_memory_space(op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.workgroup_memory_required"));
  }
  if (loom_sanitizer_race_sync_scope(op) != LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.workgroup_scope_required"));
  }

  switch (loom_sanitizer_race_sync_ordering(op)) {
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.runtime_config_required"));
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
    case LOOM_ATOMIC_ORDERING_RELEASE:
    case LOOM_ATOMIC_ORDERING_RELAXED:
    case LOOM_ATOMIC_ORDERING_COUNT_:
    default:
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.acq_rel_sync_required"));
  }
}
