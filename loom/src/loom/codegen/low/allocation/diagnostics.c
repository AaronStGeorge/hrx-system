// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/diagnostics.h"

#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"

static iree_status_t loom_low_allocation_emit_predicted_spills(
    const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[spill_plan->assignment_index];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&table->target)),
        loom_param_string(loom_low_diagnostic_export_name(&table->target)),
        loom_param_string(loom_low_diagnostic_config_key(&table->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            table->module, table->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(table->module,
                                                         spill_plan->value_id)),
        loom_param_string(loom_low_diagnostic_value_class_name(
            table->target.descriptor_set, assignment->value_class)),
        loom_param_u32(spill_plan->byte_size),
        loom_param_u32(spill_plan->store_count),
        loom_param_u32(spill_plan->reload_count),
        loom_param_string(IREE_SV("register-budget-conflict")),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            table->module, spill_plan->value_id, table->function_op),
        .error = LOOM_ERR_BACKEND_008,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_copy_decision_name(
    loom_low_allocation_copy_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_COPY_COALESCED:
      return IREE_SV("accepted");
    case LOOM_LOW_ALLOCATION_COPY_MATERIALIZED:
      return IREE_SV("rejected");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_allocation_coalescing_constraint(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_copy_decision_t* copy_decision) {
  const loom_low_allocation_assignment_t* source_assignment =
      &table->assignments[copy_decision->source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &table->assignments[copy_decision->result_assignment_index];
  if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
    return IREE_SV("source-not-register-like");
  }
  if (!loom_low_allocation_assignment_is_register_like(result_assignment)) {
    return IREE_SV("result-not-register-like");
  }
  if (loom_low_allocation_storage_assignment_locations_share(
          table->target.descriptor_set, source_assignment, result_assignment)) {
    return IREE_SV("assigned-locations-match");
  }
  return IREE_SV("assigned-locations-differ");
}

static iree_status_t loom_low_allocation_emit_copy_decisions(
    const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->copy_decision_count; ++i) {
    const loom_low_allocation_copy_decision_t* copy_decision =
        &table->copy_decisions[i];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&table->target)),
        loom_param_string(loom_low_diagnostic_export_name(&table->target)),
        loom_param_string(loom_low_diagnostic_config_key(&table->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            table->module, table->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(
            table->module, copy_decision->source_value_id)),
        loom_param_string(loom_low_diagnostic_value_name(
            table->module, copy_decision->result_value_id)),
        loom_param_string(
            loom_low_allocation_copy_decision_name(copy_decision->kind)),
        loom_param_string(
            loom_low_allocation_coalescing_constraint(table, copy_decision)),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            table->module, copy_decision->result_value_id, table->function_op),
        .error = LOOM_ERR_BACKEND_006,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_diagnostics_emit(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_diagnostic_flags_t flags,
    iree_diagnostic_emitter_t emitter) {
  IREE_ASSERT_ARGUMENT(table);
  if (iree_any_bit_set(flags,
                       LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_predicted_spills(table, emitter));
  }
  if (iree_any_bit_set(flags, LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_copy_decisions(table, emitter));
  }
  return iree_ok_status();
}
