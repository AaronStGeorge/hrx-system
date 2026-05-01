// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

typedef struct loom_low_materialized_spill_slot_t {
  // Table spill slot ordinal represented by this generated storage value.
  uint32_t slot_index;
  // SSA value ID produced by the generated low.storage.reserve op.
  loom_value_id_t storage_value_id;
} loom_low_materialized_spill_slot_t;

static iree_status_t loom_low_allocation_emit_materialized_spill(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, iree_diagnostic_emitter_t emitter) {
  if (plan->assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan references an out-of-range assignment");
  }
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, plan->value_id)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          table->module, assignment->value_class)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, storage_value_id)),
      loom_param_u32(plan->byte_size),
      loom_param_u32(plan->store_count),
      loom_param_u32(plan->reload_count),
  };
  loom_diagnostic_emission_t emission = {
      .op = loom_low_diagnostic_value_origin_op(table->module, plan->value_id,
                                                table->function_op),
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 9),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_allocation_map_slot_space(
    loom_low_spill_slot_space_t slot_space, loom_storage_space_t* out_space) {
  switch (slot_space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
      *out_space = LOOM_STORAGE_SPACE_STACK;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
      *out_space = LOOM_STORAGE_SPACE_SCRATCH;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
      *out_space = LOOM_STORAGE_SPACE_PRIVATE;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      *out_space = LOOM_STORAGE_SPACE_WORKGROUP;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown allocation spill slot space %u",
                              (unsigned)slot_space);
  }
}

static iree_status_t loom_low_allocation_verify_no_existing_storage_traffic(
    const loom_op_t* function_op) {
  loom_region_t* body = loom_low_function_body((loom_op_t*)function_op);
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation materialization requires a low function without "
            "existing low.spill or low.reload ops");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_set_storage_value_name(
    loom_module_t* module, loom_symbol_ref_t function_ref, uint32_t slot_index,
    loom_value_id_t storage_value_id, iree_arena_allocator_t* arena) {
  if (function_ref.module_id != 0 ||
      function_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spill storage function symbol is not defined");
  }
  loom_string_id_t function_name_id =
      module->symbols.entries[function_ref.symbol_id].name_id;
  if (function_name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spill storage function symbol has no name");
  }
  iree_string_view_t function_name = module->strings.entries[function_name_id];

  char suffix[64] = {0};
  int suffix_length = snprintf(suffix, sizeof(suffix), "_spill_storage_%u",
                               (unsigned)slot_index);
  if (suffix_length < 0 || (iree_host_size_t)suffix_length >= sizeof(suffix)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "spill storage value name overflow");
  }

  iree_host_size_t name_length = 0;
  if (!iree_host_size_checked_add(
          function_name.size, (iree_host_size_t)suffix_length, &name_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "spill storage value name overflow");
  }

  char* name_storage = NULL;
  if (name_length > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, name_length, (void**)&name_storage));
    memcpy(name_storage, function_name.data, function_name.size);
    memcpy(name_storage + function_name.size, suffix,
           (iree_host_size_t)suffix_length);
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_make_string_view(name_storage, name_length), &name_id));
  return loom_module_set_value_name(module, storage_value_id, name_id);
}

static bool loom_low_allocation_entry_preamble_op(const loom_op_t* op) {
  return loom_low_live_in_isa(op) || loom_low_resource_isa(op);
}

static bool loom_low_allocation_entry_storage_prefix_op(const loom_op_t* op) {
  return loom_low_allocation_entry_preamble_op(op) ||
         loom_low_storage_reserve_isa(op) || loom_low_spill_isa(op);
}

static iree_status_t loom_low_allocation_set_storage_insertion_point(
    loom_builder_t* builder, loom_op_t* function_op) {
  loom_region_t* body = loom_low_function_body(function_op);
  if (!body || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function has no entry block");
  }
  loom_builder_enter_region(builder, function_op, body);
  loom_block_t* entry_block = loom_region_entry_block(body);

  const loom_op_t* insertion_anchor = NULL;
  const loom_op_t* scan_op = entry_block->first_op;
  while (scan_op && loom_low_allocation_entry_preamble_op(scan_op)) {
    insertion_anchor = scan_op;
    scan_op = scan_op->next_op;
  }
  if (insertion_anchor) {
    loom_builder_set_after(builder, insertion_anchor);
  } else if (entry_block->first_op) {
    loom_builder_set_before(builder, entry_block->first_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_set_block_arg_spill_insertion_point(
    loom_builder_t* builder, const loom_op_t* function_op,
    loom_block_t* block) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  const loom_block_t* entry_block =
      body ? loom_region_const_entry_block(body) : NULL;
  if (block == entry_block) {
    const loom_op_t* insertion_anchor = NULL;
    const loom_op_t* scan_op = block->first_op;
    while (scan_op && loom_low_allocation_entry_storage_prefix_op(scan_op)) {
      insertion_anchor = scan_op;
      scan_op = scan_op->next_op;
    }
    if (insertion_anchor) {
      loom_builder_set_after(builder, insertion_anchor);
      return iree_ok_status();
    }
  }
  if (block->first_op) {
    loom_builder_set_before(builder, block->first_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_storage_reserves(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena, loom_low_materialized_spill_slot_t* slots) {
  loom_symbol_ref_t function_ref = loom_low_function_callee(table->function_op);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, NULL, &builder);
  IREE_RETURN_IF_ERROR(loom_low_allocation_set_storage_insertion_point(
      &builder, (loom_op_t*)table->function_op));
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* plan = &table->spill_plans[i];
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (slots[j].slot_index == plan->slot_index) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation materialization requires unique spill slot "
            "ordinals");
      }
    }

    loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_COUNT_;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_map_slot_space(plan->slot_space, &storage_space));

    loom_op_t* reserve_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
        &builder, (int64_t)plan->byte_size, (int64_t)plan->byte_alignment,
        loom_type_storage(storage_space), table->function_op->location,
        &reserve_op));
    loom_value_id_t storage_value_id =
        loom_low_storage_reserve_storage(reserve_op);
    IREE_RETURN_IF_ERROR(loom_low_allocation_set_storage_value_name(
        module, function_ref, plan->slot_index, storage_value_id, arena));
    loom_builder_set_after(&builder, reserve_op);
    slots[i] = (loom_low_materialized_spill_slot_t){
        .slot_index = plan->slot_index,
        .storage_value_id = storage_value_id,
    };
  }

  return iree_ok_status();
}

static iree_status_t loom_low_allocation_snapshot_value_uses(
    loom_module_t* module, loom_value_id_t value_id,
    iree_arena_allocator_t* arena, loom_use_t** out_uses,
    uint32_t* out_use_count) {
  loom_value_t* value = loom_module_value(module, value_id);
  *out_use_count = value->use_count;
  *out_uses = NULL;
  if (value->use_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value->use_count, sizeof(**out_uses), (void**)out_uses));
  memcpy(*out_uses, loom_value_uses(value),
         (iree_host_size_t)value->use_count * sizeof(**out_uses));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_spill_store(
    loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id) {
  loom_value_t* value = loom_module_value(module, plan->value_id);
  loom_builder_t builder;
  loom_op_t* spill_op = NULL;
  if (loom_value_is_block_arg(value)) {
    loom_block_t* block = loom_def_block(value->def);
    if (!block) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "spilled block argument has no defining block");
    }
    loom_builder_initialize(module, &module->arena, block, &builder);
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_set_block_arg_spill_insertion_point(
            &builder, function_op, block));
    return loom_low_spill_build(&builder, plan->value_id, storage_value_id, 0,
                                function_op->location, &spill_op);
  }

  loom_op_t* defining_op = loom_def_op(value->def);
  if (!defining_op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spilled op result has no defining op");
  }
  if (!defining_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spilled op result defining op is detached");
  }
  loom_builder_initialize(module, &module->arena, defining_op->parent_block,
                          &builder);
  loom_builder_set_after(&builder, defining_op);
  return loom_low_spill_build(&builder, plan->value_id, storage_value_id, 0,
                              defining_op->location, &spill_op);
}

static iree_status_t loom_low_allocation_insert_reload_for_use(
    loom_module_t* module, const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  uint16_t operand_index = loom_use_operand_index(use);
  if (!user_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spilled value user op is detached");
  }
  if (operand_index >= user_op->operand_count ||
      loom_op_const_operands(user_op)[operand_index] != plan->value_id) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation spill plan is stale for value %u",
                            (unsigned)plan->value_id);
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, user_op->parent_block,
                          &builder);
  loom_builder_set_before(&builder, user_op);
  loom_op_t* reload_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_reload_build(&builder, storage_value_id, 0,
                            loom_module_value_type(module, plan->value_id),
                            user_op->location, &reload_op));
  return loom_op_set_operand(module, user_op, operand_index,
                             loom_low_reload_result(reload_op));
}

static iree_status_t loom_low_allocation_materialize_one_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* result) {
  loom_use_t* uses = NULL;
  uint32_t use_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_snapshot_value_uses(
      module, plan->value_id, arena, &uses, &use_count));
  if (use_count != plan->reload_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan reload count is stale for value %u",
        (unsigned)plan->value_id);
  }
  uint32_t expected_store_count = use_count > 0 ? 1u : 0u;
  if (plan->store_count != expected_store_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan store count is stale for value %u",
        (unsigned)plan->value_id);
  }

  if (plan->store_count > 0) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_spill_store(
        module, table->function_op, plan, storage_value_id));
    if (result->spill_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill count overflow");
    }
    ++result->spill_count;
  }

  for (uint32_t i = 0; i < use_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_reload_for_use(
        module, plan, storage_value_id, uses[i]));
    if (result->reload_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload count overflow");
    }
    ++result->reload_count;
  }
  return loom_low_allocation_emit_materialized_spill(table, plan,
                                                     storage_value_id, emitter);
}

iree_status_t loom_low_allocation_materialize_spills(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_materialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* out_result) {
  loom_low_allocation_materialization_result_t result = {0};
  if (out_result) *out_result = result;

  IREE_RETURN_IF_ERROR(loom_low_allocation_verify_table(table));
  if (table->module != module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation table belongs to a different module");
  }
  if (table->spill_plan_count == 0) return iree_ok_status();

  const bool allow_existing_storage_traffic =
      options && options->allow_existing_storage_traffic;
  const iree_diagnostic_emitter_t emitter =
      options ? options->emitter : (iree_diagnostic_emitter_t){0};
  if (!allow_existing_storage_traffic) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_no_existing_storage_traffic(
        table->function_op));
  }
  if (table->spill_plan_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "materialized storage count overflow");
  }

  loom_low_materialized_spill_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, table->spill_plan_count, sizeof(*slots), (void**)&slots));
  memset(slots, 0, table->spill_plan_count * sizeof(*slots));

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_insert_storage_reserves(module, table, arena, slots));
  result.storage_count = (uint32_t)table->spill_plan_count;

  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_one_spill_plan(
        module, table, &table->spill_plans[i], slots[i].storage_value_id,
        emitter, arena, &result));
  }

  if (out_result) *out_result = result;
  return iree_ok_status();
}
