// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

typedef struct loom_low_materialized_spill_slot_t {
  // Sidecar spill slot ordinal represented by this generated symbol.
  uint32_t slot_index;
  // Symbol reference for the generated low.slot record.
  loom_symbol_ref_t symbol_ref;
} loom_low_materialized_spill_slot_t;

static iree_status_t loom_low_allocation_emit_materialized_spill(
    const loom_low_allocation_sidecar_t* sidecar,
    const loom_low_allocation_spill_plan_t* plan, loom_symbol_ref_t slot_ref,
    iree_diagnostic_emitter_t emitter) {
  if (plan->assignment_index >= sidecar->assignment_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan references an out-of-range assignment");
  }
  const loom_low_allocation_assignment_t* assignment =
      &sidecar->assignments[plan->assignment_index];
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&sidecar->target)),
      loom_param_string(loom_low_diagnostic_export_name(&sidecar->target)),
      loom_param_string(loom_low_diagnostic_config_key(&sidecar->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          sidecar->module, sidecar->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(sidecar->module, plan->value_id)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          sidecar->module, assignment->value_class)),
      loom_param_string(
          loom_low_diagnostic_symbol_name(sidecar->module, slot_ref)),
      loom_param_u32(plan->byte_size),
      loom_param_u32(plan->store_count),
      loom_param_u32(plan->reload_count),
  };
  loom_diagnostic_emission_t emission = {
      .op = loom_low_diagnostic_value_origin_op(sidecar->module, plan->value_id,
                                                sidecar->function_op),
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 9),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_allocation_map_slot_space(
    uint8_t slot_space, uint8_t* out_low_space) {
  switch (slot_space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
      *out_low_space = LOOM_LOW_SLOT_SPACE_STACK;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
      *out_low_space = LOOM_LOW_SLOT_SPACE_SCRATCH;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
      *out_low_space = LOOM_LOW_SLOT_SPACE_PRIVATE;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      *out_low_space = LOOM_LOW_SLOT_SPACE_LDS;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown allocation spill slot space %u",
                              (unsigned)slot_space);
  }
}

static iree_status_t loom_low_allocation_verify_no_existing_slot_traffic(
    const loom_op_t* function_op) {
  loom_region_t* body = loom_low_func_def_body(function_op);
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation materialization requires a low.func.def without "
            "existing low.spill or low.reload ops");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_create_unique_slot_symbol(
    loom_module_t* module, loom_symbol_ref_t function_ref, uint32_t slot_index,
    iree_arena_allocator_t* arena, loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  if (function_ref.module_id != 0 ||
      function_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spill slot function symbol is not defined");
  }
  loom_string_id_t function_name_id =
      module->symbols.entries[function_ref.symbol_id].name_id;
  if (function_name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spill slot function symbol has no name");
  }
  iree_string_view_t function_name = module->strings.entries[function_name_id];

  for (uint32_t collision_index = 0; collision_index < 1024;
       ++collision_index) {
    char suffix[64] = {0};
    int suffix_length = 0;
    if (collision_index == 0) {
      suffix_length =
          snprintf(suffix, sizeof(suffix), "_spill_%u", (unsigned)slot_index);
    } else {
      suffix_length = snprintf(suffix, sizeof(suffix), "_spill_%u_%u",
                               (unsigned)slot_index, (unsigned)collision_index);
    }
    if (suffix_length < 0 ||
        (iree_host_size_t)suffix_length >= sizeof(suffix)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "spill slot symbol name overflow");
    }

    iree_host_size_t name_length = 0;
    if (!iree_host_size_checked_add(function_name.size,
                                    (iree_host_size_t)suffix_length,
                                    &name_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "spill slot symbol name overflow");
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
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_symbol_ref =
        (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "could not create a unique spill slot symbol");
}

static iree_status_t loom_low_allocation_insert_slot_records(
    loom_module_t* module, const loom_low_allocation_sidecar_t* sidecar,
    iree_arena_allocator_t* arena, loom_low_materialized_spill_slot_t* slots) {
  loom_symbol_ref_t function_ref =
      loom_low_func_def_callee(sidecar->function_op);
  if (!sidecar->function_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low.func.def is not inserted in a block");
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena,
                          sidecar->function_op->parent_block, &builder);
  const loom_op_t* insertion_anchor = sidecar->function_op;
  for (iree_host_size_t i = 0; i < sidecar->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* plan = &sidecar->spill_plans[i];
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (slots[j].slot_index == plan->slot_index) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation materialization requires unique spill slot "
            "ordinals");
      }
    }

    loom_symbol_ref_t slot_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_low_allocation_create_unique_slot_symbol(
        module, function_ref, plan->slot_index, arena, &slot_ref));

    uint8_t low_slot_space = 0;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_map_slot_space(plan->slot_space, &low_slot_space));

    loom_builder_set_after(&builder, insertion_anchor);
    loom_op_t* slot_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slot_build(
        &builder, slot_ref, function_ref, low_slot_space,
        (int64_t)plan->byte_size, (int64_t)plan->byte_alignment,
        sidecar->function_op->location, &slot_op));
    insertion_anchor = slot_op;
    slots[i] = (loom_low_materialized_spill_slot_t){
        .slot_index = plan->slot_index,
        .symbol_ref = slot_ref,
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
    const loom_low_allocation_spill_plan_t* plan, loom_symbol_ref_t slot_ref) {
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
    if (block->first_op) {
      loom_builder_set_before(&builder, block->first_op);
    }
    return loom_low_spill_build(&builder, plan->value_id, slot_ref, 0,
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
  return loom_low_spill_build(&builder, plan->value_id, slot_ref, 0,
                              defining_op->location, &spill_op);
}

static iree_status_t loom_low_allocation_insert_reload_for_use(
    loom_module_t* module, const loom_low_allocation_spill_plan_t* plan,
    loom_symbol_ref_t slot_ref, loom_use_t use) {
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
  IREE_RETURN_IF_ERROR(loom_low_reload_build(
      &builder, slot_ref, 0, loom_module_value_type(module, plan->value_id),
      user_op->location, &reload_op));
  return loom_op_set_operand(module, user_op, operand_index,
                             loom_low_reload_result(reload_op));
}

static iree_status_t loom_low_allocation_materialize_one_spill_plan(
    loom_module_t* module, const loom_low_allocation_sidecar_t* sidecar,
    const loom_low_allocation_spill_plan_t* plan, loom_symbol_ref_t slot_ref,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
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
        module, sidecar->function_op, plan, slot_ref));
    if (result->spill_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill count overflow");
    }
    ++result->spill_count;
  }

  for (uint32_t i = 0; i < use_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_reload_for_use(
        module, plan, slot_ref, uses[i]));
    if (result->reload_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload count overflow");
    }
    ++result->reload_count;
  }
  return loom_low_allocation_emit_materialized_spill(sidecar, plan, slot_ref,
                                                     emitter);
}

iree_status_t loom_low_allocation_materialize_spills(
    loom_module_t* module, const loom_low_allocation_sidecar_t* sidecar,
    const loom_low_allocation_materialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* out_result) {
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "module is required");
  }
  if (!arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "arena is required");
  }
  loom_low_allocation_materialization_result_t result = {0};
  if (out_result) *out_result = result;

  IREE_RETURN_IF_ERROR(loom_low_allocation_verify_sidecar(sidecar));
  if (sidecar->module != module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar belongs to a different module");
  }
  if (sidecar->spill_plan_count == 0) return iree_ok_status();

  const bool allow_existing_slot_traffic =
      options && options->allow_existing_slot_traffic;
  const iree_diagnostic_emitter_t emitter =
      options ? options->emitter : (iree_diagnostic_emitter_t){0};
  if (!allow_existing_slot_traffic) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_no_existing_slot_traffic(
        sidecar->function_op));
  }
  if (sidecar->spill_plan_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "materialized slot count overflow");
  }

  loom_low_materialized_spill_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, sidecar->spill_plan_count, sizeof(*slots), (void**)&slots));
  memset(slots, 0, sidecar->spill_plan_count * sizeof(*slots));

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_insert_slot_records(module, sidecar, arena, slots));
  result.slot_count = (uint32_t)sidecar->spill_plan_count;

  for (iree_host_size_t i = 0; i < sidecar->spill_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_one_spill_plan(
        module, sidecar, &sidecar->spill_plans[i], slots[i].symbol_ref, emitter,
        arena, &result));
  }

  if (out_result) *out_result = result;
  return iree_ok_status();
}
