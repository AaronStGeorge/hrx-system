// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/kernel_async_legality.h"
#include "loom/analysis/vector_memory_footprint.h"
#include "loom/codegen/low/contract_query.h"
#include "loom/codegen/low/lower_internal.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"

static bool loom_low_lower_type_is_none(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_NONE;
}

static bool loom_low_lower_abi_argument_kind_is_known(
    loom_low_lower_abi_argument_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT:
    case LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_resource_import_kind_is_known(
    loom_low_resource_import_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_function_attr_present(loom_func_like_t function,
                                                 uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(function.op)[attr_index]);
}

static loom_target_abi_kind_t loom_low_lower_function_abi(
    const loom_low_lower_context_t* context) {
  const uint8_t abi_attr_index =
      context->source_function.vtable->abi_attr_index;
  if (loom_low_lower_function_attr_present(context->source_function,
                                           abi_attr_index)) {
    return (loom_target_abi_kind_t)loom_func_like_abi(context->source_function);
  }
  return context->options->bundle->export_plan->abi_kind;
}

static iree_status_t loom_low_lower_map_direct_argument(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_low_lower_map_value(context, source_op, source_argument_id,
                                  &out_argument->abi_type);
}

static iree_status_t loom_low_lower_map_argument(
    loom_low_lower_context_t* context, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  uint32_t previous_error_count = context->result->error_count;
  if (context->policy->map_argument.fn == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_map_direct_argument(context, context->source_function.op,
                                           source_argument_id, out_argument));
  } else {
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = loom_type_none(),
        .resource_semantic_type = loom_type_none(),
    };
    IREE_RETURN_IF_ERROR(context->policy->map_argument.fn(
        context->policy->map_argument.user_data, context,
        context->source_function.op, source_argument_index, source_argument_id,
        out_argument));
  }

  IREE_ASSERT(loom_low_lower_abi_argument_kind_is_known(out_argument->kind));
  if (loom_low_lower_type_is_none(out_argument->abi_type)) {
    if (context->result->error_count == previous_error_count) {
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
          context, context->source_function.op, IREE_SV("argument"),
          IREE_SV("<unknown>"),
          IREE_SV("target-low argument ABI policy did not produce a register "
                  "type")));
    }
    return iree_ok_status();
  }
  IREE_ASSERT(loom_type_is_register(out_argument->abi_type));

  if (out_argument->kind == LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
    return iree_ok_status();
  }
  IREE_ASSERT(loom_low_lower_resource_import_kind_is_known(
      out_argument->resource_import_kind));
  IREE_ASSERT_GE(out_argument->resource_index, 0);
  if (loom_low_lower_type_is_none(out_argument->resource_semantic_type)) {
    out_argument->resource_semantic_type =
        loom_module_value_type(context->module, source_argument_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_initialize_argument_map(
    loom_low_lower_context_t* context) {
  if (context->lowering.argument_map != NULL) {
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  context->lowering.argument_map_count = argument_count;
  if (argument_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, argument_count, sizeof(*context->lowering.argument_map),
      (void**)&context->lowering.argument_map));
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_map_argument(
        context, i, source_arguments[i], &context->lowering.argument_map[i]));
  }
  return iree_ok_status();
}

static uint16_t loom_low_lower_direct_argument_count(
    const loom_low_lower_context_t* context) {
  uint16_t direct_argument_count = 0;
  for (uint16_t i = 0; i < context->lowering.argument_map_count; ++i) {
    if (context->lowering.argument_map[i].kind ==
        LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
      ++direct_argument_count;
    }
  }
  return direct_argument_count;
}

static void loom_low_lower_assert_options(
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options) {
  IREE_ASSERT(module != NULL);
  IREE_ASSERT(loom_func_like_isa(source_function));
  IREE_ASSERT(options != NULL);
  IREE_ASSERT(source_function.op->kind == LOOM_OP_FUNC_DEF ||
              source_function.op->kind == LOOM_OP_KERNEL_DEF);
  IREE_ASSERT(loom_symbol_ref_is_valid(options->target_ref));
  IREE_ASSERT_EQ(options->target_ref.module_id, 0);
  IREE_ASSERT_LT(options->target_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* target_symbol =
      &module->symbols.entries[options->target_ref.symbol_id];
  IREE_ASSERT(target_symbol->defining_op != NULL);
  IREE_ASSERT_EQ(target_symbol->defining_op->kind, LOOM_OP_TARGET_PROFILE);
  IREE_ASSERT(options->bundle != NULL);
  IREE_ASSERT(options->bundle->snapshot != NULL);
  IREE_ASSERT(options->bundle->export_plan != NULL);
  IREE_ASSERT(options->bundle->config != NULL);
  IREE_ASSERT(options->fact_table != NULL);
  IREE_ASSERT(options->fact_table->context.target_bundle == options->bundle);
  IREE_ASSERT(options->descriptor_registry != NULL);
  IREE_ASSERT(options->policy != NULL);
}

static iree_status_t loom_low_lower_intern_type_id(
    loom_low_lower_context_t* context, loom_type_t type,
    loom_type_id_t* out_type_id) {
  return loom_module_intern_type_id(context->module, type, out_type_id);
}

static iree_status_t loom_low_lower_check_mapped_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_low_type) {
  uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_value_id, out_low_type));
  if (loom_low_lower_type_is_none(*out_low_type)) {
    if (context->result->error_count == previous_error_count) {
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
          context, source_op, IREE_SV("value"), IREE_SV("<unknown>"),
          IREE_SV("target-low value policy did not produce a register type")));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_check_function_signature(
    loom_low_lower_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));

  const loom_value_id_t* result_ids =
      loom_op_const_results(context->source_function.op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    loom_type_t low_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
        context, context->source_function.op, result_ids[i], &low_type));
  }

  uint16_t predicate_count = 0;
  (void)loom_func_like_predicates(context->source_function, &predicate_count);
  if (predicate_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
        context, context->source_function.op, IREE_SV("function"),
        loom_low_lower_context_function_name(context),
        IREE_SV("function predicates need value remapping before target-low "
                "lowering")));
  }
  if (context->source_function.op->tied_result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
        context, context->source_function.op, IREE_SV("function"),
        loom_low_lower_context_function_name(context),
        IREE_SV("tied function results need explicit ABI ownership lowering")));
  }
  return iree_ok_status();
}

static bool loom_low_lower_op_is_structural(const loom_module_t* module,
                                            const loom_op_t* op) {
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_are_fact_identity(traits) ||
      loom_traits_are_value_alias(traits)) {
    return true;
  }
  switch (op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_KERNEL_RETURN:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_is_source_metadata(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_ENCODING_ASSUME_SPEC:
    case LOOM_OP_ENCODING_DEFINE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED:
    case LOOM_OP_ENCODING_LAYOUT_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_STRIDED:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_uses_policy(const loom_module_t* module,
                                          const loom_op_t* op) {
  return !loom_low_lower_op_is_structural(module, op) &&
         !loom_low_lower_op_is_source_metadata(op->kind);
}

static void loom_low_lower_mark_value_storage_required(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  context->lowering.value_storage_flags[source_ordinal] |=
      LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED;
}

static bool loom_low_lower_value_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  return iree_any_bit_set(context->lowering.value_storage_flags[source_ordinal],
                          LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED);
}

static void loom_low_lower_mark_value_slice_storage_required(
    loom_low_lower_context_t* context, loom_value_slice_t values) {
  for (uint16_t i = 0; i < values.count; ++i) {
    loom_low_lower_mark_value_storage_required(context, values.values[i]);
  }
}

static bool loom_low_lower_rule_value_ref_source_value(
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t* out_source_value_id) {
  *out_source_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      IREE_ASSERT_LT(value_ref->index, source_op->operand_count);
      *out_source_value_id =
          loom_op_const_operands(source_op)[value_ref->index];
      return true;
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT_LT(value_ref->index, source_op->result_count);
      *out_source_value_id = loom_op_const_results(source_op)[value_ref->index];
      return true;
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      return false;
    default:
      IREE_ASSERT_UNREACHABLE("unknown source-low value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static void loom_low_lower_mark_rule_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  const loom_low_lower_rule_t* rule = selected_plan->rule;
  IREE_ASSERT(rule_set != NULL);
  IREE_ASSERT(rule != NULL);
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + emit_ordinal);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    for (uint16_t operand_ordinal = 0;
         operand_ordinal < emit->operand_ref_count; ++operand_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(emit->operand_ref_start + operand_ordinal);
      loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
      if (loom_low_lower_rule_value_ref_source_value(
              rule_set, selected_plan->source_op, value_ref_index,
              &source_value_id)) {
        loom_low_lower_mark_value_storage_required(context, source_value_id);
      }
    }
  }
}

static void loom_low_lower_mark_callback_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    loom_low_lower_mark_value_storage_required(context, operands[i]);
  }
}

static void loom_low_lower_mark_structural_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits)) {
    const loom_value_id_t* operands = loom_op_const_operands(source_op);
    for (uint16_t i = 0; i < source_op->operand_count; ++i) {
      loom_low_lower_mark_value_storage_required(context, operands[i]);
    }
    return;
  }
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(source_op->operand_count >= 1);
    loom_low_lower_mark_value_storage_required(
        context, loom_op_const_operands(source_op)[0]);
    return;
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
      loom_low_lower_mark_value_storage_required(
          context, loom_buffer_assume_same_root_buffer(source_op));
      return;
    case LOOM_OP_FUNC_RETURN:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_func_return_operands(source_op));
      return;
    case LOOM_OP_CFG_BR:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_cfg_br_args(source_op));
      return;
    case LOOM_OP_CFG_COND_BR:
      loom_low_lower_mark_value_storage_required(
          context, loom_cfg_cond_br_condition(source_op));
      return;
    case LOOM_OP_KERNEL_RETURN:
    default:
      return;
  }
}

static bool loom_low_lower_can_elide_source_storage(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (source_op->result_count == 0 || source_op->region_count != 0 ||
      source_op->tied_result_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT |
                                   LOOM_TRAIT_UNIQUE_IDENTITY |
                                   LOOM_TRAIT_CONVERGENT)) {
    return false;
  }
  if (loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
      loom_op_regions_have_write_effects(source_op) ||
      loom_op_regions_have_convergent_effects(source_op) ||
      loom_op_regions_have_hints(context->module, source_op)) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_ASSERT_NE(results[i], LOOM_VALUE_ID_INVALID);
    if (loom_low_lower_value_storage_required(context, results[i]) ||
        loom_module_value_has_type_uses(context->module, results[i])) {
      return false;
    }
  }
  return true;
}

static void loom_low_lower_analyze_storage_demands(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_body, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_lower_op_is_structural(context->module, op)) {
        loom_low_lower_mark_structural_storage_demands(context, op);
      }
    }
  }
  for (iree_host_size_t i = 0; i < context->lowering.selected_plan_count; ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.selected_plans[i];
    if (selected_plan->rule != NULL) {
      loom_low_lower_mark_rule_storage_demands(context, selected_plan);
    } else {
      loom_low_lower_mark_callback_plan_storage_demands(context, selected_plan);
    }
  }
  for (iree_host_size_t i = 0; i < context->lowering.selected_plan_count; ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.selected_plans[i];
    if (loom_low_lower_can_elide_source_storage(context,
                                                selected_plan->source_op)) {
      selected_plan->flags |= LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
    }
  }
}

static iree_status_t loom_low_lowering_frame_initialize_value_ordinals(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  return loom_local_value_domain_acquire_for_region(
      context->module, source_body, &context->arena,
      &context->lowering.value_domain);
}

static void loom_low_lowering_frame_deinitialize(
    loom_low_lower_context_t* context) {
  loom_local_value_domain_release(&context->lowering.value_domain);
}

static iree_status_t loom_low_lower_prepare_plan(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  iree_host_size_t plan_capacity = 0;
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_body, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_lower_op_uses_policy(context->module, op)) {
        ++plan_capacity;
      }
    }
  }
  context->lowering.selected_plan_capacity = plan_capacity;
  context->lowering.selected_plan_count = 0;
  context->lowering.selected_plan_emit_index = 0;
  if (plan_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, plan_capacity, sizeof(*context->lowering.selected_plans),
      (void**)&context->lowering.selected_plans));
  if (context->options->table_arena != NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->options->table_arena, plan_capacity,
        sizeof(*context->lowering.memory_access_records),
        (void**)&context->lowering.memory_access_records));
    context->lowering.memory_access_record_capacity = plan_capacity;
  }
  return iree_ok_status();
}

static void loom_low_lower_record_selected_plan(
    loom_low_lower_context_t* context,
    loom_low_lower_selected_plan_t selected_plan) {
  IREE_ASSERT_LT(context->lowering.selected_plan_count,
                 context->lowering.selected_plan_capacity);
  context->lowering.selected_plans[context->lowering.selected_plan_count++] =
      selected_plan;
}

static iree_status_t loom_low_lower_record_selected_rule_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t rule_set_index, const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_selection_t rule_selection) {
  const loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_resolve_emit_program(
      context, rule_set, rule_selection.rule, &resolved_emits));
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .rule_set_index = rule_set_index,
                   .rule_index = rule_selection.rule_index,
                   .rule_set = rule_set,
                   .rule = rule_selection.rule,
                   .resolved_emits = resolved_emits,
                   .plan = loom_low_lower_plan_empty(),
               });
  return iree_ok_status();
}

static bool loom_low_lower_rule_selection_is_better_failure(
    const loom_low_lower_rule_set_t* failed_rule_set,
    loom_low_lower_rule_selection_t failed_rule_selection,
    loom_low_lower_rule_selection_t rule_selection) {
  return rule_selection.has_source_op_span &&
         (failed_rule_set == NULL ||
          rule_selection.matched_guard_count >
              failed_rule_selection.matched_guard_count);
}

static iree_status_t loom_low_lower_select_custom_family(
    loom_low_lower_context_t* context,
    const loom_target_contract_case_t* contract_case,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan) {
  const loom_low_lower_contract_family_t* family =
      &context->policy->contract_families[contract_case->row_index];
  return family->select(family->user_data, context, source_op, out_plan);
}

static iree_status_t loom_low_lower_plan_op_from_contract_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_rule_set_t** inout_failed_rule_set,
    loom_low_lower_rule_selection_t* inout_failed_rule_selection,
    bool* out_selected) {
  *out_selected = false;
  const loom_target_contract_index_t* index = &context->contract_index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const uint16_t case_index = (uint16_t)(op_entry.case_start + i);
    const loom_target_contract_case_t* contract_case =
        &index->cases[case_index];
    if (contract_case->system == LOOM_TARGET_CONTRACT_SYSTEM_CUSTOM_FAMILY) {
      loom_low_lower_plan_t plan = loom_low_lower_plan_empty();
      IREE_RETURN_IF_ERROR(loom_low_lower_select_custom_family(
          context, contract_case, source_op, &plan));
      if (!loom_low_lower_plan_is_empty(plan)) {
        loom_low_lower_record_selected_plan(
            context, (loom_low_lower_selected_plan_t){
                         .source_op = source_op,
                         .rule_set_index = UINT16_MAX,
                         .rule_index = contract_case->row_index,
                         .rule_set = NULL,
                         .rule = NULL,
                         .plan = plan,
                     });
        *out_selected = true;
        return iree_ok_status();
      }
      continue;
    }
    uint16_t rule_index = UINT16_MAX;
    if (!loom_low_lower_contract_case_lower_rule_index(index, contract_case,
                                                       &rule_index)) {
      continue;
    }
    const loom_target_contract_binding_t* binding =
        &index->bindings[contract_case->binding_index];
    const loom_low_lower_rule_set_t* rule_set =
        context->policy->rule_sets.values[binding->rule_set_index];
    loom_low_lower_rule_selection_t rule_selection = {0};
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select_rule_range(
        context, rule_set, source_op, rule_index, 1, &rule_selection));
    if (rule_selection.rule != NULL) {
      IREE_RETURN_IF_ERROR(loom_low_lower_record_selected_rule_plan(
          context, source_op, binding->rule_set_index, rule_set,
          rule_selection));
      *out_selected = true;
      return iree_ok_status();
    }
    if (loom_low_lower_rule_selection_is_better_failure(
            *inout_failed_rule_set, *inout_failed_rule_selection,
            rule_selection)) {
      *inout_failed_rule_set = rule_set;
      *inout_failed_rule_selection = rule_selection;
    }
  }
  return iree_ok_status();
}

typedef struct loom_low_lower_contract_query_state_t {
  // Mutable lowering context whose scoped arena and target policy back the
  // read-only contract query.
  loom_low_lower_context_t* context;
  // Target contract environment provided by target-low legality.
  const loom_target_contract_query_environment_t* environment;
} loom_low_lower_contract_query_state_t;

static iree_status_t loom_low_lower_contract_query_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)match_context;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_low_lower_map_contract_value_callback_t map_contract_value =
      state->context->policy->map_contract_value;
  if (map_contract_value.fn == NULL) {
    return iree_ok_status();
  }
  return map_contract_value.fn(map_contract_value.user_data, state->environment,
                               source_op, source_value_id, out_mapped_value);
}

static bool loom_low_lower_contract_query_can_materialize(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id) {
  (void)match_context;
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const uint16_t materializer_index =
      (uint16_t)(value_ref->materializer_index - 1);
  const loom_low_lower_value_materializer_t* materializer =
      &rule_set->materializers[materializer_index];
  return materializer->can_materialize(state->context, source_value_id);
}

static iree_status_t loom_low_lower_query_target_contract_from_context(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_low_descriptor_set_t* saved_descriptor_set =
      context->descriptor_set;
  context->descriptor_set = environment->descriptor_set;
  loom_low_lower_contract_query_state_t state = {
      .context = context,
      .environment = environment,
  };
  const loom_low_lower_contract_query_options_t query_options = {
      .contract_index = &context->contract_index,
      .rule_sets = context->policy->rule_sets,
      .map_value =
          {
              .fn = loom_low_lower_contract_query_map_value,
              .user_data = &state,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_contract_query_can_materialize,
              .user_data = &state,
          },
      .contract_families = context->policy->contract_families,
      .contract_family_count = context->policy->contract_family_count,
  };

  iree_status_t status = loom_low_lower_query_target_contract(
      environment, &query_options, source_op, out_result);
  context->descriptor_set = saved_descriptor_set;
  return status;
}

static void loom_low_lower_record_report_row(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan,
    uint32_t emitted_low_op_count) {
  loom_low_lower_result_t* result = context->result;
  ++result->selected_source_op_count;
  result->emitted_low_op_count += emitted_low_op_count;
  ++result->report_row_total_count;
  if (result->report_rows == NULL ||
      result->report_row_count >= result->report_row_capacity) {
    return;
  }

  loom_low_lower_report_row_t* row =
      &result->report_rows[result->report_row_count++];
  *row = (loom_low_lower_report_row_t){
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name = loom_op_name(context->module, selected_plan->source_op),
      .source_op_kind = selected_plan->source_op->kind,
      .selection_kind = LOOM_LOW_LOWER_REPORT_SELECTION_PLAN,
      .rule_set_index = UINT16_MAX,
      .rule_index = UINT16_MAX,
      .plan_id = selected_plan->plan.id,
      .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
      .emitted_low_op_count = emitted_low_op_count,
  };
  if (selected_plan->rule != NULL) {
    row->selection_kind = LOOM_LOW_LOWER_REPORT_SELECTION_RULE;
    row->rule_set_index = selected_plan->rule_set_index;
    row->rule_index = selected_plan->rule_index;
    row->plan_id = LOOM_LOW_LOWER_PLAN_ID_NONE;
    row->descriptor_id = loom_low_lower_rule_first_descriptor_id(
        selected_plan->rule_set, selected_plan->rule);
  }
}

static iree_status_t loom_low_lower_plan_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  if (source_op->region_count != 0) {
    return loom_low_lower_emit_reject(
        context, source_op, IREE_SV("op"),
        loom_op_name(context->module, source_op),
        IREE_SV("nested regions must be lowered away before target-low "
                "source lowering"));
  }
  if (loom_low_lower_op_is_structural(context->module, source_op)) {
    return iree_ok_status();
  }
  if (loom_low_lower_op_is_source_metadata(source_op->kind)) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_rule_selection = {0};
  bool selected_rule = false;
  if (context->contract_index.case_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_op_from_contract_index(
        context, source_op, &failed_rule_set, &failed_rule_selection,
        &selected_rule));
    if (selected_rule) {
      return iree_ok_status();
    }
    if (failed_rule_set != NULL) {
      return loom_low_lower_rule_set_emit_selection_failure(
          context, failed_rule_set, source_op, failed_rule_selection);
    }
  }

  if (context->policy->select_op.fn != NULL) {
    loom_low_lower_plan_t plan = loom_low_lower_plan_empty();
    IREE_RETURN_IF_ERROR(context->policy->select_op.fn(
        context->policy->select_op.user_data, context, source_op, &plan));
    if (!loom_low_lower_plan_is_empty(plan)) {
      loom_low_lower_record_selected_plan(context,
                                          (loom_low_lower_selected_plan_t){
                                              .source_op = source_op,
                                              .rule_set_index = UINT16_MAX,
                                              .rule_index = UINT16_MAX,
                                              .rule_set = NULL,
                                              .rule = NULL,
                                              .plan = plan,
                                          });
      return iree_ok_status();
    }
  }

  if (failed_rule_set != NULL) {
    return loom_low_lower_rule_set_emit_selection_failure(
        context, failed_rule_set, source_op, failed_rule_selection);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("op"),
      loom_op_name(context->module, source_op),
      IREE_SV("the selected target-low lowering policy has no descriptor "
              "mapping for this op"));
}

static iree_status_t loom_low_lower_plan_body(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  IREE_RETURN_IF_ERROR(loom_low_lower_check_function_signature(context));
  if (loom_low_lower_context_should_stop(context)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_prepare_plan(context, source_body));

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_body, block_index);
    if (block_index != 0) {
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        loom_type_t low_type = loom_type_none();
        IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
            context, context->source_function.op, block->arg_ids[i],
            &low_type));
        if (loom_low_lower_context_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_low_lower_plan_op(context, op));
      if (loom_low_lower_context_should_stop(context)) {
        return iree_ok_status();
      }
    }
  }
  loom_low_lower_analyze_storage_demands(context, source_body);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_signature_types(
    loom_low_lower_context_t* context, loom_type_t** out_arg_types,
    iree_host_size_t* out_arg_count, loom_type_t** out_result_types,
    iree_host_size_t* out_result_count) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));
  *out_arg_types = NULL;
  *out_arg_count = 0;
  *out_result_types = NULL;
  *out_result_count = 0;

  uint16_t argument_count = 0;
  (void)loom_func_like_arg_ids(context->source_function, &argument_count);
  loom_type_t* arg_types = NULL;
  const uint16_t direct_argument_count =
      loom_low_lower_direct_argument_count(context);
  if (direct_argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&context->arena, direct_argument_count,
                                  sizeof(*arg_types), (void**)&arg_types));
    uint16_t direct_argument_index = 0;
    for (uint16_t i = 0; i < argument_count; ++i) {
      if (context->lowering.argument_map[i].kind !=
          LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
        continue;
      }
      arg_types[direct_argument_index] =
          context->lowering.argument_map[i].abi_type;
      IREE_ASSERT_FALSE(
          loom_low_lower_type_is_none(arg_types[direct_argument_index]));
      ++direct_argument_index;
    }
  }

  const uint16_t result_count = context->source_function.op->result_count;
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &context->arena, result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* result_ids =
        loom_op_const_results(context->source_function.op);
    for (uint16_t i = 0; i < result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_map_value(context, context->source_function.op,
                                   result_ids[i], &result_types[i]));
      IREE_ASSERT_FALSE(loom_low_lower_type_is_none(result_types[i]));
    }
  }

  *out_arg_types = arg_types;
  *out_arg_count = direct_argument_count;
  *out_result_types = result_types;
  *out_result_count = result_count;
  return iree_ok_status();
}

static loom_region_t* loom_low_lower_low_body(
    const loom_low_lower_context_t* context) {
  if (loom_low_func_def_isa(context->low_func_op)) {
    return loom_low_func_def_body(context->low_func_op);
  }
  if (loom_low_kernel_def_isa(context->low_func_op)) {
    return loom_low_kernel_def_body(context->low_func_op);
  }
  return NULL;
}

static bool loom_low_lower_source_is_kernel_entry(
    const loom_low_lower_context_t* context) {
  return loom_kernel_def_isa(context->source_function.op);
}

static iree_status_t loom_low_lower_create_func_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(context->source_function, &predicate_count);

  loom_low_func_def_build_flags_t build_flags = 0;
  uint8_t visibility = loom_func_like_visibility(context->source_function);
  uint8_t cc = loom_func_like_cc(context->source_function);
  uint8_t purity = loom_func_like_purity(context->source_function);
  loom_target_abi_kind_t abi = loom_low_lower_function_abi(context);
  loom_named_attr_slice_t abi_attrs =
      loom_func_like_abi_attrs(context->source_function);
  loom_string_id_t export_symbol =
      loom_func_like_export_symbol(context->source_function);
  loom_named_attr_slice_t export_attrs =
      loom_func_like_export_attrs(context->source_function);
  if (visibility != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
  }
  if (cc != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_CC;
  }
  if (purity != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_PURITY;
  }
  if (abi != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI;
  }
  if (export_symbol != LOOM_STRING_ID_INVALID) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }
  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module),
                          &context->builder);
  loom_builder_set_before(&context->builder, context->source_function.op);
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &context->builder, build_flags, visibility, cc, purity,
      /*allocation=*/0, /*schedule=*/0, context->options->target_ref, abi,
      abi_attrs, export_symbol, export_attrs, low_func_ref, arg_types,
      arg_count, result_types, result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, predicates, predicate_count,
      context->source_function.op->location, &context->low_func_op));

  loom_region_t* low_body = loom_low_lower_low_body(context);
  low_body->flags = source_body->flags;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_kernel_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref, const loom_type_t* arg_types,
    iree_host_size_t arg_count) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(context->source_function, &predicate_count);

  loom_low_kernel_def_build_flags_t build_flags = 0;
  loom_string_id_t export_symbol =
      loom_func_like_export_symbol(context->source_function);
  if (export_symbol != LOOM_STRING_ID_INVALID) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }

  loom_symbol_ref_t artifact =
      loom_func_like_artifact(context->source_function);
  if (loom_symbol_ref_is_valid(artifact)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_ARTIFACT;
  }

  int64_t export_ordinal = 0;
  if (loom_func_like_export_ordinal(context->source_function,
                                    &export_ordinal)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_ORDINAL;
  }

  uint8_t export_linkage = 0;
  if (loom_func_like_export_linkage(context->source_function,
                                    &export_linkage)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_LINKAGE;
  }

  uint32_t workgroup_size_x = 0;
  uint32_t workgroup_size_y = 0;
  uint32_t workgroup_size_z = 0;
  if (loom_func_like_workgroup_size(context->source_function, &workgroup_size_x,
                                    &workgroup_size_y, &workgroup_size_z)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_X |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Y |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Z;
  }

  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module),
                          &context->builder);
  loom_builder_set_before(&context->builder, context->source_function.op);
  IREE_RETURN_IF_ERROR(loom_low_kernel_def_build(
      &context->builder, build_flags, /*allocation=*/0, /*schedule=*/0,
      context->options->target_ref, export_symbol, artifact, export_ordinal,
      export_linkage, workgroup_size_x, workgroup_size_y, workgroup_size_z,
      low_func_ref, arg_types, arg_count, predicates, predicate_count,
      context->source_function.op->location, &context->low_func_op));

  loom_region_t* low_body = loom_low_lower_low_body(context);
  low_body->flags = source_body->flags;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_function_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref) {
  loom_type_t* arg_types = NULL;
  iree_host_size_t arg_count = 0;
  loom_type_t* result_types = NULL;
  iree_host_size_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_signature_types(
      context, &arg_types, &arg_count, &result_types, &result_count));

  if (loom_low_lower_source_is_kernel_entry(context)) {
    IREE_ASSERT_EQ(result_count, 0);
    IREE_RETURN_IF_ERROR(loom_low_lower_create_kernel_op(
        context, source_body, low_func_ref, arg_types, arg_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_lower_create_func_op(
        context, source_body, low_func_ref, arg_types, arg_count, result_types,
        result_count));
  }

  context->result->low_func_op = context->low_func_op;
  context->result->low_func_ref = low_func_ref;

  const loom_value_id_t* source_results =
      loom_op_const_results(context->source_function.op);
  const loom_value_id_t* low_results =
      loom_op_const_results(context->low_func_op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_blocks(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_low_body(context);
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->arena, source_body->block_count,
                                sizeof(*context->lowering.block_map),
                                (void**)&context->lowering.block_map));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, source_body->block_count,
      sizeof(*context->lowering.branch_dest_overrides),
      (void**)&context->lowering.branch_dest_overrides));
  memset(context->lowering.block_map, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.block_map));
  memset(context->lowering.branch_dest_overrides, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.branch_dest_overrides));

  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    loom_block_t* source_block = loom_region_block(source_body, i);
    loom_block_t* low_block = NULL;
    if (i == 0) {
      low_block = loom_region_entry_block(low_body);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(context->module, low_body, &low_block));
    }
    low_block->label_id = source_block->label_id;
    context->lowering.block_map[i] = low_block;
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_block_t* low_block = context->lowering.block_map[block_index];
    if (block_index == 0) {
      const uint16_t direct_argument_count =
          loom_low_lower_direct_argument_count(context);
      IREE_ASSERT_EQ(low_block->arg_count, direct_argument_count);
      uint16_t direct_argument_index = 0;
      for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
           ++arg_index) {
        if (context->lowering.argument_map[arg_index].kind !=
            LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, source_block->arg_ids[arg_index],
            low_block->arg_ids[direct_argument_index]));
        ++direct_argument_index;
      }
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
         ++arg_index) {
      loom_type_t low_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
          context, context->source_function.op,
          source_block->arg_ids[arg_index], &low_type));
      IREE_ASSERT_FALSE(loom_low_lower_type_is_none(low_type));
      loom_value_id_t low_arg = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder, low_block, low_type, &low_arg));
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_block->arg_ids[arg_index], low_arg));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_argument_resource_import(
    loom_low_lower_context_t* context, const loom_value_id_t* source_arguments,
    uint16_t argument_index) {
  const loom_low_lower_abi_argument_t* argument =
      &context->lowering.argument_map[argument_index];
  loom_type_t semantic_type = argument->resource_semantic_type;
  if (loom_low_lower_type_is_none(semantic_type)) {
    semantic_type = loom_module_value_type(context->module,
                                           source_arguments[argument_index]);
  }
  loom_type_id_t semantic_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_intern_type_id(context, semantic_type, &semantic_type_id));
  loom_op_t* resource_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resource_build(
      &context->builder, argument->resource_build_flags,
      (uint8_t)argument->resource_import_kind, argument->resource_index,
      semantic_type_id, argument->resource_valid_byte_count,
      argument->resource_cache_swizzle_stride, argument->abi_type,
      context->source_function.op->location, &resource_op));
  return loom_low_lower_bind_value(context, source_arguments[argument_index],
                                   loom_low_resource_result(resource_op));
}

static iree_status_t loom_low_lower_emit_argument_resource_imports(
    loom_low_lower_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  if (argument_count == 0 ||
      argument_count == loom_low_lower_direct_argument_count(context)) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    if (context->lowering.argument_map[i].kind !=
        LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      continue;
    }
    status = loom_low_lower_emit_argument_resource_import(context,
                                                          source_arguments, i);
  }

  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_preamble(
    loom_low_lower_context_t* context) {
  if (context->policy->emit_preamble.fn == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->emit_preamble.fn(
      context->policy->emit_preamble.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_remap_values(
    loom_low_lower_context_t* context, const loom_value_id_t* source_values,
    iree_host_size_t value_count, loom_value_id_t** out_low_values) {
  *out_low_values = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, value_count, sizeof(*low_values), (void**)&low_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_values[i], &low_values[i]));
  }
  *out_low_values = low_values;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_bind_identity_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_EQ(source_op->operand_count, source_op->result_count);
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value_alias(
        context, source_operands[i], source_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_structural_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  *out_handled = true;
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits)) {
    return loom_low_lower_bind_identity_results(context, source_op);
  }
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(source_op->operand_count >= 1);
    IREE_ASSERT(source_op->result_count == 1);
    return loom_low_lower_bind_value_alias(context,
                                           loom_op_const_operands(source_op)[0],
                                           loom_op_const_results(source_op)[0]);
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT: {
      return loom_low_lower_bind_value_alias(
          context, loom_buffer_assume_same_root_buffer(source_op),
          loom_buffer_assume_same_root_result(source_op));
    }
    case LOOM_OP_FUNC_RETURN: {
      loom_value_slice_t values = loom_func_return_operands(source_op);
      loom_value_id_t* low_values = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, values.values, values.count, &low_values));
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, low_values, values.count,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_KERNEL_RETURN: {
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, NULL, 0,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_CFG_BR: {
      loom_block_t* low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
          context, loom_cfg_br_dest(source_op), &low_dest));
      loom_region_t* source_body =
          loom_func_like_body(context->source_function);
      uint16_t source_index = 0;
      if (source_body &&
          loom_region_try_block_index(source_body, source_op->parent_block,
                                      &source_index) &&
          context->lowering.branch_dest_overrides[source_index] != NULL) {
        low_dest = context->lowering.branch_dest_overrides[source_index];
      }
      loom_value_slice_t args = loom_cfg_br_args(source_op);
      loom_value_id_t* low_args = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(context, args.values,
                                                       args.count, &low_args));
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(&context->builder, low_dest, low_args,
                               args.count, source_op->location, &low_br_op);
    }
    case LOOM_OP_CFG_COND_BR: {
      loom_block_t* low_true_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
          context, loom_cfg_cond_br_true_dest(source_op), &low_true_dest));
      loom_block_t* low_false_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
          context, loom_cfg_cond_br_false_dest(source_op), &low_false_dest));
      loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, loom_cfg_cond_br_condition(source_op), &low_condition));
      if (context->policy->emit_cond_branch.fn != NULL) {
        return context->policy->emit_cond_branch.fn(
            context->policy->emit_cond_branch.user_data, context, source_op,
            low_condition, low_true_dest, low_false_dest);
      }
      loom_op_t* low_cond_br_op = NULL;
      return loom_low_cond_br_build(&context->builder, low_condition,
                                    low_true_dest, low_false_dest,
                                    source_op->location, &low_cond_br_op);
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_low_lower_emit_elided_selected_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_elide_value(context, source_results[i]));
  }
  if (context->options->report_enabled) {
    loom_low_lower_record_report_row(context, selected_plan,
                                     /*emitted_low_op_count=*/0);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_selected_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_LT(context->lowering.selected_plan_emit_index,
                 context->lowering.selected_plan_count);
  const loom_low_lower_selected_plan_t selected_plan =
      context->lowering
          .selected_plans[context->lowering.selected_plan_emit_index++];
  IREE_ASSERT_EQ(selected_plan.source_op, source_op);
  if (iree_any_bit_set(selected_plan.flags,
                       LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED)) {
    return loom_low_lower_emit_elided_selected_plan(context, &selected_plan);
  }
  const bool report_enabled = context->options->report_enabled;
  loom_block_t* insertion_block = context->builder.ip.block;
  uint32_t before_op_count = 0;
  if (report_enabled) {
    IREE_ASSERT(insertion_block != NULL);
    before_op_count = insertion_block->op_count;
  }
  if (selected_plan.rule != NULL) {
    IREE_ASSERT(selected_plan.rule_set != NULL);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_emit_rule(
        context, selected_plan.rule_set, source_op, selected_plan.rule,
        selected_plan.resolved_emits));
  } else {
    IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(selected_plan.plan));
    IREE_ASSERT(context->policy->emit_op.fn != NULL);
    IREE_RETURN_IF_ERROR(
        context->policy->emit_op.fn(context->policy->emit_op.user_data, context,
                                    source_op, selected_plan.plan));
  }
  if (report_enabled) {
    const uint32_t after_op_count = insertion_block->op_count;
    IREE_ASSERT_GE(after_op_count, before_op_count);
    loom_low_lower_record_report_row(context, &selected_plan,
                                     after_op_count - before_op_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_body(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  iree_status_t status = iree_ok_status();

  for (uint16_t block_index = 0;
       block_index < source_body->block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_builder_set_block(&context->builder,
                           context->lowering.block_map[block_index]);
    loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      bool handled = false;
      status = loom_low_lower_structural_op(context, source_op, &handled);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (!handled) {
        if (loom_low_lower_op_is_source_metadata(source_op->kind)) {
          continue;
        }
        status = loom_low_lower_emit_selected_plan(context, source_op);
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
    }
  }

  loom_builder_restore(&context->builder, saved_ip);
  IREE_ASSERT_EQ(context->lowering.selected_plan_emit_index,
                 context->lowering.selected_plan_count);
  return status;
}

iree_status_t loom_low_lower_function(loom_module_t* module,
                                      loom_func_like_t source_function,
                                      const loom_low_lower_options_t* options,
                                      loom_low_lower_result_t* out_result) {
  IREE_ASSERT(out_result != NULL);
  loom_low_lower_assert_options(module, source_function, options);
  *out_result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };

  loom_region_t* source_body = loom_func_like_body(source_function);
  IREE_ASSERT(source_body != NULL);

  loom_low_lower_context_t context = {
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .result = out_result,
  };
  context.lowering.fact_table = options->fact_table;
  if (options->report_enabled) {
    out_result->report_rows = options->report_storage.rows;
    out_result->report_row_capacity = options->report_storage.row_capacity;
  }
  iree_arena_initialize(module->arena.block_pool, &context.arena);

  iree_status_t status =
      loom_low_lowering_frame_initialize_value_ordinals(&context, source_body);
  if (iree_status_is_ok(status)) {
    status = loom_target_contract_index_compose(
        context.policy->contract_bindings,
        context.policy->contract_binding_count, &context.contract_index,
        &context.arena);
  }

  loom_vector_memory_footprint_result_t footprint_result = {0};
  if (iree_status_is_ok(status)) {
    const loom_vector_memory_footprint_options_t footprint_options = {
        .arena = &context.arena,
        .fact_table = context.lowering.fact_table,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_vector_memory_footprint_verify_function(
        module, source_function, &footprint_options, &footprint_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += footprint_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.arena);
    return iree_ok_status();
  }

  loom_kernel_async_legality_result_t async_legality_result = {0};
  if (iree_status_is_ok(status)) {
    loom_kernel_async_legality_options_t async_legality_options = {
        .arena = &context.arena,
        .fact_table = context.lowering.fact_table,
        .value_domain = &context.lowering.value_domain,
        .emitter = options->emitter,
        .phase_name = IREE_SV("source-low"),
    };
    status = loom_kernel_async_legality_verify_function(module, source_function,
                                                        &async_legality_options,
                                                        &async_legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += async_legality_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.arena);
    return iree_ok_status();
  }

  loom_target_low_legality_result_t legality_result = {};
  if (iree_status_is_ok(status)) {
    loom_target_low_legality_options_t legality_options = {
        .bundle = options->bundle,
        .descriptor_registry = options->descriptor_registry,
        .provider_list = options->legality_provider_list,
        .contract_query =
            {
                .fn = loom_low_lower_query_target_contract_from_context,
                .user_data = &context,
            },
        .fact_table = context.lowering.fact_table,
        .diagnostic_flags = options->legality_diagnostic_flags,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_target_low_verify_function_legality(
        module, source_function, &legality_options, &legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count = legality_result.error_count;
    out_result->remark_count = legality_result.remark_count;
    out_result->descriptor_set = legality_result.descriptor_set;
    context.descriptor_set = legality_result.descriptor_set;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.arena);
    return iree_ok_status();
  }

  if (iree_status_is_ok(status) &&
      context.lowering.value_domain.value_count != 0) {
    status = iree_arena_allocate_array(
        &context.arena, context.lowering.value_domain.value_count,
        sizeof(*context.lowering.value_map),
        (void**)&context.lowering.value_map);
  }
  if (iree_status_is_ok(status) &&
      context.lowering.value_domain.value_count != 0) {
    status = iree_arena_allocate_array(
        &context.arena, context.lowering.value_domain.value_count,
        sizeof(*context.lowering.value_storage_flags),
        (void**)&context.lowering.value_storage_flags);
  }
  if (iree_status_is_ok(status)) {
    for (loom_value_ordinal_t i = 0;
         i < context.lowering.value_domain.value_count; ++i) {
      context.lowering.value_map[i] = LOOM_VALUE_ID_INVALID;
      context.lowering.value_storage_flags[i] = 0;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_plan_body(&context, source_body);
  }
  if (iree_status_is_ok(status) && context.result->error_count == 0) {
    loom_symbol_ref_t low_func_ref = loom_func_like_callee(source_function);
    status =
        loom_low_lower_create_function_op(&context, source_body, low_func_ref);
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_map_blocks(&context, source_body);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_emit_preamble(&context);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_emit_argument_resource_imports(&context);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_emit_body(&context, source_body);
    }
    if (iree_status_is_ok(status) && context.result->error_count != 0) {
      status = loom_op_erase(module, context.low_func_op);
      context.low_func_op = NULL;
      out_result->low_func_op = NULL;
      out_result->low_func_ref = loom_symbol_ref_null();
    }
    // The replacement low op carries the source symbol while the source op
    // still owns the symbol table entry. Erase clears that entry; relink it to
    // the replacement so callers keep the same symbol identity.
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_op_erase(module, source_function.op);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_module_link_symbol_defining_op(
          module, context.low_func_op,
          loom_op_vtable(module, context.low_func_op));
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0 &&
        context.lowering.memory_access_record_count != 0) {
      out_result->memory_access_table = (loom_low_memory_access_table_t){
          .function_op = context.low_func_op,
          .values = context.lowering.memory_access_records,
          .count = context.lowering.memory_access_record_count,
      };
    }
  }

  loom_low_lowering_frame_deinitialize(&context);
  iree_arena_deinitialize(&context.arena);
  return status;
}
