// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/source_workload_pipeline.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/verify/verify.h"

static iree_status_t loom_low_source_workload_verify_source_module(
    const loom_module_t* module) {
  loom_verify_options_t options = {0};
  loom_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_verify_module(module, &options, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated source failed verification");
  }
  return iree_ok_status();
}

static void loom_low_source_workload_count_module_source_ops(
    const loom_module_t* module,
    loom_low_source_workload_counts_t* out_counts) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_counts);
  memset(out_counts, 0, sizeof(*out_counts));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* op = module->symbols.entries[i].defining_op;
    if (!loom_func_def_isa(op)) {
      continue;
    }
    loom_low_source_workload_counts_t func_counts;
    loom_low_source_workload_count_func_ops(module, op, &func_counts);
    loom_low_source_workload_counts_accumulate(out_counts, &func_counts);
  }
}

static iree_status_t loom_low_source_workload_verify_low_module(
    const loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options) {
  const loom_low_verify_options_t verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = options->descriptor_registry,
      .descriptor_requirements = options->descriptor_requirements,
      .max_errors = 20,
  };
  loom_low_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &verify_options, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated low function failed verification");
  }
  return iree_ok_status();
}

static uint32_t loom_low_source_workload_count_low_descriptor_ops(
    const loom_op_t* low_func_op) {
  uint32_t count = 0;
  const loom_region_t* body = loom_low_func_def_body(low_func_op);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_op_isa(op) || loom_low_const_isa(op)) {
        ++count;
      }
    }
  }
  return count;
}

iree_status_t loom_low_source_workload_run_pipeline(
    loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_low_source_workload_pipeline_counters_t* out_counters) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->descriptor_registry);
  IREE_ASSERT_ARGUMENT(options->policy_registry);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_counters);
  memset(out_counters, 0, sizeof(*out_counters));

  loom_low_source_workload_count_module_source_ops(
      module, &out_counters->source_counts);
  iree_status_t status = loom_low_source_workload_verify_source_module(module);

  iree_arena_allocator_t lowering_arena;
  bool lowering_arena_initialized = false;
  loom_low_lower_result_t lower_result = {0};
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &lowering_arena);
    lowering_arena_initialized = true;
    const loom_low_source_selection_options_t selection_options = {
        .descriptor_registry = options->descriptor_registry,
        .policy_registry = options->policy_registry,
    };
    loom_low_source_selection_list_t selection_list = {0};
    status = loom_low_select_source_funcs(module, &selection_options,
                                          &lowering_arena, &selection_list);
    loom_op_t** lowered_funcs = NULL;
    if (iree_status_is_ok(status) && selection_list.count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "generated workload has no compatible source functions");
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(&lowering_arena, selection_list.count,
                                         sizeof(*lowered_funcs),
                                         (void**)&lowered_funcs);
    }
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      const loom_low_source_selection_t* selection = &selection_list.values[i];
      const loom_low_lower_options_t lower_options = {
          .target_ref = selection->target_ref,
          .bundle = selection->target_bundle,
          .descriptor_registry = options->descriptor_registry,
          .descriptor_requirements = options->descriptor_requirements,
          .policy = selection->policy,
          .max_errors = 20,
      };
      loom_low_lower_result_t func_lower_result = {0};
      status = loom_low_lower_function(module, selection->func, &lower_options,
                                       &func_lower_result);
      lower_result.error_count += func_lower_result.error_count;
      lower_result.remark_count += func_lower_result.remark_count;
      if (iree_status_is_ok(status) && func_lower_result.error_count != 0) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "source lowering produced errors");
      } else if (iree_status_is_ok(status) &&
                 func_lower_result.low_func_op == NULL) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "source lowering did not emit low.func.def");
      }
      if (iree_status_is_ok(status)) {
        lowered_funcs[i] = func_lower_result.low_func_op;
      }
    }
    if (iree_status_is_ok(status)) {
      out_counters->lower_error_count = lower_result.error_count;
      out_counters->lower_remark_count = lower_result.remark_count;
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_source_workload_verify_low_module(module, options);
    }
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      out_counters->low_descriptor_op_count +=
          loom_low_source_workload_count_low_descriptor_ops(lowered_funcs[i]);
    }

    iree_arena_allocator_t packet_arena;
    bool packet_arena_initialized = false;
    if (iree_status_is_ok(status)) {
      iree_arena_initialize(block_pool, &packet_arena);
      packet_arena_initialized = true;
    }
    const loom_low_packetization_options_t packet_options = {
        .descriptor_registry = options->descriptor_registry,
        .schedule_strategy = options->schedule_strategy,
    };
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      loom_low_packetization_t packetization = {0};
      status =
          loom_low_packetize_function(module, lowered_funcs[i], &packet_options,
                                      &packet_arena, &packetization);
      if (iree_status_is_ok(status)) {
        out_counters->schedule_node_count +=
            packetization.schedule.scheduled_node_count;
        out_counters->schedule_dependency_count +=
            packetization.schedule.dependency_count;
        out_counters->schedule_resource_use_count +=
            packetization.schedule.resource_use_count;
        out_counters->schedule_hazard_gap_count +=
            packetization.schedule.hazard_gap_count;
        out_counters->allocation_assignment_count +=
            packetization.allocation.assignment_count;
        out_counters->allocation_spill_count +=
            packetization.allocation.spill_count;
        out_counters->allocation_coalesced_copy_count +=
            packetization.allocation.coalesced_copy_count;
        out_counters->allocation_materialized_copy_count +=
            packetization.allocation.materialized_copy_count;
      }
    }
    if (packet_arena_initialized) {
      iree_arena_deinitialize(&packet_arena);
    }
  }

  if (lowering_arena_initialized) {
    iree_arena_deinitialize(&lowering_arena);
  }
  return status;
}
