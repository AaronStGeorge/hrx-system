// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel_async_legality.h"

#include <inttypes.h>
#include <stdio.h>

#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_KERNEL_ASYNC_LEGALITY_STAT_BLOCKS_CHECKED = 0,
  LOOM_KERNEL_ASYNC_LEGALITY_STAT_GROUPS_CHECKED = 1,
  LOOM_KERNEL_ASYNC_LEGALITY_STAT_WAITS_CHECKED = 2,
};

static const loom_pass_statistic_def_t kKernelAsyncLegalityStatistics[] = {
    {IREE_SVL("blocks-checked"),
     IREE_SVL("Number of blocks checked for async stream legality.")},
    {IREE_SVL("groups-checked"),
     IREE_SVL("Number of kernel.async.group ops checked.")},
    {IREE_SVL("waits-checked"),
     IREE_SVL("Number of kernel.async.wait ops checked.")},
};

static const loom_pass_info_t loom_kernel_async_legality_pass_info_storage = {
    .name = IREE_SVL("kernel-async-legality"),
    .description =
        IREE_SVL("Prove kernel async group/wait streams are lowerable."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kKernelAsyncLegalityStatistics,
    .statistic_count = IREE_ARRAYSIZE(kKernelAsyncLegalityStatistics),
};

const loom_pass_info_t* loom_kernel_async_legality_pass_info(void) {
  return &loom_kernel_async_legality_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// State and helpers
//===----------------------------------------------------------------------===//

typedef struct loom_kernel_async_legality_state_t {
  // Current pass instance used for diagnostics, statistics, and scratch arena
  // ownership.
  loom_pass_t* pass;

  // Module whose function body is being checked.
  loom_module_t* module;
} loom_kernel_async_legality_state_t;

typedef struct loom_kernel_async_legality_group_t {
  // SSA value produced by kernel.async.group.
  loom_value_id_t group_id;

  // Op that committed the group into the current straight-line stream.
  const loom_op_t* group_op;

  // True once a wait has completed this group directly or indirectly.
  bool completed;
} loom_kernel_async_legality_group_t;

typedef struct loom_kernel_async_legality_stream_t {
  // Groups committed in the current block, in program order.
  loom_kernel_async_legality_group_t* groups;

  // Number of committed groups in groups.
  iree_host_size_t count;
} loom_kernel_async_legality_stream_t;

#define LOOM_KERNEL_ASYNC_LEGALITY_INITIAL_REGION_CAPACITY 16

typedef struct loom_kernel_async_legality_region_worklist_t {
  // Region pointers waiting to be checked.
  loom_region_t** regions;

  // Index of the next queued region to check.
  iree_host_size_t next_index;

  // Number of queued region pointers.
  iree_host_size_t count;

  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_kernel_async_legality_region_worklist_t;

static iree_status_t loom_kernel_async_legality_region_worklist_initialize(
    iree_arena_allocator_t* arena,
    loom_kernel_async_legality_region_worklist_t* worklist) {
  worklist->next_index = 0;
  worklist->count = 0;
  worklist->capacity = LOOM_KERNEL_ASYNC_LEGALITY_INITIAL_REGION_CAPACITY;
  return iree_arena_allocate_array(arena, worklist->capacity,
                                   sizeof(loom_region_t*),
                                   (void**)&worklist->regions);
}

static iree_status_t loom_kernel_async_legality_region_worklist_push(
    iree_arena_allocator_t* arena,
    loom_kernel_async_legality_region_worklist_t* worklist,
    loom_region_t* region) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (worklist->count >= worklist->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, worklist->count, worklist->count + 1, sizeof(loom_region_t*),
        &worklist->capacity, (void**)&worklist->regions));
  }
  worklist->regions[worklist->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_kernel_async_legality_region_worklist_pop(
    loom_kernel_async_legality_region_worklist_t* worklist) {
  if (worklist->next_index >= worklist->count) return NULL;
  return worklist->regions[worklist->next_index++];
}

static void loom_kernel_async_legality_add_stat(
    loom_kernel_async_legality_state_t* state, uint16_t statistic_index,
    int64_t delta) {
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, statistic_index, delta);
  }
}

static iree_string_view_t loom_kernel_async_legality_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) return IREE_SV("<unknown>");
  return loom_op_vtable_name(vtable);
}

static iree_status_t loom_kernel_async_legality_fail(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    iree_string_view_t reason) {
  iree_string_view_t op_name =
      loom_kernel_async_legality_op_name(state->module, op);
  iree_string_view_t pass_name = state->pass->info->name;
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_string(pass_name),
      loom_param_string(reason),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = &loom_err_lowering_001,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission));
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "%.*s cannot be lowered by %.*s: %.*s",
                          (int)op_name.size, op_name.data, (int)pass_name.size,
                          pass_name.data, (int)reason.size, reason.data);
}

static bool loom_kernel_async_legality_use_is_wait(const loom_use_t* use) {
  return loom_kernel_async_wait_isa(loom_use_user_op(*use));
}

static iree_status_t loom_kernel_async_legality_check_group_uses(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    loom_value_id_t group_id) {
  if (group_id >= state->module->values.count) {
    return loom_kernel_async_legality_fail(
        state, op, IREE_SV("async group result value is out of range"));
  }
  const loom_value_t* group_value = loom_module_value(state->module, group_id);
  if (group_value->use_count == 0) {
    return loom_kernel_async_legality_fail(
        state, op,
        IREE_SV("async group has no wait in the current async stream"));
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(group_value, use) {
    if (!loom_kernel_async_legality_use_is_wait(use)) {
      return loom_kernel_async_legality_fail(
          state, op,
          IREE_SV("carried async groups require pipeline-aware legality "
                  "analysis before lowering"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_append_group(
    loom_kernel_async_legality_state_t* state,
    loom_kernel_async_legality_stream_t* stream, const loom_op_t* op) {
  if (op->result_count == 0) {
    return loom_kernel_async_legality_fail(
        state, op, IREE_SV("kernel.async.group is missing its group result"));
  }
  loom_value_id_t group_id = loom_kernel_async_group_group(op);
  IREE_RETURN_IF_ERROR(
      loom_kernel_async_legality_check_group_uses(state, op, group_id));
  stream->groups[stream->count++] = (loom_kernel_async_legality_group_t){
      .group_id = group_id,
      .group_op = op,
      .completed = false,
  };
  loom_kernel_async_legality_add_stat(
      state, LOOM_KERNEL_ASYNC_LEGALITY_STAT_GROUPS_CHECKED, 1);
  return iree_ok_status();
}

static iree_host_size_t loom_kernel_async_legality_find_group(
    const loom_kernel_async_legality_stream_t* stream,
    loom_value_id_t group_id) {
  for (iree_host_size_t i = 0; i < stream->count; ++i) {
    if (stream->groups[i].group_id == group_id) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_host_size_t loom_kernel_async_legality_newer_uncompleted_count(
    const loom_kernel_async_legality_stream_t* stream,
    iree_host_size_t group_index) {
  iree_host_size_t newer_groups = 0;
  for (iree_host_size_t i = group_index + 1; i < stream->count; ++i) {
    if (!stream->groups[i].completed) ++newer_groups;
  }
  return newer_groups;
}

static void loom_kernel_async_legality_complete_through(
    loom_kernel_async_legality_stream_t* stream, iree_host_size_t group_index) {
  for (iree_host_size_t i = 0; i <= group_index; ++i) {
    stream->groups[i].completed = true;
  }
}

static iree_status_t loom_kernel_async_legality_fail_newer_groups(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    int64_t actual_newer_groups, iree_host_size_t expected_newer_groups) {
  char reason_buffer[160];
  int reason_length = snprintf(reason_buffer, IREE_ARRAYSIZE(reason_buffer),
                               "newer_groups is %" PRId64 " but %" PRIhsz
                               " younger async group(s) remain outstanding",
                               actual_newer_groups, expected_newer_groups);
  if (reason_length <= 0 ||
      (iree_host_size_t)reason_length >= IREE_ARRAYSIZE(reason_buffer)) {
    return loom_kernel_async_legality_fail(
        state, op,
        IREE_SV("newer_groups does not match the async stream depth"));
  }
  return loom_kernel_async_legality_fail(
      state, op,
      iree_make_string_view(reason_buffer, (iree_host_size_t)reason_length));
}

static iree_status_t loom_kernel_async_legality_check_wait(
    loom_kernel_async_legality_state_t* state,
    loom_kernel_async_legality_stream_t* stream, const loom_op_t* op) {
  if (op->operand_count == 0) {
    return loom_kernel_async_legality_fail(
        state, op, IREE_SV("kernel.async.wait is missing its group operand"));
  }
  if (op->attribute_count == 0) {
    return loom_kernel_async_legality_fail(
        state, op, IREE_SV("kernel.async.wait is missing newer_groups"));
  }

  loom_value_id_t group_id = loom_kernel_async_wait_group(op);
  iree_host_size_t group_index =
      loom_kernel_async_legality_find_group(stream, group_id);
  if (group_index == IREE_HOST_SIZE_MAX) {
    return loom_kernel_async_legality_fail(
        state, op,
        IREE_SV("waited async group is not committed in the current "
                "straight-line async stream"));
  }
  if (stream->groups[group_index].completed) {
    return loom_kernel_async_legality_fail(
        state, op,
        IREE_SV("async group was already completed by an earlier "
                "wait in the current stream"));
  }

  int64_t actual_newer_groups = loom_kernel_async_wait_newer_groups(op);
  if (actual_newer_groups < 0) {
    return loom_kernel_async_legality_fail(
        state, op, IREE_SV("newer_groups must be nonnegative"));
  }
  iree_host_size_t expected_newer_groups =
      loom_kernel_async_legality_newer_uncompleted_count(stream, group_index);
  if ((uint64_t)actual_newer_groups != (uint64_t)expected_newer_groups) {
    return loom_kernel_async_legality_fail_newer_groups(
        state, op, actual_newer_groups, expected_newer_groups);
  }

  loom_kernel_async_legality_complete_through(stream, group_index);
  loom_kernel_async_legality_add_stat(
      state, LOOM_KERNEL_ASYNC_LEGALITY_STAT_WAITS_CHECKED, 1);
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_check_uncompleted_groups(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream) {
  for (iree_host_size_t i = 0; i < stream->count; ++i) {
    if (stream->groups[i].completed) continue;
    return loom_kernel_async_legality_fail(
        state, stream->groups[i].group_op,
        IREE_SV("async group is not waited before leaving its block"));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_check_block(
    loom_kernel_async_legality_state_t* state, loom_block_t* block) {
  if (block->op_count == 0) return iree_ok_status();

  iree_host_size_t group_capacity = 0;
  bool has_wait = false;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_kernel_async_group_isa(op)) {
      ++group_capacity;
    } else if (loom_kernel_async_wait_isa(op)) {
      has_wait = true;
    }
  }
  if (group_capacity == 0 && !has_wait) return iree_ok_status();

  loom_kernel_async_legality_stream_t stream = {0};
  if (group_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, group_capacity,
        sizeof(loom_kernel_async_legality_group_t), (void**)&stream.groups));
  }

  loom_block_for_each_op(block, op) {
    if (loom_kernel_async_group_isa(op)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_async_legality_append_group(state, &stream, op));
    } else if (loom_kernel_async_wait_isa(op)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_async_legality_check_wait(state, &stream, op));
    }
  }

  loom_kernel_async_legality_add_stat(
      state, LOOM_KERNEL_ASYNC_LEGALITY_STAT_BLOCKS_CHECKED, 1);
  return loom_kernel_async_legality_check_uncompleted_groups(state, &stream);
}

static iree_status_t loom_kernel_async_legality_check_regions(
    loom_kernel_async_legality_state_t* state, loom_region_t* root_region) {
  loom_kernel_async_legality_region_worklist_t worklist;
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_initialize(
      state->pass->arena, &worklist));
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_push(
      state->pass->arena, &worklist, root_region));

  loom_region_t* region = NULL;
  while ((region = loom_kernel_async_legality_region_worklist_pop(&worklist))) {
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_async_legality_check_block(state, block));
    }

    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_push(
              state->pass->arena, &worklist, regions[i]));
        }
      }
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_kernel_async_legality_run(loom_pass_t* pass,
                                             loom_module_t* module,
                                             loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  loom_kernel_async_legality_state_t state = {
      .pass = pass,
      .module = module,
  };
  return loom_kernel_async_legality_check_regions(&state, body);
}
