// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/cleanup.h"

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/types.h"
#include "loom/passes/dce.h"

typedef struct loom_low_cleanup_deadness_context_t {
  // Resolved target selected by the low function target bundle.
  const loom_low_resolved_target_t* target;
} loom_low_cleanup_deadness_context_t;

static iree_status_t loom_low_cleanup_deadness_query(
    void* user_data, const loom_module_t* module, const loom_op_t* op,
    bool* out_is_dead) {
  loom_low_cleanup_deadness_context_t* context =
      (loom_low_cleanup_deadness_context_t*)user_data;
  *out_is_dead = false;

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_descriptor_packet(module, context->target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    *out_is_dead = loom_op_is_trivially_dead(module, op);
    return iree_ok_status();
  }

  if (op->result_count == 0 || !loom_op_results_unused(module, op)) {
    return iree_ok_status();
  }

  if (packet.descriptor == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "failed to look up low descriptor '%.*s'",
                            (int)packet.key.size, packet.key.data);
  }
  *out_is_dead = iree_any_bit_set(packet.descriptor->flags,
                                  LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE);
  return iree_ok_status();
}

iree_status_t loom_low_cleanup_function(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter) {
  if (!module || !low_func_op || !descriptor_registry) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, low function op, and descriptor registry are required");
  }
  if (!loom_low_func_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def");
  }

  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, descriptor_registry, emitter, &target));
  if (!target.descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function target did not resolve");
  }

  loom_func_like_t function = loom_func_like_cast(module, low_func_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def to be function-like");
  }

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(module->arena.block_pool, &pass_arena);
  loom_pass_t pass = {
      .info = loom_dce_pass_info(),
      .arena = &pass_arena,
  };
  loom_low_cleanup_deadness_context_t deadness_context = {
      .target = &target,
  };
  const loom_dce_deadness_query_callback_t deadness_query = {
      .fn = loom_low_cleanup_deadness_query,
      .user_data = &deadness_context,
  };
  iree_status_t status =
      loom_dce_run_with_deadness_query(&pass, module, function, deadness_query);
  iree_arena_deinitialize(&pass_arena);
  return status;
}
