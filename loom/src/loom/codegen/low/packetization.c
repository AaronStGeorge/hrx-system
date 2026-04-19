// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packetization.h"

#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"

iree_status_t loom_low_packetize_function(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_packetization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_packetization_t* out_packetization) {
  if (!module || !low_func_op || !options || !options->descriptor_registry ||
      !arena || !out_packetization) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, low function op, options with descriptor registry, arena, and "
        "output packetization are required");
  }
  if (options->allocation_budget_count > 0 && !options->allocation_budgets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation budgets are required when allocation_budget_count is "
        "non-zero");
  }
  if (!loom_low_func_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def");
  }

  *out_packetization = (loom_low_packetization_t){0};
  loom_low_schedule_options_t schedule_options = {
      .descriptor_registry = options->descriptor_registry,
      .emitter = options->emitter,
      .diagnostic_flags = options->schedule_diagnostic_flags,
      .strategy = options->schedule_strategy,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_function(module, low_func_op, &schedule_options, arena,
                                 &out_packetization->schedule));

  loom_low_allocation_options_t allocation_options = {
      .descriptor_registry = options->descriptor_registry,
      .budgets = options->allocation_budgets,
      .budget_count = options->allocation_budget_count,
      .emitter = options->emitter,
      .diagnostic_flags = options->allocation_diagnostic_flags,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocate_function(module, low_func_op, &allocation_options,
                                 arena, &out_packetization->allocation));

  return loom_low_packet_validate_sidecars(&out_packetization->schedule,
                                           &out_packetization->allocation);
}
