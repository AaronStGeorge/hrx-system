// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/kernel/collective.h"

#include "loom/ops/kernel/ops.h"

loom_value_fact_uniform_scope_t loom_kernel_collective_scope(
    const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_KERNEL_SUBGROUP_SHUFFLE:
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST:
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST:
    case LOOM_OP_KERNEL_SUBGROUP_REDUCE:
    case LOOM_OP_KERNEL_SUBGROUP_SCAN:
    case LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY:
    case LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL:
    case LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT:
    case LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK:
    case LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY:
    case LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL:
      return LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP;
    case LOOM_OP_KERNEL_WORKGROUP_REDUCE:
    case LOOM_OP_KERNEL_WORKGROUP_SCAN:
    case LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY:
    case LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL:
    case LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT:
      return LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP;
    case LOOM_OP_KERNEL_BARRIER:
      return loom_kernel_barrier_scope(op) == LOOM_ATOMIC_SCOPE_SUBGROUP
                 ? LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP
                 : LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP;
    case LOOM_OP_KERNEL_BARRIER_ARRIVE:
      return loom_kernel_barrier_arrive_scope(op) == LOOM_ATOMIC_SCOPE_SUBGROUP
                 ? LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP
                 : LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP;
    default:
      return LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE;
  }
}
