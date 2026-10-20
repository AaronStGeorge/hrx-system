// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_BOUNDARY_PROJECTION_BRANCH_H_
#define LOOM_TRANSFORMS_BOUNDARY_PROJECTION_BRANCH_H_

#include "loom/transforms/boundary/projection_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_boundary_projection_branch_t {
  // Original structured branch whose regions retain their operations.
  loom_region_branch_t branch;
  // Original terminators in physical region order.
  loom_op_t** terminators;
  // Result-column candidate indices, or IREE_HOST_SIZE_MAX when unclaimed.
  iree_host_size_t* candidates;
  // Incoming recipes in region-major, original-result order.
  loom_boundary_projection_source_t* sources;
  // Prefix offsets from original results to their physical components.
  uint16_t* offsets;
  // Preallocated physical results, shared by every incoming yield.
  loom_value_id_t* result_ids;
  // Final result arity after projection.
  uint16_t result_count;
  // Whether at least one result column remains projected.
  bool selected;
};

// Resolves retained branch result identities after candidate sorting.
void loom_boundary_projection_index_branches(
    loom_boundary_projection_function_t* function);

// Selects every branch result schema before structured source planning.
iree_status_t loom_boundary_projection_select_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function);

// Retains RegionBranch results and their yield edges in dominance preorder.
iree_status_t loom_boundary_projection_collect_branch(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_region_branch_t branch);

// Retains every incoming projection recipe after all structured schemas exist.
iree_status_t loom_boundary_projection_plan_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function);

// Allocates physical result identities after connected rejection propagation.
// All identities precede application so enclosing loops and branches can
// transport a nested branch's components without rediscovering its yields.
iree_status_t loom_boundary_projection_preallocate_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function);

// Rebuilds retained branches and all yields with the selected result schema.
iree_status_t loom_boundary_projection_apply_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BOUNDARY_PROJECTION_BRANCH_H_
