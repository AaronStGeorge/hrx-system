// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/ownership/ownership_lifetime.h"

#include "loom/analysis/ownership_lifetime.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_OWNERSHIP_LIFETIME_STAT_BLOCKS_CHECKED = 0,
  LOOM_OWNERSHIP_LIFETIME_STAT_OPS_CHECKED = 1,
  LOOM_OWNERSHIP_LIFETIME_STAT_EFFECTS_CHECKED = 2,
};

static const loom_pass_statistic_def_t kOwnershipLifetimeStatistics[] = {
    {IREE_SVL("blocks-checked"),
     IREE_SVL("Number of blocks checked for ownership lifetimes.")},
    {IREE_SVL("ops-checked"),
     IREE_SVL("Number of operations visited for ownership lifetimes.")},
    {IREE_SVL("effects-checked"),
     IREE_SVL("Number of ownership effects interpreted.")},
};

static const loom_pass_info_t loom_ownership_lifetime_pass_info_storage = {
    .name = IREE_SVL("ownership-lifetime"),
    .description = IREE_SVL("Analyze descriptor-backed owned-resource "
                            "lifetimes across function and CFG control flow."),
    .kind = LOOM_PASS_MODULE,
    .statistic_defs = kOwnershipLifetimeStatistics,
    .statistic_count = IREE_ARRAYSIZE(kOwnershipLifetimeStatistics),
};

const loom_pass_info_t* loom_ownership_lifetime_pass_info(void) {
  return &loom_ownership_lifetime_pass_info_storage;
}

static void loom_ownership_lifetime_add_stat(loom_pass_t* pass,
                                             uint16_t statistic_index,
                                             uint64_t value) {
  if (pass->statistics && value != 0) {
    loom_pass_statistic_add(pass, statistic_index, (int64_t)value);
  }
}

iree_status_t loom_ownership_lifetime_run(loom_pass_t* pass,
                                          loom_module_t* module) {
  loom_ownership_lifetime_options_t options = {
      .arena = pass->arena,
      .emitter = pass->diagnostic_emitter,
      .phase_name = pass->info->name,
  };
  loom_ownership_lifetime_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_analyze_module(module, &options, &result));

  loom_ownership_lifetime_add_stat(
      pass, LOOM_OWNERSHIP_LIFETIME_STAT_BLOCKS_CHECKED, result.blocks_checked);
  loom_ownership_lifetime_add_stat(
      pass, LOOM_OWNERSHIP_LIFETIME_STAT_OPS_CHECKED, result.ops_checked);
  loom_ownership_lifetime_add_stat(pass,
                                   LOOM_OWNERSHIP_LIFETIME_STAT_EFFECTS_CHECKED,
                                   result.effects_checked);
  return iree_ok_status();
}
