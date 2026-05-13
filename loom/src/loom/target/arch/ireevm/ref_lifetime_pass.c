// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/ref_lifetime_pass.h"

#include "loom/analysis/ownership_lifetime.h"
#include "loom/ir/module.h"
#include "loom/target/arch/ireevm/ops/ops.h"

typedef struct loom_ireevm_ref_lifetime_policy_t {
  // Interned type name for ireevm.ref.
  loom_string_id_t ref_type_name_id;
} loom_ireevm_ref_lifetime_policy_t;

static bool loom_ireevm_ref_lifetime_type_matches(loom_type_t type,
                                                  void* user_data) {
  const loom_ireevm_ref_lifetime_policy_t* policy =
      (const loom_ireevm_ref_lifetime_policy_t*)user_data;
  return loom_type_is_dialect(type) &&
         loom_type_dialect_name_id(type) == policy->ref_type_name_id &&
         loom_type_dialect_param_count(type) == 1;
}

static iree_status_t loom_ireevm_ref_lifetime_build_release(
    loom_builder_t* builder, loom_value_id_t value_id,
    loom_location_id_t location, void* user_data, loom_op_t** out_op) {
  (void)user_data;
  return loom_ireevm_ref_release_build(builder, value_id, location, out_op);
}

enum {
  LOOM_IREEVM_REF_LIFETIME_STAT_BLOCKS_CHECKED = 0,
  LOOM_IREEVM_REF_LIFETIME_STAT_OPS_CHECKED = 1,
  LOOM_IREEVM_REF_LIFETIME_STAT_EFFECTS_CHECKED = 2,
  LOOM_IREEVM_REF_LIFETIME_STAT_RELEASES_INSERTED = 3,
  LOOM_IREEVM_REF_LIFETIME_STAT_EDGES_SPLIT = 4,
};

static const loom_pass_statistic_def_t kIreeVmRefLifetimeStatisticDefs[] = {
    {IREE_SVL("blocks-checked"),
     IREE_SVL("Number of blocks checked for IREE VM ref lifetimes.")},
    {IREE_SVL("ops-checked"),
     IREE_SVL("Number of operations visited for IREE VM ref lifetimes.")},
    {IREE_SVL("effects-checked"),
     IREE_SVL("Number of ownership effects interpreted.")},
    {IREE_SVL("releases-inserted"),
     IREE_SVL("Number of ireevm.ref.release operations inserted.")},
    {IREE_SVL("edges-split"),
     IREE_SVL("Number of CFG edges split for reference cleanup.")},
};

static const loom_pass_info_t loom_ireevm_ref_lifetime_pass_info_storage = {
    .name = IREE_SVL("ireevm-ref-lifetime"),
    .description = IREE_SVL("Materialize explicit IREE VM ref lifetimes."),
    .kind = LOOM_PASS_MODULE,
    .statistic_defs = kIreeVmRefLifetimeStatisticDefs,
    .statistic_count = IREE_ARRAYSIZE(kIreeVmRefLifetimeStatisticDefs),
};

const loom_pass_info_t* loom_ireevm_ref_lifetime_pass_info(void) {
  return &loom_ireevm_ref_lifetime_pass_info_storage;
}

static void loom_ireevm_ref_lifetime_add_stat(loom_pass_t* pass,
                                              uint16_t statistic_index,
                                              uint64_t value) {
  if (pass->statistics && value != 0) {
    loom_pass_statistic_add(pass, statistic_index, (int64_t)value);
  }
}

iree_status_t loom_ireevm_ref_lifetime_run(loom_pass_t* pass,
                                           loom_module_t* module) {
  loom_ireevm_ref_lifetime_policy_t policy_data = {0};
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("ireevm.ref"), &policy_data.ref_type_name_id));
  const loom_ownership_lifetime_materialization_policy_t policy = {
      .family =
          {
              .name = IREE_SVL("ireevm.ref"),
              .type_matches = loom_ireevm_ref_lifetime_type_matches,
              .user_data = &policy_data,
          },
      .flags =
          LOOM_OWNERSHIP_LIFETIME_MATERIALIZATION_POLICY_OWNED_ARGUMENTS |
          LOOM_OWNERSHIP_LIFETIME_MATERIALIZATION_POLICY_OWNED_BODYLESS_RESULTS,
      .build_release = loom_ireevm_ref_lifetime_build_release,
  };
  loom_ownership_lifetime_materialize_options_t options = {
      .arena = pass->arena,
      .emitter = pass->diagnostic_emitter,
      .phase_name = pass->info->name,
      .policies = &policy,
      .policy_count = 1,
  };
  loom_ownership_lifetime_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_materialize_module(module, &options, &result));
  loom_ireevm_ref_lifetime_add_stat(
      pass, LOOM_IREEVM_REF_LIFETIME_STAT_BLOCKS_CHECKED,
      result.blocks_checked);
  loom_ireevm_ref_lifetime_add_stat(
      pass, LOOM_IREEVM_REF_LIFETIME_STAT_OPS_CHECKED, result.ops_checked);
  loom_ireevm_ref_lifetime_add_stat(
      pass, LOOM_IREEVM_REF_LIFETIME_STAT_EFFECTS_CHECKED,
      result.effects_checked);
  loom_ireevm_ref_lifetime_add_stat(
      pass, LOOM_IREEVM_REF_LIFETIME_STAT_RELEASES_INSERTED,
      result.releases_inserted);
  loom_ireevm_ref_lifetime_add_stat(
      pass, LOOM_IREEVM_REF_LIFETIME_STAT_EDGES_SPLIT, result.edges_split);
  if (result.releases_inserted != 0 || result.edges_split != 0) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}
