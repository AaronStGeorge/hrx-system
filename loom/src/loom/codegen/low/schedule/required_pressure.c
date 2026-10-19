// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/required_pressure.h"

void loom_low_schedule_required_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  for (uint16_t domain = 0;
       domain < state->pressure_limits.unspillable_completion_domain_count;
       ++domain) {
    const loom_low_schedule_required_pressure_t* record =
        &pressure_state->required_pressure[domain];
    const int64_t growth =
        iree_max(record->candidate_delta_units,
                 iree_max(record->candidate_write_delta_units,
                          (int64_t)record->candidate_early_added_units));
    const uint32_t capacity =
        state->pressure_limits.unspillable_completion_domains[domain].capacity;
    if (growth > 0 && record->live_units + (uint64_t)growth > capacity) {
      score->flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY;
    }
  }
}
