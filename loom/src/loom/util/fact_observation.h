// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_FACT_OBSERVATION_H_
#define LOOM_UTIL_FACT_OBSERVATION_H_

#include "loom/util/fact_table.h"

#ifndef NDEBUG

// Diagnoses reuse of a coherent fact observation across pending rewrite work or
// table scope retirement. The table instance must outlive the observation.
// Independent result definitions extend the table without expiring
// observations; queued equations expire them even if publication later produces
// equal facts.
//
// This token never publishes facts, resets a cache, or owns storage. It and its
// operations exist only in assertion builds, alongside the existing query path.
typedef struct loom_value_fact_observation_t {
  // Borrowed table; NULL represents disabled fact analysis.
  const loom_value_fact_table_t* table;
  // Semantic mutation batch captured for this observation.
  uint64_t generation;
} loom_value_fact_observation_t;

// Captures the current batch without observing its facts. An emptied derived
// cache may capture before publication, but queries must wait for completion.
static inline loom_value_fact_observation_t loom_value_fact_observation_capture(
    const loom_value_fact_table_t* table) {
  return (loom_value_fact_observation_t){
      .table = table,
      .generation = table ? table->observation.generation : 0,
  };
}

// Checks before consuming table facts or a retained result derived from them.
// Direct inference queries may observe a partial fixed point and do not use
// this contract; the token is for consumers requiring a coherently published
// table.
static inline void loom_value_fact_observation_assert_valid(
    loom_value_fact_observation_t observation) {
  IREE_ASSERT((!observation.table || !observation.table->observation.pending) &&
              "fact observation has pending equations");
  IREE_ASSERT(
      (!observation.table ||
       observation.generation == observation.table->observation.generation) &&
      "fact observation expired");
}

// Starts a diagnostic batch when existing maintenance queues an equation.
// Repeated enqueues within that batch share a generation, including equations
// discovered during publication. This does not schedule any work itself.
static inline void loom_value_fact_table_invalidate_observations(
    loom_value_fact_table_t* table) {
  if (table && !table->observation.pending) {
    ++table->observation.generation;
    table->observation.pending = true;
  }
}

// Completes a successfully published batch. Old observations stay expired;
// a derived cache must retire its contents before capturing the new generation.
static inline void loom_value_fact_table_publish_observations(
    loom_value_fact_table_t* table) {
  if (table) {
    table->observation.pending = false;
  }
}

#endif  // !NDEBUG

#endif  // LOOM_UTIL_FACT_OBSERVATION_H_
