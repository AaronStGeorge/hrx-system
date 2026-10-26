// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static dependency ordering for exclusive storage, tied partial-register
// continuations, and repair materializations.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_SETUP_ORDER_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_SETUP_ORDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_schedule_setup_order_t {
  // Whether node classification found setup requiring dependency ordering.
  bool has_members;
  // During graph construction, the sole complete successor, UINT32_MAX for
  // none, or the producer itself for fan-out. Finish replaces these facts
  // with transitive setup completion nodes. NULL disables setup ordering.
  uint32_t* completion_nodes;
  // During construction, the sole SSA successor, UINT32_MAX for none, or
  // the producer itself for value fan-out. Finish records completion entries.
  uint32_t* entry_nodes;
} loom_low_schedule_setup_order_t;

// Allocates graph scratch after membership classification.
iree_status_t loom_low_schedule_setup_order_initialize(
    uint32_t node_count, iree_arena_allocator_t* arena,
    loom_low_schedule_setup_order_t* out_order);

// Records complete dependency fan-out and the distinct SSA consumer identity.
static inline void loom_low_schedule_setup_order_record_dependency(
    loom_low_schedule_setup_order_t* order, uint32_t producer,
    uint32_t consumer, bool is_value_dependency) {
  if (order->completion_nodes == NULL) {
    return;
  }
  if (is_value_dependency) {
    uint32_t* value_successor = &order->entry_nodes[producer];
    if (*value_successor == UINT32_MAX) {
      *value_successor = consumer;
    } else if (*value_successor != consumer) {
      *value_successor = producer;
    }
  }
  uint32_t* completion = &order->completion_nodes[producer];
  if (*completion == UINT32_MAX) {
    *completion = consumer;
  } else if (*completion != consumer) {
    *completion = producer;
  }
}

struct loom_low_schedule_build_state_t;
struct loom_low_schedule_node_t;

// Classifies exclusive storage setup from the retained pressure domains and
// tied partial-register continuations from descriptor contracts. Retained
// private producers receive ordering independently of the optional
// function-wide structural setup policy used during allocation repair. The
// existing node walk records membership before dependency construction.
void loom_low_schedule_setup_order_classify_node(
    struct loom_low_schedule_build_state_t* state,
    struct loom_low_schedule_node_t* node, bool order_structural_setup);

// Closes setup dependencies before adjacency construction. Every member has
// one value consumer inside its source-order range, and any other successor
// already depends on that consumer. External prerequisites can therefore
// precede the entire setup chain without creating a cycle.
iree_status_t loom_low_schedule_setup_order_finish(
    struct loom_low_schedule_build_state_t* state, uint32_t node_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_SETUP_ORDER_H_
