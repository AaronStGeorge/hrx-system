// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Supplemental source values carried across target-created CFG edges.

#ifndef LOOM_CODEGEN_LOW_LOWER_CONTROL_CARRIERS_H_
#define LOOM_CODEGEN_LOW_LOWER_CONTROL_CARRIERS_H_

#include "loom/codegen/low/lower/control_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_control_carriers_t
    loom_low_lower_control_carriers_t;

// Planned target-created conditional edge whose destination may acquire a
// supplemental carrier tuple after the target has selected its control shape.
// The shared finalizer interposes an argument-free forwarding block only when
// the frozen destination requires one.
typedef struct loom_low_lower_control_carrier_edge_plan_t {
  loom_low_lower_block_ref_t destination;
  loom_low_lower_block_ref_t branch_destination;
} loom_low_lower_control_carrier_edge_plan_t;

// Source values retained at one authored resume destination. Values are in the
// exact supplemental block-argument order used by planning and emission.
typedef struct loom_low_lower_control_carrier_slice_t {
  const loom_value_id_t* values;
  uint16_t count;
} loom_low_lower_control_carrier_slice_t;

// Closes a target-created resume edge over materialized source values live at
// |destination|. A value already dominates every block in |capture_sources|
// and therefore the target's shared capture join; it needs no supplemental
// carrier. Every retained value is propagated only through reachable blocks
// where canonical source liveness marks it live-in.
//
// |capture_sources| contains dense source-CFG block indices for every semantic
// point that may feed the target's shared capture.
iree_status_t loom_low_lower_control_retain_resume_closure(
    loom_low_lower_context_t* context, const loom_block_t* destination,
    const uint16_t* capture_sources, uint16_t capture_source_count);

// Retains one target-created low.cond_br edge for carrier finalization. The
// destination must have no authored, realization, or target-private arguments;
// supplemental carriers are forwarded through an adapter when present.
iree_status_t loom_low_lower_control_plan_carrier_conditional_edge(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    loom_low_lower_block_ref_t destination,
    loom_low_lower_control_carrier_edge_plan_t* out_plan);

// Returns the supplemental carrier tuple of an authored source block.
loom_low_lower_control_carrier_slice_t loom_low_lower_control_source_carriers(
    const loom_low_lower_context_t* context, const loom_block_t* source_block);

// Returns the supplemental carrier tuple copied by a planned block reference.
// Appended target-private arguments are excluded.
loom_low_lower_control_carrier_slice_t loom_low_lower_control_block_carriers(
    const loom_low_lower_context_t* context, loom_low_lower_block_ref_t block);

// Returns the selected Low carrier type for one supplemental source value.
loom_type_t loom_low_lower_control_carrier_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Freezes closure requests and plans edge-local forwarding blocks for source
// conditional branches, whose Low form has no argument payload.
iree_status_t loom_low_lower_control_carriers_finalize(
    loom_low_lower_context_t* context);

// Appends supplemental arguments to authored Low blocks after ordinary and
// realization arguments have been mapped.
iree_status_t loom_low_lower_control_carriers_map_blocks(
    loom_low_lower_context_t* context);

// Temporarily maps carried source values to one authored Low block's
// supplemental arguments while that source block is emitted.
void loom_low_lower_control_carriers_enter_block(
    loom_low_lower_context_t* context, const loom_block_t* source_block);
void loom_low_lower_control_carriers_leave_block(
    loom_low_lower_context_t* context, const loom_block_t* source_block);

// Appends the current Low values for |destination|'s carrier tuple after an
// already-materialized prefix. The returned slice uses emission scratch.
iree_status_t loom_low_lower_control_append_carrier_arguments(
    loom_low_lower_context_t* context, loom_low_lower_block_ref_t destination,
    const loom_value_id_t* prefix_values, uint16_t prefix_count,
    loom_value_slice_t* out_values);

// Emits edge-local forwarding blocks planned for a source conditional branch.
// The caller must invoke this while the source block's carrier bindings are
// active and before constructing the final Low branch.
iree_status_t loom_low_lower_control_carriers_emit_edge_forwards(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_CONTROL_CARRIERS_H_
