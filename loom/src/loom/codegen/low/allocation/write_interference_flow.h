// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Control-flow spans for retained reads at allocation write points.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_FLOW_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_FLOW_H_

#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/write_interference_events.h"
#include "loom/util/cfg_graph.h"

iree_status_t loom_low_write_flow_build(
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    uint32_t point_count, iree_arena_allocator_t* arena,
    loom_low_write_flow_t* out_flow);

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_FLOW_H_
