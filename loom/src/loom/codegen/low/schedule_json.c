// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule_json.h"

#include <inttypes.h>

#include "loom/analysis/liveness_json.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_string_view_t loom_low_schedule_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) return IREE_SV("<unnamed>");
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_low_schedule_json_function_name(
    const loom_low_schedule_sidecar_t* sidecar) {
  if (loom_low_func_def_isa(sidecar->function_op)) {
    return loom_low_schedule_json_symbol_name(
        sidecar->module, loom_low_func_def_callee(sidecar->function_op));
  }
  return IREE_SV("<unnamed>");
}

static const char* loom_low_schedule_json_node_kind(
    loom_low_schedule_node_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_SCHEDULE_NODE_STRUCTURAL:
      return "structural";
    case LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR:
      return "descriptor";
    case LOOM_LOW_SCHEDULE_NODE_TERMINATOR:
      return "terminator";
    default:
      return "unknown";
  }
}

static const char* loom_low_schedule_json_dependency_kind(
    loom_low_schedule_dependency_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_SCHEDULE_DEPENDENCY_SSA:
      return "ssa";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT:
      return "effect";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_CONTROL:
      return "control";
    default:
      return "unknown";
  }
}

static iree_status_t loom_low_schedule_json_write_nullable_string(
    loom_output_stream_t* stream, iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, value);
}

iree_status_t loom_low_schedule_format_json(
    const loom_low_schedule_sidecar_t* sidecar,
    iree_string_builder_t* builder) {
  if (!sidecar || !builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sidecar and builder are required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.low.schedule.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_schedule_json_function_name(sidecar)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, sidecar->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, sidecar->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"block_count\":%zu,\"node_count\":%zu,\"dependency_count\":%zu",
      sidecar->block_count, sidecar->node_count, sidecar->dependency_count));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"blocks\":["));
  for (iree_host_size_t i = 0; i < sidecar->block_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_low_schedule_block_t* block = &sidecar->blocks[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        "{\"index\":%zu,\"node_start\":%" PRIu32 ",\"node_count\":%" PRIu32
        ",\"scheduled_node_start\":%" PRIu32
        ",\"scheduled_node_count\":%" PRIu32 "}",
        i, block->node_start, block->node_count, block->scheduled_node_start,
        block->scheduled_node_count));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"nodes\":["));
  for (iree_host_size_t i = 0; i < sidecar->node_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_low_schedule_node_t* node = &sidecar->nodes[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        "{\"index\":%zu,\"block\":%" PRIu32 ",\"source_ordinal\":%" PRIu32
        ",\"scheduled_ordinal\":%" PRIu32 ",\"kind\":",
        i, node->block_index, node->source_ordinal, node->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        &stream, loom_low_schedule_json_node_kind(node->kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"op\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, loom_op_name(sidecar->module, node->op)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"descriptor\":"));
    IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_string(
        &stream, node->descriptor_key));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"schedule_class\":"));
    IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_string(
        &stream, node->schedule_class_name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"latency_cycles\":%" PRIu16 ",\"issue_use_count\":%" PRIu16
        ",\"effect_count\":%" PRIu16 "}",
        node->latency_cycles, node->issue_use_count, node->effect_count));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, ",\"scheduled_node_indices\":["));
  for (iree_host_size_t i = 0; i < sidecar->scheduled_node_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, "%" PRIu32, sidecar->scheduled_node_indices[i]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"dependencies\":["));
  for (iree_host_size_t i = 0; i < sidecar->dependency_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_low_schedule_dependency_t* dependency =
        &sidecar->dependencies[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, "{\"from\":%" PRIu32 ",\"to\":%" PRIu32 ",\"kind\":",
        dependency->producer_node, dependency->consumer_node));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        &stream, loom_low_schedule_json_dependency_kind(dependency->kind)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"operand\":"));
    if (dependency->operand_index == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "null"));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, "%" PRIu32, dependency->operand_index));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  if (sidecar->pressure_step_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"scheduled_pressure_steps\":["));
    for (iree_host_size_t i = 0; i < sidecar->pressure_step_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_pressure_step_t* step =
          &sidecar->pressure_steps[i];
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"node\":%" PRIu32 ",\"block\":%" PRIu32
          ",\"scheduled_ordinal\":%" PRIu32 ",\"live_units_before\":%" PRIu64
          ",\"killed_live_units\":%" PRIu64 ",\"produced_live_units\":%" PRIu64
          ",\"live_units_after\":%" PRIu64 "}",
          step->node_index, step->block_index, step->scheduled_ordinal,
          step->live_units_before, step->killed_live_units,
          step->produced_live_units, step->live_units_after));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"liveness\":"));
  IREE_RETURN_IF_ERROR(loom_liveness_format_json(&sidecar->liveness, builder));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  return iree_ok_status();
}
