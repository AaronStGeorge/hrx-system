// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_json.h"

#include <inttypes.h>

#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_string_view_t loom_low_allocation_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) return IREE_SV("<unnamed>");
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_low_allocation_json_function_name(
    const loom_low_allocation_sidecar_t* sidecar) {
  if (loom_low_func_def_isa(sidecar->function_op)) {
    return loom_low_allocation_json_symbol_name(
        sidecar->module, loom_low_func_def_callee(sidecar->function_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_allocation_json_type_kind_name(
    loom_type_kind_t type_kind) {
  switch (type_kind) {
    case LOOM_TYPE_NONE:
      return IREE_SV("none");
    case LOOM_TYPE_SCALAR:
      return IREE_SV("scalar");
    case LOOM_TYPE_TILE:
      return IREE_SV("tile");
    case LOOM_TYPE_TENSOR:
      return IREE_SV("tensor");
    case LOOM_TYPE_GROUP:
      return IREE_SV("group");
    case LOOM_TYPE_FUNCTION:
      return IREE_SV("function");
    case LOOM_TYPE_DIALECT:
      return IREE_SV("dialect");
    case LOOM_TYPE_ENCODING:
      return IREE_SV("encoding");
    case LOOM_TYPE_POOL:
      return IREE_SV("pool");
    case LOOM_TYPE_VECTOR:
      return IREE_SV("vector");
    case LOOM_TYPE_VIEW:
      return IREE_SV("view");
    case LOOM_TYPE_BUFFER:
      return IREE_SV("buffer");
    case LOOM_TYPE_REGISTER:
      return IREE_SV("register");
    case LOOM_TYPE_COUNT_:
      break;
  }
  return IREE_SV("unknown");
}

static const char* loom_low_allocation_json_mode_name(uint8_t mode) {
  switch (mode) {
    case 0:
    case LOOM_LOW_ALLOCATION_VIRTUAL:
      return "virtual";
    case LOOM_LOW_ALLOCATION_ASSIGNED:
      return "assigned";
    case LOOM_LOW_ALLOCATION_FIXED:
      return "fixed";
    default:
      return "unknown";
  }
}

static const char* loom_low_allocation_json_location_kind_name(
    loom_low_allocation_location_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED:
      return "unassigned";
    case LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER:
      return "physical_register";
    case LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID:
      return "target_id";
    case LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT:
      return "spill_slot";
    default:
      return "unknown";
  }
}

static const char* loom_low_allocation_json_remark_kind_name(
    loom_low_allocation_remark_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_REMARK_SPILL:
      return "spill";
    default:
      return "unknown";
  }
}

static const char* loom_low_allocation_json_copy_kind_name(
    loom_low_allocation_copy_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_COPY_COALESCED:
      return "coalesced";
    case LOOM_LOW_ALLOCATION_COPY_MATERIALIZED:
      return "materialized";
    default:
      return "unknown";
  }
}

static iree_status_t loom_low_allocation_json_write_string_or_null(
    const loom_module_t* module, loom_string_id_t string_id,
    loom_output_stream_t* stream) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream,
                                        module->strings.entries[string_id]);
}

static iree_status_t loom_low_allocation_json_write_scalar_name_or_null(
    loom_scalar_type_t scalar_type, loom_output_stream_t* stream) {
  const char* name = loom_scalar_type_name(scalar_type);
  if (!name) return loom_output_stream_write_cstring(stream, "null");
  return loom_json_write_escaped_cstring(stream, name);
}

static iree_status_t loom_low_allocation_json_write_type(
    const loom_module_t* module, loom_type_t type,
    loom_output_stream_t* stream) {
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_text_print_type(type, module, &escape_stream));
  return loom_output_stream_write_char(stream, '"');
}

static iree_status_t loom_low_allocation_json_write_value(
    const loom_low_allocation_sidecar_t* sidecar, loom_value_id_t value_id,
    loom_output_stream_t* stream) {
  const loom_module_t* module = sidecar->module;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"id\":%u,\"name\":", (unsigned)value_id));
  if (value_id < module->values.count) {
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_or_null(
        module, value->name_id, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"type\":"));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_json_write_type(module, value->type, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "null,\"type\":null"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_allocation_json_write_value_class(
    const loom_low_allocation_sidecar_t* sidecar,
    loom_liveness_value_class_t value_class, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"type_kind\":%u,\"type_kind_name\":",
      (unsigned)value_class.type_kind));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_low_allocation_json_type_kind_name(value_class.type_kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"element_type\":%u,\"element_type_name\":",
      (unsigned)value_class.element_type));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_scalar_name_or_null(
      value_class.element_type, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"register_class_id\":"));
  if (value_class.register_class_id == LOOM_STRING_ID_INVALID ||
      value_class.register_class_id >= sidecar->module->strings.count) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%u", (unsigned)value_class.register_class_id));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"register_class\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_or_null(
      sidecar->module, value_class.register_class_id, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_allocation_json_write_location(
    const loom_low_allocation_assignment_t* assignment,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream,
      loom_low_allocation_json_location_kind_name(assignment->location_kind)));
  const char* base_name =
      assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT
          ? "slot"
          : "base";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"%s\":%" PRIu32 ",\"count\":%" PRIu32 "}", base_name,
      assignment->location_base, assignment->location_count));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_json_write_assignment(
    const loom_low_allocation_sidecar_t* sidecar, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_assignment_t* assignment =
      &sidecar->assignments[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"value\":", index));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      sidecar, assignment->value_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"class\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value_class(
      sidecar, assignment->value_class, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"start_point\":%" PRIu32 ",\"end_point\":%" PRIu32
      ",\"unit_count\":%" PRIu32 ",\"location\":",
      assignment->start_point, assignment->end_point, assignment->unit_count));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_json_write_location(assignment, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_allocation_json_write_remark(
    const loom_low_allocation_sidecar_t* sidecar, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_remark_t* remark = &sidecar->remarks[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_remark_kind_name(remark->kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"assignment\":%" PRIu32 ",\"budget_units\":",
      remark->assignment_index));
  if (remark->budget_units == UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, "%" PRIu32,
                                                         remark->budget_units));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"required_units\":%" PRIu32 "}", remark->required_units));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_json_write_copy_decision(
    const loom_low_allocation_sidecar_t* sidecar, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_copy_decision_t* copy_decision =
      &sidecar->copy_decisions[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"kind\":", index));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_copy_kind_name(copy_decision->kind)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"source_value\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      sidecar, copy_decision->source_value_id, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"result_value\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      sidecar, copy_decision->result_value_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"source_assignment\":%" PRIu32 ",\"result_assignment\":%" PRIu32 "}",
      copy_decision->source_assignment_index,
      copy_decision->result_assignment_index));
  return iree_ok_status();
}

iree_status_t loom_low_allocation_format_json(
    const loom_low_allocation_sidecar_t* sidecar,
    iree_string_builder_t* builder) {
  if (!sidecar || !builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sidecar and builder are required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.low.allocation.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_allocation_json_function_name(sidecar)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, sidecar->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, sidecar->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"allocation_mode\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, loom_low_allocation_json_mode_name(sidecar->allocation_mode)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"assignment_count\":%zu,\"remark_count\":%zu"
      ",\"copy_decision_count\":%zu,\"spill_count\":%zu"
      ",\"coalesced_copy_count\":%zu,\"materialized_copy_count\":%zu",
      sidecar->assignment_count, sidecar->remark_count,
      sidecar->copy_decision_count, sidecar->spill_count,
      sidecar->coalesced_copy_count, sidecar->materialized_copy_count));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"assignments\":["));
  for (iree_host_size_t i = 0; i < sidecar->assignment_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_json_write_assignment(sidecar, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"copy_decisions\":["));
  for (iree_host_size_t i = 0; i < sidecar->copy_decision_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_json_write_copy_decision(sidecar, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"remarks\":["));
  for (iree_host_size_t i = 0; i < sidecar->remark_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_json_write_remark(sidecar, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]}"));
  return iree_ok_status();
}
