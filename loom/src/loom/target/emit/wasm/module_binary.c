// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/module_binary.h"

#include <inttypes.h>

#include "loom/codegen/low/packet.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/emit/wasm/binary_writer.h"
#include "loom/target/emit/wasm/function_body.h"
#include "loom/target/emit/wasm/types.h"

enum {
  LOOM_WASM_SECTION_TYPE = 1,
  LOOM_WASM_SECTION_FUNCTION = 3,
  LOOM_WASM_SECTION_MEMORY = 5,
  LOOM_WASM_SECTION_EXPORT = 7,
  LOOM_WASM_SECTION_CODE = 10,
};

enum {
  LOOM_WASM_EXPORT_KIND_FUNCTION = 0,
};

enum {
  LOOM_WASM_LIMITS_MIN_ONLY = 0,
};

enum {
  LOOM_WASM_FUNCTION_TYPE = 0x60,
};

static iree_status_t loom_wasm_module_write_section(
    loom_wasm_binary_writer_t* module_writer, uint8_t section_id,
    const loom_wasm_binary_writer_t* payload_writer) {
  if (payload_writer->length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm section %" PRIu8 " exceeds u32 size",
                            section_id);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(module_writer, section_id));
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(
      module_writer, (uint32_t)payload_writer->length));
  return loom_wasm_binary_write_bytes(module_writer, payload_writer->data,
                                      payload_writer->length);
}

static iree_status_t loom_wasm_module_write_name(
    loom_wasm_binary_writer_t* writer, iree_string_view_t name) {
  if (name.size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm name exceeds u32 size");
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(writer, (uint32_t)name.size));
  return loom_wasm_binary_write_bytes(writer, (const uint8_t*)name.data,
                                      name.size);
}

static iree_status_t loom_wasm_module_write_value_type_list(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_value_id_t* value_ids, uint32_t value_count,
    loom_wasm_binary_writer_t* writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(writer, value_count));
  for (uint32_t i = 0; i < value_count; ++i) {
    loom_wasm_value_type_t value_type = 0;
    IREE_RETURN_IF_ERROR(loom_wasm_value_type_from_register_type(
        module, descriptor_set, loom_module_value_type(module, value_ids[i]),
        &value_type));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(writer, value_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_type_section_payload(
    const loom_low_schedule_sidecar_t* schedule,
    loom_wasm_binary_writer_t* payload_writer) {
  loom_func_like_t function =
      loom_func_like_cast(schedule->module, (loom_op_t*)schedule->function_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm module emission requires a func-like op");
  }

  uint16_t parameter_count = 0;
  const loom_value_id_t* parameters =
      loom_func_like_arg_ids(function, &parameter_count);
  loom_value_slice_t results = loom_low_func_def_results(schedule->function_op);

  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(payload_writer, 1));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(payload_writer, LOOM_WASM_FUNCTION_TYPE));
  IREE_RETURN_IF_ERROR(loom_wasm_module_write_value_type_list(
      schedule->module, schedule->target.descriptor_set, parameters,
      parameter_count, payload_writer));
  return loom_wasm_module_write_value_type_list(
      schedule->module, schedule->target.descriptor_set, results.values,
      results.count, payload_writer);
}

static iree_status_t loom_wasm_module_write_type_section(
    const loom_low_schedule_sidecar_t* schedule,
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_type_section_payload(schedule, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_TYPE, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_function_section_payload(
    loom_wasm_binary_writer_t* payload_writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(payload_writer, 1));
  return loom_wasm_binary_write_u32_leb(payload_writer, 0);
}

static iree_status_t loom_wasm_module_write_function_section(
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_function_section_payload(&payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_FUNCTION, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_memory_section_payload(
    loom_wasm_binary_writer_t* payload_writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(payload_writer, 1));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(payload_writer, LOOM_WASM_LIMITS_MIN_ONLY));
  return loom_wasm_binary_write_u32_leb(payload_writer, 1);
}

static iree_status_t loom_wasm_module_write_memory_section(
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status =
      loom_wasm_module_write_memory_section_payload(&payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_MEMORY, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_export_section_payload(
    iree_string_view_t export_name, loom_wasm_binary_writer_t* payload_writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(payload_writer, 1));
  IREE_RETURN_IF_ERROR(
      loom_wasm_module_write_name(payload_writer, export_name));
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(
      payload_writer, LOOM_WASM_EXPORT_KIND_FUNCTION));
  return loom_wasm_binary_write_u32_leb(payload_writer, 0);
}

static iree_status_t loom_wasm_module_write_export_section(
    iree_string_view_t export_name, loom_wasm_binary_writer_t* module_writer,
    iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status = loom_wasm_module_write_export_section_payload(
      export_name, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_EXPORT, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_write_code_section_payload(
    const loom_wasm_function_body_t* function_body,
    loom_wasm_binary_writer_t* payload_writer) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u32_leb(payload_writer, 1));
  return loom_wasm_binary_write_bytes(payload_writer, function_body->data,
                                      function_body->data_length);
}

static iree_status_t loom_wasm_module_write_code_section(
    const loom_wasm_function_body_t* function_body,
    loom_wasm_binary_writer_t* module_writer, iree_allocator_t allocator) {
  loom_wasm_binary_writer_t payload_writer;
  loom_wasm_binary_writer_initialize(allocator, &payload_writer);
  iree_status_t status = loom_wasm_module_write_code_section_payload(
      function_body, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_section(
        module_writer, LOOM_WASM_SECTION_CODE, &payload_writer);
  }
  loom_wasm_binary_writer_deinitialize(&payload_writer);
  return status;
}

static iree_status_t loom_wasm_module_uses_memory(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation, bool* out_uses_memory) {
  *out_uses_memory = false;
  iree_host_size_t packet_count = loom_low_packet_count(schedule);
  for (iree_host_size_t i = 0; i < packet_count; ++i) {
    loom_low_packet_view_t packet = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_packet_view_at(schedule, allocation, i, &packet));
    if (!packet.descriptor ||
        packet.descriptor->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
      continue;
    }
    const uint16_t schedule_class_id = packet.descriptor->schedule_class_id;
    const loom_low_descriptor_set_t* descriptor_set =
        schedule->target.descriptor_set;
    if (schedule_class_id >= descriptor_set->schedule_class_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Wasm descriptor references invalid schedule class %" PRIu16,
          schedule_class_id);
    }
    const loom_low_schedule_class_t* schedule_class =
        &descriptor_set->schedule_classes[schedule_class_id];
    const loom_low_schedule_class_flags_t memory_flags =
        LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD |
        LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE;
    if ((schedule_class->flags & memory_flags) != 0) {
      *out_uses_memory = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_write_header(
    loom_wasm_binary_writer_t* module_writer) {
  static const uint8_t header[] = {
      0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
  };
  return loom_wasm_binary_write_bytes(module_writer, header, sizeof(header));
}

void loom_wasm_module_binary_deinitialize(loom_wasm_module_binary_t* module,
                                          iree_allocator_t allocator) {
  if (!module) {
    return;
  }
  iree_allocator_free(allocator, module->data);
  *module = (loom_wasm_module_binary_t){0};
}

iree_status_t loom_wasm_emit_single_function_module(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_view_t export_name, iree_allocator_t allocator,
    loom_wasm_module_binary_t* out_module) {
  if (!out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm module output is required");
  }
  *out_module = (loom_wasm_module_binary_t){0};
  if (iree_string_view_is_empty(export_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm function export name is required");
  }

  loom_wasm_function_body_t function_body = {0};
  iree_status_t status = loom_wasm_emit_function_body(
      schedule, allocation, allocator, &function_body);

  bool has_memory = false;
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_uses_memory(schedule, allocation, &has_memory);
  }

  loom_wasm_binary_writer_t module_writer;
  loom_wasm_binary_writer_initialize(allocator, &module_writer);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_header(&module_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_type_section(schedule, &module_writer,
                                                 allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_function_section(&module_writer, allocator);
  }
  if (iree_status_is_ok(status) && has_memory) {
    status = loom_wasm_module_write_memory_section(&module_writer, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_export_section(export_name, &module_writer,
                                                   allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_module_write_code_section(&function_body, &module_writer,
                                                 allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_module = (loom_wasm_module_binary_t){
        .data = module_writer.data,
        .data_length = module_writer.length,
        .has_memory = has_memory,
    };
    module_writer.data = NULL;
  }

  loom_wasm_binary_writer_deinitialize(&module_writer);
  loom_wasm_function_body_deinitialize(&function_body, allocator);
  if (!iree_status_is_ok(status)) {
    loom_wasm_module_binary_deinitialize(out_module, allocator);
  }
  return status;
}
