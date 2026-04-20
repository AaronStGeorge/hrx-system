// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/assembly.h"

#include <inttypes.h>

#include "loom/ops/low/ops.h"

iree_string_view_t loom_native_assembly_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_status_t loom_native_assembly_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string) {
  IREE_ASSERT_ARGUMENT(out_string);
  *out_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, string_offset, out_string));
  if (iree_string_view_is_empty(*out_string)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly descriptor string is empty");
  }
  return iree_ok_status();
}

const loom_named_attr_t* loom_native_assembly_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_native_assembly_module_string(module, attr->name_id), name)) {
      return attr;
    }
  }
  return NULL;
}

iree_status_t loom_native_assembly_read_i64_attr(const loom_module_t* module,
                                                 loom_named_attr_slice_t attrs,
                                                 iree_string_view_t name,
                                                 int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  const loom_named_attr_t* attr =
      loom_native_assembly_find_attr(module, attrs, name);
  if (attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly requires attribute '%.*s'",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly attribute '%.*s' must be i64",
                            (int)name.size, name.data);
  }
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

iree_status_t loom_native_assembly_append_block_label(
    const loom_low_schedule_sidecar_t* schedule, const loom_block_t* block,
    iree_string_builder_t* builder) {
  if (schedule == NULL || block == NULL || builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly block label requires schedule, "
                            "block, and builder");
  }
  const uint32_t block_index = loom_low_packet_block_index(schedule, block);
  if (block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly block is outside the scheduled "
                            "function");
  }
  return iree_string_builder_append_format(builder, ".Lbb%" PRIu32,
                                           block_index);
}

static iree_status_t loom_native_assembly_append_with_callback(
    const loom_native_assembly_append_packet_callback_t* callback,
    const char* missing_name,
    const loom_native_assembly_packet_context_t* context) {
  if (callback->fn == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "native assembly emitter does not support %s",
                            missing_name);
  }
  return callback->fn(callback->user_data, context);
}

static iree_status_t loom_native_assembly_append_structural_packet(
    const loom_native_assembly_format_options_t* options,
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  if (loom_low_copy_isa(op)) {
    return loom_native_assembly_append_with_callback(
        &options->append_copy_packet, "low.copy", context);
  }
  if (loom_low_return_isa(op)) {
    return loom_native_assembly_append_with_callback(
        &options->append_return_packet, "low.return", context);
  }
  if (loom_low_br_isa(op)) {
    return loom_native_assembly_append_with_callback(
        &options->append_branch_packet, "low.br", context);
  }
  if (loom_low_cond_br_isa(op)) {
    return loom_native_assembly_append_with_callback(
        &options->append_cond_branch_packet, "low.cond_br", context);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "native assembly emitter does not support "
                          "structural low packet");
}

static iree_status_t loom_native_assembly_append_packet(
    const loom_native_assembly_format_options_t* options,
    const loom_native_assembly_packet_context_t* context) {
  if (context->packet->descriptor != NULL) {
    return options->append_descriptor_packet.fn(
        options->append_descriptor_packet.user_data, context);
  }
  return loom_native_assembly_append_structural_packet(options, context);
}

iree_status_t loom_native_assembly_format_fragment(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_native_assembly_format_options_t* options,
    iree_string_builder_t* builder) {
  if (options == NULL || options->append_descriptor_packet.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly descriptor packet formatter is "
                            "required");
  }
  if (builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly output builder is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_sidecars(schedule, allocation));

  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    IREE_RETURN_IF_ERROR(loom_native_assembly_append_block_label(
        schedule, block->block, builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ":\n"));
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + i;
      loom_low_packet_view_t packet = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_packet_view_at(schedule, allocation, packet_index, &packet));
      if (loom_low_live_in_isa(packet.node->op)) {
        continue;
      }
      loom_native_assembly_packet_context_t context = {
          .schedule = schedule,
          .allocation = allocation,
          .packet = &packet,
          .builder = builder,
      };
      if (options->append_before_packet.fn != NULL) {
        IREE_RETURN_IF_ERROR(options->append_before_packet.fn(
            options->append_before_packet.user_data, &context));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "  "));
      IREE_RETURN_IF_ERROR(
          loom_native_assembly_append_packet(options, &context));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}
