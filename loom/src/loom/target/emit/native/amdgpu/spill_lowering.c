// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/spill_lowering.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/target_refs.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/registers.h"

#define LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BITS 32u

typedef struct loom_amdgpu_spill_storage_reference_t {
  // Resolved backing storage reservation.
  loom_amdgpu_storage_layout_reservation_t reservation;
  // Static byte offset from the reservation to the referenced storage view.
  uint64_t byte_offset;
  // Static byte length visible through the referenced storage handle.
  uint64_t byte_length;
} loom_amdgpu_spill_storage_reference_t;

typedef struct loom_amdgpu_spill_descriptor_t {
  // Descriptor table row for the selected scratch packet.
  const loom_low_descriptor_t* descriptor;
  // Module string ID for the descriptor key.
  loom_string_id_t opcode_id;
  // Offset immediate row used to validate lowering-created attributes.
  const loom_low_immediate_t* offset_immediate;
} loom_amdgpu_spill_descriptor_t;

typedef struct loom_amdgpu_spill_lowering_context_t {
  // Module being rewritten.
  loom_module_t* module;
  // Target-low descriptor set selected for this function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Built fixed-segment layout for the function storage reservations.
  loom_amdgpu_storage_layout_t storage_layout;
  // Descriptor-set-local ID for the AMDGPU VGPR register class.
  uint16_t vgpr_class_id;
  // Bytes in one VGPR allocation unit.
  uint32_t vgpr_unit_bytes;
  // Attribute name used by scratch packet descriptors.
  loom_string_id_t offset_attr_id;
} loom_amdgpu_spill_lowering_context_t;

static iree_status_t loom_amdgpu_spill_lowering_checked_add_u64(
    uint64_t lhs, uint64_t rhs, uint64_t* out_result) {
  *out_result = 0;
  if (lhs > UINT64_MAX - rhs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU scratch spill offset overflows");
  }
  *out_result = lhs + rhs;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_descriptor_ref(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_spill_descriptor_t* out_spill_descriptor) {
  *out_spill_descriptor = (loom_amdgpu_spill_descriptor_t){0};
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU spill lowering references missing descriptor ref %" PRIu16,
        descriptor_ref);
  }
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU spill lowering descriptor ref %" PRIu16
                            " has no descriptor key",
                            descriptor_ref);
  }
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(rewriter->module, key, &opcode_id));

  const loom_low_immediate_t* offset_immediate = NULL;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t immediate_index = descriptor->immediate_start + i;
    if (immediate_index >= descriptor_set->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU spill lowering descriptor ref %" PRIu16
                              " references immediate row %" PRIu32
                              " outside the descriptor set",
                              descriptor_ref, immediate_index);
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    iree_string_view_t immediate_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    if (iree_string_view_equal(immediate_name, IREE_SV("offset"))) {
      offset_immediate = immediate;
      break;
    }
  }
  if (offset_immediate == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU spill lowering descriptor ref %" PRIu16
                            " has no offset immediate",
                            descriptor_ref);
  }

  *out_spill_descriptor = (loom_amdgpu_spill_descriptor_t){
      .descriptor = descriptor,
      .opcode_id = opcode_id,
      .offset_immediate = offset_immediate,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_validate_offset_immediate(
    const loom_amdgpu_spill_descriptor_t* spill_descriptor, int64_t offset) {
  const loom_low_immediate_t* immediate = spill_descriptor->offset_immediate;
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                  ? INT64_MAX
                                  : (int64_t)immediate->unsigned_max;
      if (offset >= immediate->signed_min && offset <= maximum) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU scratch spill offset %" PRId64
          " does not fit the selected offset-only scratch encoding range "
          "[%" PRId64 ", %" PRId64 "]",
          offset, immediate->signed_min, maximum);
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      if (offset >= 0 && (uint64_t)offset <= immediate->unsigned_max) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU scratch spill offset %" PRId64
          " does not fit the selected offset-only scratch encoding range "
          "[0, %" PRIu64 "]",
          offset, immediate->unsigned_max);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU scratch spill offset immediate has unsupported kind %u",
          (unsigned)immediate->kind);
  }
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_storage_reference(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_value_id_t storage_value_id,
    loom_amdgpu_spill_storage_reference_t* out_reference) {
  *out_reference = (loom_amdgpu_spill_storage_reference_t){0};
  if (storage_value_id == LOOM_VALUE_ID_INVALID ||
      storage_value_id >= context->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found an invalid storage value");
  }
  const loom_value_t* storage_value =
      loom_module_value(context->module, storage_value_id);
  if (loom_value_is_block_arg(storage_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU spill lowering requires storage defined by "
                            "low.storage.reserve or low.storage.view");
  }
  const loom_op_t* defining_op = loom_value_def_op(storage_value);
  if (defining_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found storage without a defining op");
  }

  if (loom_low_storage_reserve_isa(defining_op)) {
    loom_amdgpu_storage_layout_reservation_t reservation = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_lookup(
        &context->storage_layout, storage_value_id, &reservation));
    *out_reference = (loom_amdgpu_spill_storage_reference_t){
        .reservation = reservation,
        .byte_offset = 0,
        .byte_length = reservation.byte_size,
    };
    return iree_ok_status();
  }

  if (!loom_low_storage_view_isa(defining_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU spill lowering requires storage defined by "
                            "low.storage.reserve or low.storage.view");
  }

  loom_amdgpu_spill_storage_reference_t source_reference = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_storage_reference(
      context, loom_low_storage_view_source(defining_op), &source_reference));
  const int64_t signed_offset = loom_low_storage_view_offset(defining_op);
  const int64_t signed_byte_length =
      loom_low_storage_view_byte_length(defining_op);
  if (signed_offset < 0 || signed_byte_length <= 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires non-negative storage view offsets and "
        "positive storage view lengths");
  }
  const uint64_t offset = (uint64_t)signed_offset;
  const uint64_t byte_length = (uint64_t)signed_byte_length;
  if (offset > source_reference.byte_length ||
      byte_length > source_reference.byte_length - offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU spill lowering storage view range exceeds its source storage");
  }
  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      source_reference.byte_offset, offset, &byte_offset));
  *out_reference = (loom_amdgpu_spill_storage_reference_t){
      .reservation = source_reference.reservation,
      .byte_offset = byte_offset,
      .byte_length = byte_length,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_validate_storage_space(
    const loom_amdgpu_spill_storage_reference_t* storage_reference) {
  switch (storage_reference->reservation.space) {
    case LOOM_STORAGE_SPACE_PRIVATE:
    case LOOM_STORAGE_SPACE_SCRATCH:
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU spill lowering only supports private or scratch storage, not "
          "%s storage",
          loom_storage_space_name(storage_reference->reservation.space));
  }
}

static iree_status_t loom_amdgpu_spill_lowering_access_offset(
    const loom_amdgpu_spill_storage_reference_t* storage_reference,
    int64_t operation_offset, uint64_t chunk_byte_offset,
    uint64_t chunk_byte_length, int64_t* out_absolute_offset) {
  *out_absolute_offset = 0;
  if (operation_offset < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires non-negative spill offsets");
  }
  uint64_t access_offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      (uint64_t)operation_offset, chunk_byte_offset, &access_offset));
  uint64_t access_end = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      access_offset, chunk_byte_length, &access_end));
  if (access_end > storage_reference->byte_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU spill lowering access range [%" PRIu64 ", %" PRIu64
        ") exceeds storage reference size %" PRIu64,
        access_offset, access_end, storage_reference->byte_length);
  }

  uint64_t absolute_offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      storage_reference->reservation.byte_offset,
      storage_reference->byte_offset, &absolute_offset));
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      absolute_offset, access_offset, &absolute_offset));
  if (absolute_offset > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU scratch spill offset %" PRIu64
                            " exceeds int64_t range",
                            absolute_offset);
  }
  *out_absolute_offset = (int64_t)absolute_offset;
  return iree_ok_status();
}

static uint32_t loom_amdgpu_spill_lowering_chunk_units(
    uint32_t remaining_units) {
  if (remaining_units >= 4) {
    return 4;
  }
  if (remaining_units >= 2) {
    return 2;
  }
  return 1;
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_spill_lowering_load_descriptor_ref(uint32_t chunk_units) {
  switch (chunk_units) {
    case 1:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_OFFSET_ONLY;
    case 2:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B64_OFFSET_ONLY;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B128_OFFSET_ONLY;
  }
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_spill_lowering_store_descriptor_ref(uint32_t chunk_units) {
  switch (chunk_units) {
    case 1:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_OFFSET_ONLY;
    case 2:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B64_OFFSET_ONLY;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B128_OFFSET_ONLY;
  }
}

static iree_status_t loom_amdgpu_spill_lowering_make_offset_attr(
    const loom_amdgpu_spill_lowering_context_t* context, int64_t offset,
    loom_named_attr_t* out_attr) {
  *out_attr = (loom_named_attr_t){
      .name_id = context->offset_attr_id,
      .value = loom_attr_i64(offset),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_make_chunk_type(
    loom_module_t* module, loom_type_t base_type, uint32_t chunk_units,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_type_t chunk_type =
      loom_low_register_type_with_unit_count(base_type, chunk_units);
  return loom_module_intern_type(module, chunk_type, out_type);
}

static iree_status_t loom_amdgpu_spill_lowering_validate_register_type(
    const loom_amdgpu_spill_lowering_context_t* context, loom_type_t type) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          context->descriptor_set->stable_id ||
      loom_low_register_type_class_id(type) != context->vgpr_class_id) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU spill lowering currently supports only VGPR register values");
  }
  if (context->vgpr_unit_bytes == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found a zero-byte VGPR allocation unit");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_slice(
    loom_rewriter_t* rewriter, loom_value_id_t source, uint32_t source_units,
    uint32_t chunk_start, uint32_t chunk_units, loom_type_t source_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (source_units == chunk_units) {
    *out_value = source;
    return iree_ok_status();
  }
  loom_type_t chunk_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_chunk_type(
      rewriter->module, source_type, chunk_units, &chunk_type));
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(&rewriter->builder, source,
                                            chunk_start, chunk_type, location,
                                            &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_store_chunk(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_value_id_t value, uint32_t chunk_units,
    int64_t offset, loom_location_id_t location) {
  loom_amdgpu_spill_descriptor_t spill_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_descriptor_ref(
      rewriter, context->descriptor_set,
      loom_amdgpu_spill_lowering_store_descriptor_ref(chunk_units),
      &spill_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_offset_immediate(
      &spill_descriptor, offset));
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_make_offset_attr(context, offset, &attr));
  const loom_value_id_t operands[] = {value};
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, spill_descriptor.descriptor,
      spill_descriptor.opcode_id, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(&attr, 1), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &store_op);
}

static iree_status_t loom_amdgpu_spill_lowering_load_chunk(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, uint32_t chunk_units, loom_type_t result_type,
    int64_t offset, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_spill_descriptor_t spill_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_descriptor_ref(
      rewriter, context->descriptor_set,
      loom_amdgpu_spill_lowering_load_descriptor_ref(chunk_units),
      &spill_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_offset_immediate(
      &spill_descriptor, offset));
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_make_offset_attr(context, offset, &attr));
  const loom_type_t result_types[] = {result_type};
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, spill_descriptor.descriptor,
      spill_descriptor.opcode_id, /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(&attr, 1), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_rewrite_spill(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_value_id_t value = loom_low_spill_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_validate_register_type(context, value_type));
  const uint32_t unit_count = loom_low_register_type_unit_count(value_type);
  if (unit_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found a zero-unit spill value");
  }

  loom_amdgpu_spill_storage_reference_t storage_reference = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_storage_reference(
      context, loom_low_spill_storage(op), &storage_reference));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_validate_storage_space(&storage_reference));

  loom_builder_set_before(&rewriter->builder, op);
  for (uint32_t chunk_start = 0; chunk_start < unit_count;) {
    const uint32_t chunk_units =
        loom_amdgpu_spill_lowering_chunk_units(unit_count - chunk_start);
    const uint64_t chunk_byte_offset =
        (uint64_t)chunk_start * context->vgpr_unit_bytes;
    const uint64_t chunk_byte_length =
        (uint64_t)chunk_units * context->vgpr_unit_bytes;
    int64_t absolute_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_access_offset(
        &storage_reference, loom_low_spill_offset(op), chunk_byte_offset,
        chunk_byte_length, &absolute_offset));
    loom_value_id_t chunk_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_slice(
        rewriter, value, unit_count, chunk_start, chunk_units, value_type,
        op->location, &chunk_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_store_chunk(
        context, rewriter, chunk_value, chunk_units, absolute_offset,
        op->location));
    chunk_start += chunk_units;
  }
  return loom_rewriter_erase(rewriter, op);
}

static iree_status_t loom_amdgpu_spill_lowering_rewrite_reload(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_value_id_t result = loom_low_reload_result(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_validate_register_type(context, result_type));
  const uint32_t unit_count = loom_low_register_type_unit_count(result_type);
  if (unit_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found a zero-unit reload result");
  }

  loom_amdgpu_spill_storage_reference_t storage_reference = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_storage_reference(
      context, loom_low_reload_storage(op), &storage_reference));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_validate_storage_space(&storage_reference));

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const iree_host_size_t max_loaded_chunk_count = (unit_count + 3u) / 4u;
  loom_value_id_t* loaded_chunks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, max_loaded_chunk_count, sizeof(*loaded_chunks),
      (void**)&loaded_chunks));
  iree_host_size_t loaded_chunk_count = 0;
  for (uint32_t chunk_start = 0; chunk_start < unit_count;) {
    const uint32_t chunk_units =
        loom_amdgpu_spill_lowering_chunk_units(unit_count - chunk_start);
    const uint64_t chunk_byte_offset =
        (uint64_t)chunk_start * context->vgpr_unit_bytes;
    const uint64_t chunk_byte_length =
        (uint64_t)chunk_units * context->vgpr_unit_bytes;
    int64_t absolute_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_access_offset(
        &storage_reference, loom_low_reload_offset(op), chunk_byte_offset,
        chunk_byte_length, &absolute_offset));
    loom_type_t chunk_type = result_type;
    if (chunk_units != unit_count) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_chunk_type(
          context->module, result_type, chunk_units, &chunk_type));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_load_chunk(
        context, rewriter, chunk_units, chunk_type, absolute_offset,
        op->location, &loaded_chunks[loaded_chunk_count++]));
    chunk_start += chunk_units;
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (loaded_chunk_count == 1) {
    replacement = loaded_chunks[0];
  } else {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        &rewriter->builder, loaded_chunks, loaded_chunk_count, result_type,
        op->location, &concat_op));
    replacement = loom_low_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_amdgpu_spill_lowering_collect_ops(
    loom_region_t* body, loom_op_t** ops, iree_host_size_t op_capacity,
    iree_host_size_t* out_op_count) {
  *out_op_count = 0;
  loom_block_t* block = NULL;
  loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_spill_isa(op) && !loom_low_reload_isa(op)) {
        continue;
      }
      if (ops != NULL && *out_op_count >= op_capacity) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU spill lowering op count changed while collecting ops");
      }
      if (ops != NULL) {
        ops[*out_op_count] = op;
      }
      ++(*out_op_count);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_initialize_context(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_spill_lowering_context_t* out_context,
    iree_arena_allocator_t* scratch_arena) {
  *out_context = (loom_amdgpu_spill_lowering_context_t){
      .module = module,
      .descriptor_set = descriptor_set,
  };
  if (descriptor_set->reg_class_count <= LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU spill lowering descriptor set has no VGPR register class");
  }
  const loom_low_reg_class_t* vgpr_class =
      &descriptor_set->reg_classes[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  if (vgpr_class->alloc_unit_bits != LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BITS) {
    iree_string_view_t class_name = loom_low_descriptor_set_string(
        descriptor_set, vgpr_class->name_string_offset);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU spill lowering expects 32-bit VGPR allocation units for "
        "scratch packets, but register class '%.*s' has %" PRIu16 " bits",
        (int)class_name.size, class_name.data, vgpr_class->alloc_unit_bits);
  }
  out_context->vgpr_unit_bytes = vgpr_class->alloc_unit_bits / 8u;
  out_context->vgpr_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("offset"),
                                                 &out_context->offset_attr_id));
  return loom_amdgpu_storage_layout_build(module, function_op, scratch_arena,
                                          &out_context->storage_layout);
}

iree_status_t loom_amdgpu_lower_spill_traffic(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* scratch_arena) {
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires module, function, descriptor set, and "
        "scratch arena");
  }
  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires a low function body");
  }

  iree_host_size_t op_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_collect_ops(body, NULL, 0, &op_count));
  if (op_count == 0) {
    return iree_ok_status();
  }

  loom_op_t** ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, op_count,
                                                 sizeof(*ops), (void**)&ops));
  iree_host_size_t collected_op_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_collect_ops(
      body, ops, op_count, &collected_op_count));
  if (collected_op_count != op_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering op count changed while collecting ops");
  }

  loom_amdgpu_spill_lowering_context_t context = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_context(
      module, function_op, descriptor_set, &context, scratch_arena));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < op_count && iree_status_is_ok(status); ++i) {
    if (loom_low_spill_isa(ops[i])) {
      status =
          loom_amdgpu_spill_lowering_rewrite_spill(&context, &rewriter, ops[i]);
    } else if (loom_low_reload_isa(ops[i])) {
      status = loom_amdgpu_spill_lowering_rewrite_reload(&context, &rewriter,
                                                         ops[i]);
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
