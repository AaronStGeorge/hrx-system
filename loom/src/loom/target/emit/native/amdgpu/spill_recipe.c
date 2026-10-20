// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/spill_recipe.h"

#include <inttypes.h>

#include "loom/target/registers.h"

// SGPR, VGPR and AGPR allocation units are all 32-bit words.
#define LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BYTES 4u

typedef struct loom_amdgpu_spill_recipe_descriptor_t {
  // Descriptor table row for the selected scratch packet.
  const loom_low_descriptor_t* descriptor;
  // Offset immediate row used to validate the packet offset.
  const loom_low_immediate_t* offset_immediate;
} loom_amdgpu_spill_recipe_descriptor_t;

static uint32_t loom_amdgpu_spill_recipe_chunk_units(uint32_t remaining_units) {
  if (remaining_units >= 4) {
    return 4;
  }
  if (remaining_units >= 2) {
    return 2;
  }
  return 1;
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_spill_recipe_offset_descriptor_ref(
    loom_amdgpu_spill_recipe_kind_t kind, uint32_t chunk_units) {
  if (kind == LOOM_AMDGPU_SPILL_RECIPE_KIND_STORE) {
    switch (chunk_units) {
      case 1:
        return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_OFFSET_ONLY;
      case 2:
        return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B64_OFFSET_ONLY;
      default:
        return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B128_OFFSET_ONLY;
    }
  }
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
loom_amdgpu_spill_recipe_address_descriptor_ref(
    loom_amdgpu_spill_recipe_kind_t kind, uint32_t chunk_units) {
  if (kind == LOOM_AMDGPU_SPILL_RECIPE_KIND_STORE) {
    switch (chunk_units) {
      case 1:
        return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_VADDR;
      case 2:
        return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B64_VADDR;
      default:
        return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B128_VADDR;
    }
  }
  switch (chunk_units) {
    case 1:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_VADDR;
    case 2:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B64_VADDR;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B128_VADDR;
  }
}

static loom_amdgpu_spill_recipe_descriptor_t
loom_amdgpu_spill_recipe_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  if (descriptor == NULL) {
    return (loom_amdgpu_spill_recipe_descriptor_t){0};
  }
  const loom_amdgpu_descriptor_immediate_slots_t immediate_slots =
      loom_amdgpu_descriptor_immediate_slots(descriptor_set, descriptor);
  IREE_ASSERT(immediate_slots.address_offset !=
                  LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE,
              "generated spill descriptor must have an address offset");
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  IREE_ASSERT_LT(immediate_slots.address_offset,
                 descriptor_layout->immediate_count);
  const uint32_t immediate_index =
      descriptor_layout->immediate_start + immediate_slots.address_offset;
  IREE_ASSERT_LT(immediate_index, descriptor_set->immediate_count);
  return (loom_amdgpu_spill_recipe_descriptor_t){
      .descriptor = descriptor,
      .offset_immediate = &descriptor_set->immediates[immediate_index],
  };
}

static bool loom_amdgpu_spill_recipe_offset_fits(
    const loom_low_immediate_t* immediate, int64_t offset) {
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                  ? INT64_MAX
                                  : (int64_t)immediate->unsigned_max;
      return offset >= immediate->signed_min && offset <= maximum;
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      return offset >= 0 && (uint64_t)offset <= immediate->unsigned_max;
    default:
      return false;
  }
}

loom_amdgpu_spill_register_kind_t loom_amdgpu_spill_recipe_register_kind(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          descriptor_set->stable_id) {
    return LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED;
  }
  const uint16_t class_id = loom_low_register_type_class_id(type);
  if (class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR;
  }
  if (class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      class_id == loom_amdgpu_agpr_reg_class_id(descriptor_set)) {
    return LOOM_AMDGPU_SPILL_REGISTER_KIND_PER_LANE;
  }
  return LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED;
}

static uint32_t loom_amdgpu_spill_recipe_register_chunk_units(
    loom_amdgpu_spill_register_kind_t register_kind, uint32_t remaining_units) {
  if (register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    return 1;
  }
  return loom_amdgpu_spill_recipe_chunk_units(remaining_units);
}

static uint32_t loom_amdgpu_spill_recipe_aligned_chunk_units(
    loom_amdgpu_spill_register_kind_t register_kind, uint32_t chunk_units,
    int64_t segment_offset) {
  if (register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    return 1;
  }
  IREE_ASSERT_GE(segment_offset, 0);
  while (chunk_units > 1) {
    const uint32_t chunk_byte_length =
        chunk_units * LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BYTES;
    if (((uint64_t)segment_offset % chunk_byte_length) == 0) {
      return chunk_units;
    }
    chunk_units /= 2;
  }
  return chunk_units;
}

iree_status_t loom_amdgpu_spill_recipe_build(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_spill_recipe_kind_t kind, loom_type_t value_type,
    const loom_amdgpu_storage_layout_reference_t* storage_reference,
    int64_t operation_byte_offset, iree_arena_allocator_t* arena,
    loom_amdgpu_spill_recipe_t* out_recipe) {
  *out_recipe = (loom_amdgpu_spill_recipe_t){
      .kind = kind,
      .register_kind =
          loom_amdgpu_spill_recipe_register_kind(descriptor_set, value_type),
      .value_type = value_type,
  };
  if (out_recipe->register_kind ==
      LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED) {
    return iree_ok_status();
  }

  IREE_ASSERT_GE(operation_byte_offset, 0);
  const uint32_t unit_count = loom_low_register_type_unit_count(value_type);
  loom_amdgpu_spill_recipe_chunk_t* chunks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unit_count, sizeof(*chunks), (void**)&chunks));
  uint32_t chunk_count = 0;
  for (uint32_t unit_start = 0; unit_start < unit_count;) {
    uint32_t chunk_units = loom_amdgpu_spill_recipe_register_chunk_units(
        out_recipe->register_kind, unit_count - unit_start);
    const uint64_t chunk_byte_offset =
        (uint64_t)unit_start * LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BYTES;
    const uint64_t chunk_byte_length =
        (uint64_t)chunk_units * LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BYTES;
    const uint64_t access_byte_offset =
        (uint64_t)operation_byte_offset + chunk_byte_offset;
    IREE_ASSERT_LE(access_byte_offset + chunk_byte_length,
                   storage_reference->byte_length);

    uint64_t segment_byte_offset = 0;
    if (!iree_checked_add_u64(storage_reference->reservation.byte_offset,
                              storage_reference->byte_offset,
                              &segment_byte_offset) ||
        !iree_checked_add_u64(segment_byte_offset, access_byte_offset,
                              &segment_byte_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU scratch spill offset overflows");
    }
    if (segment_byte_offset > INT64_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU scratch spill offset %" PRIu64
                              " exceeds int64_t range",
                              segment_byte_offset);
    }
    chunk_units = loom_amdgpu_spill_recipe_aligned_chunk_units(
        out_recipe->register_kind, chunk_units, (int64_t)segment_byte_offset);

    loom_amdgpu_descriptor_ref_t descriptor_ref =
        loom_amdgpu_spill_recipe_offset_descriptor_ref(kind, chunk_units);
    loom_amdgpu_spill_recipe_descriptor_t descriptor =
        loom_amdgpu_spill_recipe_descriptor(descriptor_set, descriptor_ref);
    int64_t packet_byte_offset = (int64_t)segment_byte_offset;
    loom_amdgpu_spill_recipe_chunk_flags_t flags = 0;
    if (descriptor.descriptor == NULL ||
        !loom_amdgpu_spill_recipe_offset_fits(descriptor.offset_immediate,
                                              packet_byte_offset)) {
      if (access_byte_offset > INT64_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU scratch spill storage offset %" PRIu64
                                " exceeds int64_t range",
                                access_byte_offset);
      }
      descriptor_ref =
          loom_amdgpu_spill_recipe_address_descriptor_ref(kind, chunk_units);
      descriptor =
          loom_amdgpu_spill_recipe_descriptor(descriptor_set, descriptor_ref);
      IREE_ASSERT(descriptor.descriptor != NULL,
                  "generated descriptor set must provide addressed spill "
                  "packets");
      packet_byte_offset = 0;
      flags |= LOOM_AMDGPU_SPILL_RECIPE_CHUNK_FLAG_STORAGE_ADDRESS;
    }
    IREE_ASSERT(loom_amdgpu_spill_recipe_offset_fits(
                    descriptor.offset_immediate, packet_byte_offset),
                "selected spill descriptor must encode its packet offset");
    chunks[chunk_count++] = (loom_amdgpu_spill_recipe_chunk_t){
        .storage_byte_offset = access_byte_offset,
        .packet_byte_offset = packet_byte_offset,
        .unit_start = unit_start,
        .descriptor_ref = descriptor_ref,
        .unit_count = (uint8_t)chunk_units,
        .flags = flags,
    };
    unit_start += chunk_units;
  }
  out_recipe->chunks = chunks;
  out_recipe->chunk_count = chunk_count;
  return iree_ok_status();
}

typedef struct loom_amdgpu_spill_recipe_visitor_t {
  // Compact recipe being expanded.
  const loom_amdgpu_spill_recipe_t* recipe;
  // Consumer invoked for each expanded step.
  loom_amdgpu_spill_recipe_step_fn_t visit_step;
  // Opaque state forwarded to |visit_step|.
  void* user_data;
  // Next dense recipe-local register value index.
  loom_amdgpu_spill_recipe_value_index_t next_value;
  // Saved EXEC result consumed by the terminal restore step.
  loom_amdgpu_spill_recipe_value_index_t saved_exec;
} loom_amdgpu_spill_recipe_visitor_t;

static loom_amdgpu_spill_recipe_value_index_t
loom_amdgpu_spill_recipe_visitor_next_value(
    loom_amdgpu_spill_recipe_visitor_t* visitor) {
  IREE_ASSERT_LT(visitor->next_value, LOOM_AMDGPU_SPILL_RECIPE_VALUE_SOURCE);
  return visitor->next_value++;
}

static iree_status_t loom_amdgpu_spill_recipe_visitor_step(
    loom_amdgpu_spill_recipe_visitor_t* visitor,
    loom_amdgpu_spill_recipe_step_kind_t kind,
    loom_amdgpu_spill_recipe_step_result_flags_t result_flags,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_spill_recipe_value_index_t result,
    loom_amdgpu_spill_recipe_value_index_t operand0,
    loom_amdgpu_spill_recipe_value_index_t operand1, uint8_t operand_count,
    const loom_amdgpu_spill_recipe_chunk_t* chunk) {
  const loom_amdgpu_spill_recipe_step_t step = {
      .kind = kind,
      .result_flags = result_flags,
      .operand_count = operand_count,
      .descriptor_ref = descriptor_ref,
      .result = result,
      .operands = {operand0, operand1},
      .chunk = chunk,
  };
  return visitor->visit_step(visitor->user_data, &step);
}

static iree_status_t loom_amdgpu_spill_recipe_visit_exec_enter(
    loom_amdgpu_spill_recipe_visitor_t* visitor) {
  visitor->saved_exec = loom_amdgpu_spill_recipe_visitor_next_value(visitor);
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visitor_step(
      visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_READ,
      /*result_flags=*/0, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      visitor->saved_exec, LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
      LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
      /*operand_count=*/0,
      /*chunk=*/NULL));
  return loom_amdgpu_spill_recipe_visitor_step(
      visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_FULL,
      /*result_flags=*/0, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_FULL,
      LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
      LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, /*operand_count=*/0,
      /*chunk=*/NULL);
}

static iree_status_t loom_amdgpu_spill_recipe_visit_exec_restore(
    loom_amdgpu_spill_recipe_visitor_t* visitor) {
  return loom_amdgpu_spill_recipe_visitor_step(
      visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_RESTORE,
      /*result_flags=*/0, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, visitor->saved_exec,
      LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, /*operand_count=*/1,
      /*chunk=*/NULL);
}

static iree_status_t loom_amdgpu_spill_recipe_visit_address(
    loom_amdgpu_spill_recipe_visitor_t* visitor,
    const loom_amdgpu_spill_recipe_chunk_t* chunk,
    loom_amdgpu_spill_recipe_step_result_flags_t result_flags,
    loom_amdgpu_spill_recipe_value_index_t* out_address) {
  *out_address = LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE;
  if (!iree_all_bits_set(chunk->flags,
                         LOOM_AMDGPU_SPILL_RECIPE_CHUNK_FLAG_STORAGE_ADDRESS)) {
    return iree_ok_status();
  }
  *out_address = loom_amdgpu_spill_recipe_visitor_next_value(visitor);
  return loom_amdgpu_spill_recipe_visitor_step(
      visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_STORAGE_ADDRESS, result_flags,
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE, *out_address,
      LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
      /*operand_count=*/0, chunk);
}

static iree_status_t loom_amdgpu_spill_recipe_visit_store(
    loom_amdgpu_spill_recipe_visitor_t* visitor) {
  const loom_amdgpu_spill_recipe_t* recipe = visitor->recipe;
  const uint32_t source_units =
      loom_low_register_type_unit_count(recipe->value_type);
  for (uint32_t i = 0; i < recipe->chunk_count; ++i) {
    const loom_amdgpu_spill_recipe_chunk_t* chunk = &recipe->chunks[i];
    loom_amdgpu_spill_recipe_value_index_t value =
        LOOM_AMDGPU_SPILL_RECIPE_VALUE_SOURCE;
    if (source_units != chunk->unit_count) {
      const loom_amdgpu_spill_recipe_value_index_t slice =
          loom_amdgpu_spill_recipe_visitor_next_value(visitor);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visitor_step(
          visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_SLICE,
          /*result_flags=*/0, LOOM_AMDGPU_DESCRIPTOR_REF_NONE, slice, value,
          LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, /*operand_count=*/1, chunk));
      value = slice;
    }
    if (recipe->register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
      const loom_amdgpu_spill_recipe_value_index_t transport =
          loom_amdgpu_spill_recipe_visitor_next_value(visitor);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visitor_step(
          visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_SGPR_TO_VGPR,
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPANDED_EXECUTION,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY, transport, value,
          LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, /*operand_count=*/1, chunk));
      value = transport;
    }
    loom_amdgpu_spill_recipe_value_index_t address =
        LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE;
    const loom_amdgpu_spill_recipe_step_result_flags_t address_flags =
        recipe->register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR
            ? LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPANDED_EXECUTION
            : 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visit_address(
        visitor, chunk, address_flags, &address));
    const bool has_address = address != LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visitor_step(
        visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_STORE,
        /*result_flags=*/0, chunk->descriptor_ref,
        LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, has_address ? address : value,
        has_address ? value : LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
        has_address ? 2 : 1, chunk));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_recipe_visit_reload(
    loom_amdgpu_spill_recipe_visitor_t* visitor) {
  const loom_amdgpu_spill_recipe_t* recipe = visitor->recipe;
  for (uint32_t i = 0; i < recipe->chunk_count; ++i) {
    const loom_amdgpu_spill_recipe_chunk_t* chunk = &recipe->chunks[i];
    loom_amdgpu_spill_recipe_value_index_t address =
        LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE;
    loom_amdgpu_spill_recipe_step_result_flags_t address_flags =
        LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_PRIVATE_TO_CONSUMER;
    if (recipe->register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
      address_flags |=
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPANDED_EXECUTION;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visit_address(
        visitor, chunk, address_flags, &address));

    loom_amdgpu_spill_recipe_step_result_flags_t load_flags = 0;
    if (recipe->register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
      load_flags |=
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPANDED_EXECUTION |
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_PRIVATE_TO_CONSUMER;
    } else {
      load_flags |= LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_RELOAD_CHUNK;
      if (recipe->chunk_count == 1) {
        load_flags |= LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPORTED;
      } else {
        load_flags |=
            LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_PRIVATE_TO_CONSUMER;
      }
    }
    const loom_amdgpu_spill_recipe_value_index_t loaded =
        loom_amdgpu_spill_recipe_visitor_next_value(visitor);
    const bool has_address = address != LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visitor_step(
        visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_LOAD, load_flags,
        chunk->descriptor_ref, loaded,
        has_address ? address : LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
        LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, has_address ? 1 : 0, chunk));

    if (recipe->register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
      loom_amdgpu_spill_recipe_step_result_flags_t convert_flags =
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_RELOAD_CHUNK;
      if (recipe->chunk_count == 1) {
        convert_flags |= LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPORTED;
      } else {
        convert_flags |=
            LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_PRIVATE_TO_CONSUMER;
      }
      const loom_amdgpu_spill_recipe_value_index_t converted =
          loom_amdgpu_spill_recipe_visitor_next_value(visitor);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visitor_step(
          visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_VGPR_TO_SGPR, convert_flags,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32, converted, loaded,
          LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, /*operand_count=*/1, chunk));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_spill_recipe_visit(
    const loom_amdgpu_spill_recipe_t* recipe,
    loom_amdgpu_spill_recipe_step_fn_t visit_step, void* user_data) {
  IREE_ASSERT(recipe != NULL);
  IREE_ASSERT(visit_step != NULL);
  IREE_ASSERT_NE(recipe->register_kind,
                 LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED);
  loom_amdgpu_spill_recipe_visitor_t visitor = {
      .recipe = recipe,
      .visit_step = visit_step,
      .user_data = user_data,
      .saved_exec = LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
  };
  const bool requires_full_exec =
      recipe->register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR;
  if (requires_full_exec) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visit_exec_enter(&visitor));
  }
  if (recipe->kind == LOOM_AMDGPU_SPILL_RECIPE_KIND_STORE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visit_store(&visitor));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visit_reload(&visitor));
  }
  if (requires_full_exec) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visit_exec_restore(&visitor));
  }
  if (recipe->kind == LOOM_AMDGPU_SPILL_RECIPE_KIND_RELOAD &&
      recipe->chunk_count > 1) {
    const loom_amdgpu_spill_recipe_value_index_t result =
        loom_amdgpu_spill_recipe_visitor_next_value(&visitor);
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visitor_step(
        &visitor, LOOM_AMDGPU_SPILL_RECIPE_STEP_CONCAT,
        LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPORTED,
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE, result,
        LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE,
        LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE, /*operand_count=*/0,
        /*chunk=*/NULL));
  }
  return iree_ok_status();
}
