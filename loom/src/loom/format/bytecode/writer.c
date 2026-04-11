// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer.h"

#include <string.h>

#include "loom/format/bytecode/varint.h"
#include "loom/ir/context.h"
#include "loom/ops/op_defs.h"

#define LOOM_BYTECODE_DEFAULT_PRODUCER "loom-c"

// Maximum nesting depth for regions. Prevents stack exhaustion from
// pathologically deep IR (e.g., 1000 nested scf.execute_region ops).
// Each level of nesting uses ~256 bytes of stack (local variables in
// the write_region/write_block/write_operation chain).
#define LOOM_BYTECODE_MAX_REGION_DEPTH 256

//===----------------------------------------------------------------------===//
// Page-buffered stream writer
//===----------------------------------------------------------------------===//
//
// Amortizes iree_io_stream_write vtable dispatch cost by accumulating
// small writes in a local 4KB page buffer. Flushes to the stream when
// full. All section data flows through this writer.

#define LOOM_BYTECODE_PAGE_SIZE 4096

typedef struct loom_bytecode_page_writer_t {
  iree_io_stream_t* stream;
  uint8_t page[LOOM_BYTECODE_PAGE_SIZE];
  iree_host_size_t position;  // Bytes currently in the page.
  uint64_t total_written;     // Logical total (flushed + position).
                              // uint64_t to match the format's 64-bit
                              // offsets and avoid overflow on 32-bit.
} loom_bytecode_page_writer_t;

static void loom_bytecode_page_writer_initialize(
    loom_bytecode_page_writer_t* writer, iree_io_stream_t* stream) {
  writer->stream = stream;
  writer->position = 0;
  writer->total_written = 0;
}

static iree_status_t loom_bytecode_page_writer_flush(
    loom_bytecode_page_writer_t* writer) {
  if (writer->position == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(writer->stream, writer->position, writer->page));
  writer->position = 0;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_page_writer_write(
    loom_bytecode_page_writer_t* writer, const void* data,
    iree_host_size_t length) {
  writer->total_written += length;
  const uint8_t* source = (const uint8_t*)data;

  // Fast path: fits in current page.
  if (IREE_LIKELY(writer->position + length <= LOOM_BYTECODE_PAGE_SIZE)) {
    memcpy(writer->page + writer->position, source, length);
    writer->position += length;
    return iree_ok_status();
  }

  // Fill the remainder of the current page.
  iree_host_size_t remaining = LOOM_BYTECODE_PAGE_SIZE - writer->position;
  if (remaining > 0) {
    memcpy(writer->page + writer->position, source, remaining);
    writer->position = LOOM_BYTECODE_PAGE_SIZE;
    source += remaining;
    length -= remaining;
  }

  // Flush the full page.
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_flush(writer));

  // Write full pages directly to the stream (skip the page buffer).
  while (length >= LOOM_BYTECODE_PAGE_SIZE) {
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(writer->stream, LOOM_BYTECODE_PAGE_SIZE, source));
    source += LOOM_BYTECODE_PAGE_SIZE;
    length -= LOOM_BYTECODE_PAGE_SIZE;
  }

  // Copy remainder into the empty page.
  if (length > 0) {
    memcpy(writer->page, source, length);
    writer->position = length;
  }
  return iree_ok_status();
}

// Typed write helpers. Each encodes into a small stack buffer, then
// writes through the page writer.

static iree_status_t loom_bytecode_page_writer_write_u8(
    loom_bytecode_page_writer_t* writer, uint8_t value) {
  return loom_bytecode_page_writer_write(writer, &value, 1);
}

static iree_status_t loom_bytecode_page_writer_write_u16_le(
    loom_bytecode_page_writer_t* writer, uint16_t value) {
  uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
  return loom_bytecode_page_writer_write(writer, bytes, 2);
}

static iree_status_t loom_bytecode_page_writer_write_u32_le(
    loom_bytecode_page_writer_t* writer, uint32_t value) {
  uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8),
                      (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
  return loom_bytecode_page_writer_write(writer, bytes, 4);
}

static iree_status_t loom_bytecode_page_writer_write_u64_le(
    loom_bytecode_page_writer_t* writer, uint64_t value) {
  uint8_t bytes[8] = {
      (uint8_t)value,         (uint8_t)(value >> 8),  (uint8_t)(value >> 16),
      (uint8_t)(value >> 24), (uint8_t)(value >> 32), (uint8_t)(value >> 40),
      (uint8_t)(value >> 48), (uint8_t)(value >> 56),
  };
  return loom_bytecode_page_writer_write(writer, bytes, 8);
}

static iree_status_t loom_bytecode_page_writer_write_uvarint(
    loom_bytecode_page_writer_t* writer, uint64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_encode(value, span, &length));
  return loom_bytecode_page_writer_write(writer, buffer, length);
}

static iree_status_t loom_bytecode_page_writer_write_svarint(
    loom_bytecode_page_writer_t* writer, int64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_svarint_encode(value, span, &length));
  return loom_bytecode_page_writer_write(writer, buffer, length);
}

static iree_status_t loom_bytecode_page_writer_write_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, text.size));
  return loom_bytecode_page_writer_write(writer, text.data, text.size);
}

static iree_status_t loom_bytecode_page_writer_write_zeros(
    loom_bytecode_page_writer_t* writer, iree_host_size_t count) {
  static const uint8_t zeros[64] = {0};
  while (count > 0) {
    iree_host_size_t chunk = count < sizeof(zeros) ? count : sizeof(zeros);
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, zeros, chunk));
    count -= chunk;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_page_writer_pad_to_alignment(
    loom_bytecode_page_writer_t* writer, iree_host_size_t alignment) {
  iree_host_size_t remainder = writer->total_written % alignment;
  if (remainder == 0) return iree_ok_status();
  return loom_bytecode_page_writer_write_zeros(writer, alignment - remainder);
}

// Writes a null-terminated string followed by padding to 8-byte alignment.
static iree_status_t loom_bytecode_page_writer_write_null_terminated_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write(writer, text.data, text.size));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 0));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// String builder emit helpers (for SYMBOLS section buffering)
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_emit_u8(iree_string_builder_t* builder,
                                           uint8_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 1, &head));
  if (head) head[0] = (char)value;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_u16_le(iree_string_builder_t* builder,
                                               uint16_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 2, &head));
  if (head) {
    head[0] = (char)(value & 0xFF);
    head[1] = (char)((value >> 8) & 0xFF);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_u32_le(iree_string_builder_t* builder,
                                               uint32_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 4, &head));
  if (head) {
    head[0] = (char)(value & 0xFF);
    head[1] = (char)((value >> 8) & 0xFF);
    head[2] = (char)((value >> 16) & 0xFF);
    head[3] = (char)((value >> 24) & 0xFF);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_u64_le(iree_string_builder_t* builder,
                                               uint64_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 8, &head));
  if (head) {
    for (int i = 0; i < 8; ++i) {
      head[i] = (char)((value >> (i * 8)) & 0xFF);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_uvarint(iree_string_builder_t* builder,
                                                uint64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_encode(value, span, &length));
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)buffer, length));
}

static iree_status_t loom_bytecode_emit_svarint(iree_string_builder_t* builder,
                                                int64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_svarint_encode(value, span, &length));
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)buffer, length));
}

static void loom_bytecode_patch_u64_le(iree_string_builder_t* builder,
                                       iree_host_size_t offset,
                                       uint64_t value) {
  char* buffer = builder->buffer;
  for (int i = 0; i < 8; ++i) {
    buffer[offset + i] = (char)((value >> (i * 8)) & 0xFF);
  }
}

//===----------------------------------------------------------------------===//
// Numbering context
//===----------------------------------------------------------------------===//

// Sentinel for "not yet assigned" in mapping arrays.
#define LOOM_WRITER_ID_NONE UINT32_MAX

// External string entry: a string from a vtable B-string, not in
// the module's string table.
typedef struct loom_bytecode_external_string_t {
  iree_string_view_t view;
  uint32_t writer_id;
} loom_bytecode_external_string_t;

// Op name entry in the numbering context.
typedef struct loom_bytecode_op_entry_t {
  loom_op_kind_t kind;
  uint32_t writer_op_id;
  uint32_t string_writer_id;
} loom_bytecode_op_entry_t;

typedef struct loom_bytecode_numbering_t {
  const loom_module_t* module;
  iree_arena_allocator_t* arena;

  // Ordered list of all strings for the STRINGS section.
  iree_string_view_t* string_entries;
  iree_host_size_t string_count;
  iree_host_size_t string_capacity;

  // Fast lookup: module string_id → writer string_id.
  // Parallel array sized module->strings.count.
  uint32_t* module_string_map;

  // External strings not in the module table.
  loom_bytecode_external_string_t* external_strings;
  iree_host_size_t external_string_count;
  iree_host_size_t external_string_capacity;

  // Type mapping: module type index → writer type_id.
  uint32_t* type_map;
  // Writer type_id → module type index (for section writing).
  iree_host_size_t* type_order;
  iree_host_size_t type_count;
  iree_host_size_t type_order_capacity;

  // Op name registry.
  loom_bytecode_op_entry_t* op_entries;
  iree_host_size_t op_count;
  iree_host_size_t op_capacity;
} loom_bytecode_numbering_t;

// Initializes the numbering context. All allocations come from |arena|,
// which the caller owns. No individual frees needed — the arena handles
// bulk deallocation.
static iree_status_t loom_bytecode_numbering_initialize(
    loom_bytecode_numbering_t* numbering, const loom_module_t* module,
    iree_arena_allocator_t* arena) {
  memset(numbering, 0, sizeof(*numbering));
  numbering->module = module;
  numbering->arena = arena;

  // Module string map: parallel array for O(1) module_string_id → writer_id.
  if (module->strings.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->strings.count, sizeof(uint32_t),
        (void**)&numbering->module_string_map));
    memset(numbering->module_string_map, 0xFF,
           module->strings.count * sizeof(uint32_t));
  }

  // Type map: parallel array for O(1) module_type_index → writer_type_id.
  if (module->types.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, module->types.count, sizeof(uint32_t),
                                  (void**)&numbering->type_map));
    memset(numbering->type_map, 0xFF, module->types.count * sizeof(uint32_t));
  }

  return iree_ok_status();
}

// Appends a string_view to the ordered string list, growing if needed.
static iree_status_t loom_bytecode_numbering_append_string(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  if (numbering->string_count >= numbering->string_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->string_count, /*minimum_capacity=*/16,
        sizeof(iree_string_view_t), &numbering->string_capacity,
        (void**)&numbering->string_entries));
  }
  if (numbering->string_count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode string count exceeds format maximum "
                            "(16M)");
  }
  uint32_t id = (uint32_t)numbering->string_count;
  numbering->string_entries[numbering->string_count++] = view;
  *out_writer_id = id;
  return iree_ok_status();
}

// Interns a module string by its module string_id. Returns writer string ID.
static iree_status_t loom_bytecode_numbering_intern_module_string(
    loom_bytecode_numbering_t* numbering, loom_string_id_t string_id,
    uint32_t* out_writer_id) {
  if (string_id == LOOM_STRING_ID_INVALID) {
    *out_writer_id = 0;
    return iree_ok_status();
  }
  if (string_id >= numbering->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "string_id %u out of range (module has %" PRIhsz
                            " strings)",
                            string_id, numbering->module->strings.count);
  }
  if (numbering->module_string_map[string_id] != LOOM_WRITER_ID_NONE) {
    *out_writer_id = numbering->module_string_map[string_id];
    return iree_ok_status();
  }
  iree_string_view_t view = numbering->module->strings.entries[string_id];
  uint32_t writer_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_append_string(numbering, view, &writer_id));
  numbering->module_string_map[string_id] = writer_id;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Interns an arbitrary string_view (for vtable names not in the module).
static iree_status_t loom_bytecode_numbering_intern_string_view(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  // Check external strings first (linear scan, small list).
  for (iree_host_size_t i = 0; i < numbering->external_string_count; ++i) {
    if (iree_string_view_equal(numbering->external_strings[i].view, view)) {
      *out_writer_id = numbering->external_strings[i].writer_id;
      return iree_ok_status();
    }
  }
  // Also check if it happens to match a module string.
  for (iree_host_size_t i = 0; i < numbering->module->strings.count; ++i) {
    if (iree_string_view_equal(numbering->module->strings.entries[i], view) &&
        numbering->module_string_map[i] != LOOM_WRITER_ID_NONE) {
      *out_writer_id = numbering->module_string_map[i];
      return iree_ok_status();
    }
  }
  // New external string.
  uint32_t writer_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_append_string(numbering, view, &writer_id));
  if (numbering->external_string_count >= numbering->external_string_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->external_string_count,
        /*minimum_capacity=*/16, sizeof(loom_bytecode_external_string_t),
        &numbering->external_string_capacity,
        (void**)&numbering->external_strings));
  }
  numbering->external_strings[numbering->external_string_count++] =
      (loom_bytecode_external_string_t){.view = view, .writer_id = writer_id};
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Finds the module type table index for a given type.
static iree_host_size_t loom_bytecode_find_type_index(
    const loom_module_t* module, loom_type_t type) {
  for (iree_host_size_t i = 0; i < module->types.count; ++i) {
    // Types are interned by value: compare header + encoding fields.
    // For inline dims (rank <= 2), the 24-byte comparison works.
    // For overflow dims, we compare header + encoding_id and dims pointer.
    if (memcmp(&module->types.entries[i], &type, sizeof(loom_type_t)) == 0) {
      return i;
    }
  }
  return SIZE_MAX;
}

// Forward declaration for recursive type interning.
static iree_status_t loom_bytecode_numbering_intern_type(
    loom_bytecode_numbering_t* numbering, loom_type_t type,
    uint32_t* out_writer_id);

// Interns a type, recursing into sub-types first (topological order).
static iree_status_t loom_bytecode_numbering_intern_type(
    loom_bytecode_numbering_t* numbering, loom_type_t type,
    uint32_t* out_writer_id) {
  iree_host_size_t module_index =
      loom_bytecode_find_type_index(numbering->module, type);
  if (module_index == SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type not found in module type table");
  }
  if (numbering->type_map[module_index] != LOOM_WRITER_ID_NONE) {
    *out_writer_id = numbering->type_map[module_index];
    return iree_ok_status();
  }

  // Recurse into sub-types first (topological ordering).
  uint32_t unused_id = 0;
  loom_type_kind_t kind = loom_type_kind(type);
  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      // Element type is a scalar — intern it.
      loom_type_t element_type = loom_type_scalar(loom_type_element_type(type));
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, element_type, &unused_id));
      break;
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* func_data = loom_type_func_data(type);
      for (uint16_t i = 0; i < func_data->arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, func_data->types[i], &unused_id));
      }
      for (uint16_t i = 0; i < func_data->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, func_data->types[func_data->arg_count + i], &unused_id));
      }
      break;
    }
    case LOOM_TYPE_GROUP:
      // Group scope is serialized as a byte, not a string. No interning.
      break;
    case LOOM_TYPE_DIALECT: {
      // Intern the dialect type name string.
      loom_string_id_t name_id = loom_type_dialect_name_id(type);
      if (name_id < numbering->module->strings.count) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
            numbering, numbering->module->strings.entries[name_id],
            &unused_id));
      }
      // Recurse into type parameters.
      uint16_t param_count = loom_type_dialect_param_count(type);
      const loom_type_t* params = loom_type_dialect_params(type);
      for (uint16_t i = 0; i < param_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, params[i], &unused_id));
      }
      break;
    }
    default:
      break;
  }

  // Intern the parent type.
  if (numbering->type_count >= (1u << 16)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode type count exceeds format maximum (64K)");
  }
  uint32_t writer_id = numbering->type_count;
  numbering->type_map[module_index] = writer_id;
  if (numbering->type_count >= numbering->type_order_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->type_count, /*minimum_capacity=*/16,
        sizeof(iree_host_size_t), &numbering->type_order_capacity,
        (void**)&numbering->type_order));
  }
  numbering->type_order[numbering->type_count] = module_index;
  numbering->type_count++;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Interns an op kind. Returns writer op name ID.
static iree_status_t loom_bytecode_numbering_intern_op(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    uint32_t* out_writer_op_id) {
  // Check existing entries.
  for (uint32_t i = 0; i < numbering->op_count; ++i) {
    if (numbering->op_entries[i].kind == op->kind) {
      *out_writer_op_id = numbering->op_entries[i].writer_op_id;
      return iree_ok_status();
    }
  }
  // New op kind.
  iree_string_view_t name = loom_op_name(numbering->module, op);
  uint32_t string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, name, &string_writer_id));

  if (numbering->op_count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode op count exceeds format maximum (16M)");
  }
  uint32_t writer_op_id = (uint32_t)numbering->op_count;
  if (numbering->op_count >= numbering->op_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->op_count, /*minimum_capacity=*/16,
        sizeof(loom_bytecode_op_entry_t), &numbering->op_capacity,
        (void**)&numbering->op_entries));
  }
  numbering->op_entries[numbering->op_count++] = (loom_bytecode_op_entry_t){
      .kind = op->kind,
      .writer_op_id = writer_op_id,
      .string_writer_id = string_writer_id,
  };
  *out_writer_op_id = writer_op_id;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Value numbering (per-function)
//===----------------------------------------------------------------------===//

typedef struct loom_bytecode_value_numbering_t {
  uint32_t* map;              // module value_id → function-local number.
  uint32_t next_number;       // Next value number to assign.
  iree_host_size_t capacity;  // Size of |map| array.
} loom_bytecode_value_numbering_t;

// Resolves a module value_id to its function-local value number.
// Returns INVALID_ARGUMENT if the value_id is out of bounds or was
// never assigned a number (indicates a malformed IR graph).
static iree_status_t loom_bytecode_resolve_value_number(
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_value_id_t value_id, uint32_t* out_number) {
  if (value_id >= value_numbering->capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u exceeds value table capacity %" PRIhsz,
                            value_id, value_numbering->capacity);
  }
  uint32_t number = value_numbering->map[value_id];
  if (number == LOOM_WRITER_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u has no assigned value number "
                            "(undefined or not in scope)",
                            value_id);
  }
  *out_number = number;
  return iree_ok_status();
}

static void loom_bytecode_value_numbering_assign_region(
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_module_t* module, const loom_region_t* region) {
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    // Block arguments define values.
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_value_id_t value_id = loom_block_arg_id(block, arg_index);
      if (value_id < value_numbering->capacity &&
          value_numbering->map[value_id] == LOOM_WRITER_ID_NONE) {
        value_numbering->map[value_id] = value_numbering->next_number++;
      }
    }
    // Op results define values.
    for (uint16_t op_index = 0; op_index < block->op_count; ++op_index) {
      const loom_op_t* op = loom_block_const_op(block, op_index);
      if (op->flags & LOOM_OP_FLAG_DEAD) continue;
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        loom_value_id_t value_id = results[result_index];
        if (value_id < value_numbering->capacity &&
            value_numbering->map[value_id] == LOOM_WRITER_ID_NONE) {
          value_numbering->map[value_id] = value_numbering->next_number++;
        }
      }
      // Recurse into nested regions.
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        if (regions[region_index]) {
          loom_bytecode_value_numbering_assign_region(value_numbering, module,
                                                      regions[region_index]);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Numbering walk
//===----------------------------------------------------------------------===//
//
// Walks the module in the same order as the Python writer's
// _number_module to produce identical string/type/op ordering.

// Forward declarations for recursive numbering.
static iree_status_t loom_bytecode_number_region(
    loom_bytecode_numbering_t* numbering, const loom_region_t* region,
    uint32_t depth);
static iree_status_t loom_bytecode_number_operation(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op, uint32_t depth);

static iree_status_t loom_bytecode_number_attr_value(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor) {
  uint32_t unused_id = 0;
  switch (attr.kind) {
    case LOOM_ATTR_STRING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &unused_id));
      break;
    }
    case LOOM_ATTR_ENUM: {
      if (descriptor && descriptor->enum_case_names) {
        uint8_t case_index = (uint8_t)attr.raw;
        loom_bstring_t case_name = descriptor->enum_case_names[case_index];
        if (case_name) {
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
              numbering, loom_bstring_view(case_name), &unused_id));
        }
      }
      break;
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target_symbol =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target_symbol->name_id, &unused_id));
      }
      break;
    }
    case LOOM_ATTR_DICT: {
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, attr.dict_entries[i].name_id, &unused_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_number_attr_value(
            numbering, attr.dict_entries[i].value, NULL));
      }
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_function(
    loom_bytecode_numbering_t* numbering, loom_func_like_t func_like) {
  uint32_t unused_id = 0;

  // Arg names and types.
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_value_id_t value_id = arg_ids[i];
    if (value_id < numbering->module->values.count) {
      loom_string_id_t name_id =
          numbering->module->values.entries[value_id].name_id;
      if (name_id != LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, name_id, &unused_id));
      }
    }
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_type_t arg_type = numbering->module->values.entries[arg_ids[i]].type;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_numbering_intern_type(numbering, arg_type, &unused_id));
  }

  // Result types (recursive).
  uint16_t result_count = func_like.op->result_count;
  const loom_value_id_t* result_ids = loom_op_const_results(func_like.op);
  for (uint16_t i = 0; i < result_count; ++i) {
    loom_type_t result_type =
        numbering->module->values.entries[result_ids[i]].type;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, result_type, &unused_id));
  }

  // Predicate value name strings.
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(func_like, &predicate_count);
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    for (uint8_t arg_index = 0; arg_index < predicate->arg_count; ++arg_index) {
      if (predicate->arg_tags[arg_index] == LOOM_PRED_ARG_VALUE) {
        // The value is a string_id referencing a dim name.
        loom_string_id_t name_id = (loom_string_id_t)predicate->args[arg_index];
        if (name_id != LOOM_STRING_ID_INVALID) {
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
              numbering, name_id, &unused_id));
        }
      }
    }
  }

  // Body (recursive).
  loom_region_t* body = loom_func_like_body(func_like);
  if (body) {
    IREE_RETURN_IF_ERROR(loom_bytecode_number_region(numbering, body, 0));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_region(
    loom_bytecode_numbering_t* numbering, const loom_region_t* region,
    uint32_t depth) {
  if (depth >= LOOM_BYTECODE_MAX_REGION_DEPTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region nesting exceeds maximum depth %d",
                            LOOM_BYTECODE_MAX_REGION_DEPTH);
  }
  uint32_t unused_id = 0;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    // Block label.
    if (block->label_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, block->label_id, &unused_id));
    }

    // Block arg names and types.
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_value_id_t value_id = loom_block_arg_id(block, arg_index);
      const loom_value_t* value = &numbering->module->values.entries[value_id];
      if (value->name_id != LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, value->name_id, &unused_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, value->type, &unused_id));
    }

    // Operations.
    for (uint16_t op_index = 0; op_index < block->op_count; ++op_index) {
      const loom_op_t* op = loom_block_const_op(block, op_index);
      if (op->flags & LOOM_OP_FLAG_DEAD) continue;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_operation(numbering, op, depth));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_operation(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op, uint32_t depth) {
  uint32_t unused_id = 0;

  // Op name (into both op table and string table).
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &unused_id));

  // Result names and types.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_t* value = &numbering->module->values.entries[results[i]];
    if (value->name_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, value->name_id, &unused_id));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, value->type, &unused_id));
  }

  // Attribute keys and values.
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(numbering->module->context, op->kind);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    // Attribute key name (from vtable descriptor).
    if (vtable && vtable->attr_descriptors) {
      iree_string_view_t key_name =
          loom_attr_descriptor_name(&vtable->attr_descriptors[i]);
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
          numbering, key_name, &unused_id));
    }
    // Attribute value strings.
    const loom_attr_descriptor_t* descriptor =
        (vtable && vtable->attr_descriptors) ? &vtable->attr_descriptors[i]
                                             : NULL;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attrs[i], descriptor));
  }

  // Nested regions.
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_region(numbering, regions[i], depth + 1));
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type kind mapping (C enum → bytecode kind byte)
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_type_kind_byte(loom_type_kind_t kind,
                                                  uint8_t* out_byte) {
  switch (kind) {
    case LOOM_TYPE_NONE:
      *out_byte = LOOM_BYTECODE_TYPE_NONE;
      return iree_ok_status();
    case LOOM_TYPE_SCALAR:
      *out_byte = LOOM_BYTECODE_TYPE_SCALAR;
      return iree_ok_status();
    case LOOM_TYPE_TILE:
      *out_byte = LOOM_BYTECODE_TYPE_TILE;
      return iree_ok_status();
    case LOOM_TYPE_TENSOR:
      *out_byte = LOOM_BYTECODE_TYPE_TENSOR;
      return iree_ok_status();
    case LOOM_TYPE_VECTOR:
      *out_byte = LOOM_BYTECODE_TYPE_VECTOR;
      return iree_ok_status();
    case LOOM_TYPE_VIEW:
      *out_byte = LOOM_BYTECODE_TYPE_VIEW;
      return iree_ok_status();
    case LOOM_TYPE_BUFFER:
      *out_byte = LOOM_BYTECODE_TYPE_BUFFER;
      return iree_ok_status();
    case LOOM_TYPE_GROUP:
      *out_byte = LOOM_BYTECODE_TYPE_GROUP;
      return iree_ok_status();
    case LOOM_TYPE_FUNCTION:
      *out_byte = LOOM_BYTECODE_TYPE_FUNCTION;
      return iree_ok_status();
    case LOOM_TYPE_DIALECT:
      *out_byte = LOOM_BYTECODE_TYPE_DIALECT;
      return iree_ok_status();
    case LOOM_TYPE_ENCODING:
      *out_byte = LOOM_BYTECODE_TYPE_ENCODING;
      return iree_ok_status();
    case LOOM_TYPE_POOL:
      *out_byte = LOOM_BYTECODE_TYPE_POOL;
      return iree_ok_status();
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown type kind %d",
                          (int)kind);
}

//===----------------------------------------------------------------------===//
// Section writers
//===----------------------------------------------------------------------===//

// Writes the IR section: function bodies streamed through the page writer.
// Returns per-symbol (offset, length) pairs for the SYMBOLS section.
typedef struct loom_bytecode_ir_offset_t {
  uint64_t offset;
  uint32_t length;
} loom_bytecode_ir_offset_t;

// Forward declarations for recursive IR writing.
static iree_status_t loom_bytecode_write_region(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region, uint32_t depth);

static iree_status_t loom_bytecode_write_value_def(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_value_t* value) {
  uint32_t name_writer_id = 0;
  if (value->name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, value->name_id, &name_writer_id));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, name_writer_id));

  uint32_t type_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
      numbering, value->type, &type_writer_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, type_writer_id));

  // Dim bindings: count dynamic dims, then emit value refs.
  loom_type_t type = value->type;
  uint8_t rank = loom_type_rank(type);
  uint32_t dynamic_count = 0;
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) ++dynamic_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, dynamic_count));
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t packed = loom_type_dim(type, i);
    if (!loom_dim_is_dynamic(packed)) continue;
    loom_value_id_t dim_value_id = loom_dim_value_id(packed);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, dim_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_svarint(writer, (int64_t)value_number));
  }

  // Encoding binding.
  if (loom_type_has_ssa_encoding(type)) {
    uint16_t encoding_value_id = loom_type_encoding_value_id(type);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, encoding_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, 1 + value_number));
  } else {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(writer, 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_attr_value(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor) {
  switch (attr.kind) {
    case LOOM_ATTR_I64: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 0));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_svarint(writer, attr.i64));
      break;
    }
    case LOOM_ATTR_F64: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 1));
      uint8_t bytes[8];
      memcpy(bytes, &attr.f64, 8);
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, bytes, 8));
      break;
    }
    case LOOM_ATTR_STRING: {
      uint32_t string_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &string_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 2));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      break;
    }
    case LOOM_ATTR_BOOL: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 3));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, attr.raw ? 1 : 0));
      break;
    }
    case LOOM_ATTR_ENUM: {
      uint8_t case_index = (uint8_t)attr.raw;
      uint32_t string_writer_id = 0;
      if (descriptor && descriptor->enum_case_names &&
          descriptor->enum_case_names[case_index]) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
            numbering,
            loom_bstring_view(descriptor->enum_case_names[case_index]),
            &string_writer_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 4));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      break;
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 5));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_svarint(writer, attr.i64_array[i]));
      }
      break;
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      uint32_t string_writer_id = 0;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target->name_id, &string_writer_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 6));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      break;
    }
    case LOOM_ATTR_TYPE: {
      uint32_t type_writer_id = 0;
      if (attr.type_id < numbering->module->types.count) {
        loom_type_t type = numbering->module->types.entries[attr.type_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, type, &type_writer_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 7));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, type_writer_id));
      break;
    }
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 8));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(writer, predicate->kind));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(writer, predicate->arg_count));
        for (uint8_t arg_index = 0; arg_index < predicate->arg_count;
             ++arg_index) {
          uint8_t tag = predicate->arg_tags[arg_index];
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, tag));
          switch (tag) {
            case LOOM_PRED_ARG_VALUE: {
              loom_string_id_t name_id =
                  (loom_string_id_t)predicate->args[arg_index];
              uint32_t string_writer_id = 0;
              IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
                  numbering, name_id, &string_writer_id));
              IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                  writer, string_writer_id));
              break;
            }
            case LOOM_PRED_ARG_CONST: {
              IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_svarint(
                  writer, predicate->args[arg_index]));
              break;
            }
            default:
              return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "unknown predicate arg tag %d", (int)tag);
          }
        }
      }
      break;
    }
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 9));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      // Dict attrs are stored canonically at IR construction time; emit that
      // order directly so logically equivalent dicts produce stable bytes.
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_named_attr_t* entry = &attr.dict_entries[i];
        uint32_t key_writer_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, entry->name_id, &key_writer_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
            writer, numbering, value_numbering, entry->value, NULL));
      }
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 10));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.encoding_id));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported attribute kind %d", (int)attr.kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_operation(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering, const loom_op_t* op,
    uint32_t depth) {
  const loom_module_t* module = numbering->module;
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, op->kind);
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "op kind 0x%04x has no registered vtable",
                            (unsigned)op->kind);
  }

  // Op kind ID.
  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, writer_op_id));

  // Flags byte.
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 0));

  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->location));

  // Operands.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->operand_count));
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, operands[i], &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, value_number));
  }

  // Results.
  const loom_value_id_t* results = loom_op_const_results(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->result_count));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_t* value = &module->values.entries[results[i]];
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(writer, numbering,
                                                       value_numbering, value));
  }

  // Tied results.
  const loom_tied_result_t* tied = loom_op_tied_results(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->tied_result_count));
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, tied[i].result_index));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, tied[i].operand_index));
  }

  // Attributes: each has a key name from the vtable descriptor and a
  // tagged value from the op's trailing data.
  const loom_attribute_t* attrs = loom_op_attrs(op);
  if (op->attribute_count > 0 && !vtable->attr_descriptors) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "op kind 0x%04x has %d attributes but no attr_descriptors in vtable",
        (unsigned)op->kind, (int)op->attribute_count);
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->attribute_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    iree_string_view_t key_name =
        loom_attr_descriptor_name(&vtable->attr_descriptors[i]);
    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, key_name, &key_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
    // Value.
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
        writer, numbering, value_numbering, attrs[i], descriptor));
  }

  // Regions.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->region_count));
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_bytecode_write_region(
          writer, numbering, value_numbering, regions[i], depth + 1));
    } else {
      // Empty region: 0 blocks.
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(writer, 0));
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_block(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_block_t* block, uint32_t depth) {
  const loom_module_t* module = numbering->module;

  // Label.
  bool has_label = block->label_id != LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_u8(writer, has_label ? 1 : 0));
  if (has_label) {
    uint32_t label_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, block->label_id, &label_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, label_writer_id));
  }

  // Block args.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, block->arg_count));
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t value_id = loom_block_arg_id(block, i);
    const loom_value_t* value = &module->values.entries[value_id];
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(writer, numbering,
                                                       value_numbering, value));
  }

  // Ops (skip dead ops, matching Python which checks is_dead).
  uint16_t live_op_count = 0;
  for (uint16_t i = 0; i < block->op_count; ++i) {
    if (!iree_any_bit_set(loom_block_const_op(block, i)->flags,
                          LOOM_OP_FLAG_DEAD)) {
      ++live_op_count;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, live_op_count));
  for (uint16_t i = 0; i < block->op_count; ++i) {
    const loom_op_t* op = loom_block_const_op(block, i);
    if (op->flags & LOOM_OP_FLAG_DEAD) continue;
    IREE_RETURN_IF_ERROR(loom_bytecode_write_operation(
        writer, numbering, value_numbering, op, depth));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_region(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region, uint32_t depth) {
  if (depth >= LOOM_BYTECODE_MAX_REGION_DEPTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region nesting exceeds maximum depth %d",
                            LOOM_BYTECODE_MAX_REGION_DEPTH);
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, region->block_count));
  for (uint16_t i = 0; i < region->block_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_write_block(writer, numbering, value_numbering,
                                  loom_region_const_block(region, i), depth));
  }
  return iree_ok_status();
}

// Writes the IR section and returns per-symbol offsets.
static iree_status_t loom_bytecode_write_ir_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering,
    loom_bytecode_ir_offset_t* ir_offsets) {
  const loom_module_t* module = numbering->module;
  iree_host_size_t section_start = page_writer->total_written;

  // Allocate value numbering map from the arena. Allocated once,
  // reused per-function via memset. The arena never frees individual
  // allocations so there's nothing to clean up.
  loom_bytecode_value_numbering_t value_numbering = {0};
  value_numbering.capacity = module->values.count;
  if (value_numbering.capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        numbering->arena, value_numbering.capacity, sizeof(uint32_t),
        (void**)&value_numbering.map));
  }

  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    if (!loom_symbol_kind_is_function_like(symbol->kind) ||
        !symbol->defining_op) {
      ir_offsets[symbol_index].offset = 0;
      ir_offsets[symbol_index].length = 0;
      continue;
    }
    loom_func_like_t func_like =
        loom_func_like_cast(module, symbol->defining_op);
    loom_region_t* body = loom_func_like_body(func_like);
    if (!loom_func_like_isa(func_like) || !body) {
      ir_offsets[symbol_index].offset = 0;
      ir_offsets[symbol_index].length = 0;
      continue;
    }

    // Intern function signature (matching Python walk order).
    IREE_RETURN_IF_ERROR(loom_bytecode_number_function(numbering, func_like));

    // Reset value numbering for this function.
    if (value_numbering.map) {
      memset(value_numbering.map, 0xFF,
             value_numbering.capacity * sizeof(uint32_t));
    }
    value_numbering.next_number = 0;

    // Assign value numbers.
    loom_bytecode_value_numbering_assign_region(&value_numbering, module, body);

    // Write the function body.
    iree_host_size_t body_start = page_writer->total_written;
    IREE_RETURN_IF_ERROR(loom_bytecode_write_region(page_writer, numbering,
                                                    &value_numbering, body, 0));
    iree_host_size_t body_length = page_writer->total_written - body_start;

    if (body_length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "function body length %" PRIhsz
                              " exceeds uint32 maximum",
                              body_length);
    }
    ir_offsets[symbol_index].offset = body_start - section_start;
    ir_offsets[symbol_index].length = (uint32_t)body_length;
  }

  return iree_ok_status();
}

// Writes the function-like metadata fields for a single symbol entry.
// Called from loom_bytecode_write_symbols_section for each function-like
// symbol that has a defining op.
static iree_status_t loom_bytecode_write_func_metadata(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, loom_func_like_t func_like,
    loom_bytecode_ir_offset_t ir_offset) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_u8(builder, loom_func_like_cc(func_like)));

  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  uint16_t result_count = func_like.op->result_count;
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, arg_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, result_count));

  // Arg types.
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_type_t arg_type = module->values.entries[arg_ids[i]].type;
    uint32_t type_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, arg_type, &type_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, type_writer_id));
  }

  // Result types with tied info.
  const loom_value_id_t* result_ids = loom_op_const_results(func_like.op);
  const loom_tied_result_t* tied_results = loom_op_tied_results(func_like.op);
  uint16_t tied_result_count = func_like.op->tied_result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    bool is_tied = false;
    uint16_t tied_operand_index = 0;
    for (uint16_t t = 0; t < tied_result_count; ++t) {
      if (tied_results[t].result_index == i) {
        is_tied = true;
        tied_operand_index = tied_results[t].operand_index;
        break;
      }
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, is_tied ? 1 : 0));
    loom_type_t result_type = module->values.entries[result_ids[i]].type;
    uint32_t type_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, result_type, &type_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, type_writer_id));
    if (is_tied) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, tied_operand_index));
    }
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, tied_result_count));

  // Predicates.
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(func_like, &predicate_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, predicate_count));
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->arg_count));
    for (uint8_t arg_index = 0; arg_index < predicate->arg_count; ++arg_index) {
      uint8_t tag = predicate->arg_tags[arg_index];
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, tag));
      switch (tag) {
        case LOOM_PRED_ARG_VALUE: {
          loom_string_id_t name_id =
              (loom_string_id_t)predicate->args[arg_index];
          uint32_t string_writer_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
              numbering, name_id, &string_writer_id));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_emit_uvarint(builder, string_writer_id));
          break;
        }
        case LOOM_PRED_ARG_CONST: {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_emit_svarint(builder, predicate->args[arg_index]));
          break;
        }
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unknown predicate arg tag %d", (int)tag);
      }
    }
  }

  // Template/ukernel dispatch metadata: the name of the op kind this function
  // provides an implementation for, and its matching priority. Written only
  // for templates and ukernels; absent for def/decl.
  loom_string_id_t implements_id = loom_func_like_implements(func_like);
  if (implements_id != LOOM_STRING_ID_INVALID) {
    uint32_t implements_string_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, implements_id, &implements_string_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_uvarint(builder, implements_string_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(
        builder, (uint64_t)loom_func_like_priority(func_like)));
  }

  bool has_body = ir_offset.length > 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, has_body ? 1 : 0));
  if (has_body) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, ir_offset.offset));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u32_le(builder, ir_offset.length));
  }

  return iree_ok_status();
}

// Writes the SYMBOLS section into a string builder (for offset table patching).
static iree_status_t loom_bytecode_write_symbols_section(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_ir_offset_t* ir_offsets) {
  const loom_module_t* module = numbering->module;

  // Classify symbols.
  uint32_t export_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (module->symbols.entries[i].flags & LOOM_SYMBOL_FLAG_PUBLIC) {
      ++export_count;
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, module->symbols.count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, 0));  // imports
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, export_count));

  // Reserve export offset table (patched after writing entries).
  iree_host_size_t export_table_offset = iree_string_builder_size(builder);
  for (uint32_t i = 0; i < export_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, 0));
  }

  iree_host_size_t entries_start = iree_string_builder_size(builder);
  uint32_t export_index = 0;

  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];

    // Track export offset.
    if (symbol->flags & LOOM_SYMBOL_FLAG_PUBLIC) {
      uint64_t entry_offset = iree_string_builder_size(builder) - entries_start;
      loom_bytecode_patch_u64_le(
          builder, export_table_offset + (iree_host_size_t)export_index * 8,
          entry_offset);
      ++export_index;
    }

    // Name.
    uint32_t name_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, symbol->name_id, &name_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, name_writer_id));

    // Kind.
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, symbol->kind));

    // Visibility.
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(
        builder, (symbol->flags & LOOM_SYMBOL_FLAG_PUBLIC) ? 0 : 1));

    // Flags.
    uint16_t bytecode_flags = (symbol->flags & LOOM_SYMBOL_FLAG_PUBLIC)
                                  ? LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC
                                  : 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u16_le(builder, bytecode_flags));

    // Function metadata.
    if (loom_symbol_kind_is_function_like(symbol->kind) &&
        symbol->defining_op) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(func_like)) {
        IREE_RETURN_IF_ERROR(loom_bytecode_write_func_metadata(
            builder, numbering, module, func_like, ir_offsets[symbol_index]));
      }
    }
  }

  return iree_ok_status();
}

// Writes the STRINGS section through the page writer.
static iree_status_t loom_bytecode_write_strings_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->string_count));
  for (iree_host_size_t i = 0; i < numbering->string_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_string(
        page_writer, numbering->string_entries[i]));
  }
  return iree_ok_status();
}

// Writes the SOURCES section through the page writer.
static iree_status_t loom_bytecode_write_sources_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module) {
  iree_host_size_t source_count =
      module->context ? module->context->sources.count : 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, source_count));
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_string(
        page_writer, module->context->sources.entries[i]));
  }
  return iree_ok_status();
}

// Writes the TYPES section through the page writer.
static iree_status_t loom_bytecode_write_types_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->type_count));

  for (uint32_t writer_type_id = 0; writer_type_id < numbering->type_count;
       ++writer_type_id) {
    iree_host_size_t module_index = numbering->type_order[writer_type_id];
    loom_type_t type = module->types.entries[module_index];
    loom_type_kind_t kind = loom_type_kind(type);

    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_kind_byte(kind, &kind_byte));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, kind_byte));

    switch (kind) {
      case LOOM_TYPE_NONE:
        break;
      case LOOM_TYPE_SCALAR: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_element_type(type)));
        break;
      }
      case LOOM_TYPE_TILE:
      case LOOM_TYPE_TENSOR:
      case LOOM_TYPE_VECTOR:
      case LOOM_TYPE_VIEW: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_element_type(type)));
        uint8_t rank = loom_type_rank(type);
        if (kind == LOOM_TYPE_VECTOR) {
          if (rank == 0) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "vector types must have rank >= 1");
          }
          if (type.encoding_id != 0 || type.encoding_flags != 0) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "vector types must not carry encoding or layout attachments");
          }
        }
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(page_writer, rank));
        // Encoding.
        if (loom_type_has_ssa_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_uvarint(page_writer, 0));
        } else if (loom_type_has_static_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, type.encoding_id));
        } else {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_uvarint(page_writer, 0));
        }
        // Dims.
        for (uint8_t i = 0; i < rank; ++i) {
          if (loom_type_dim_is_dynamic_at(type, i)) {
            IREE_RETURN_IF_ERROR(
                loom_bytecode_page_writer_write_u8(page_writer, 1));
          } else {
            IREE_RETURN_IF_ERROR(
                loom_bytecode_page_writer_write_u8(page_writer, 0));
            IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                page_writer, (uint64_t)loom_type_dim_static_size_at(type, i)));
          }
        }
        break;
      }
      case LOOM_TYPE_GROUP: {
        loom_group_scope_t scope = loom_type_group_scope(type);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(page_writer, (uint8_t)scope));
        break;
      }
      case LOOM_TYPE_FUNCTION: {
        const loom_func_type_data_t* func_data = loom_type_func_data(type);
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, func_data->arg_count));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, func_data->result_count));
        for (uint16_t i = 0; i < func_data->arg_count + func_data->result_count;
             ++i) {
          uint32_t sub_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
              numbering, func_data->types[i], &sub_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, sub_type_id));
        }
        break;
      }
      case LOOM_TYPE_DIALECT: {
        loom_string_id_t name_id = loom_type_dialect_name_id(type);
        uint32_t name_writer_id = 0;
        if (name_id < numbering->module->strings.count) {
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
              numbering, numbering->module->strings.entries[name_id],
              &name_writer_id));
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, name_writer_id));
        uint16_t param_count = loom_type_dialect_param_count(type);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_uvarint(page_writer, param_count));
        const loom_type_t* params = loom_type_dialect_params(type);
        for (uint16_t i = 0; i < param_count; ++i) {
          uint32_t param_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
              numbering, params[i], &param_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, param_type_id));
        }
        break;
      }
      case LOOM_TYPE_ENCODING:
        // No additional data.
        break;
      case LOOM_TYPE_BUFFER:
        // No additional data.
        break;
      case LOOM_TYPE_POOL: {
        uint64_t dim = loom_type_dim(type, 0);
        if (loom_dim_is_dynamic(dim)) {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_u8(page_writer, 1));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_u8(page_writer, 0));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, (uint64_t)loom_dim_static_size(dim)));
        }
        break;
      }
      default:
        // Unreachable: loom_bytecode_type_kind_byte above rejects
        // unknown kinds before we get here.
        break;
    }
  }

  return iree_ok_status();
}

// Writes the ENCODINGS section: encoding family registry + instances.
static iree_status_t loom_bytecode_write_encodings_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;

  // Build the encoding family registry from unique encoding names.
  // Small (typically <10 families), so linear dedup is fine.
  typedef struct {
    loom_string_id_t name_id;
    uint32_t writer_string_id;
  } encoding_family_entry_t;
  encoding_family_entry_t family_entries[256];
  iree_host_size_t family_count = 0;

  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    loom_string_id_t name_id = module->encodings.entries[i].name_id;
    bool found = false;
    for (iree_host_size_t k = 0; k < family_count; ++k) {
      if (family_entries[k].name_id == name_id) {
        found = true;
        break;
      }
    }
    if (!found) {
      if (family_count >= 256) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "more than 256 unique encoding families");
      }
      uint32_t writer_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, name_id, &writer_string_id));
      family_entries[family_count++] = (encoding_family_entry_t){
          .name_id = name_id, .writer_string_id = writer_string_id};
    }
  }

  // Encoding family count and family name string IDs.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, family_count));
  for (iree_host_size_t k = 0; k < family_count; ++k) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, family_entries[k].writer_string_id));
  }

  // Encoding instances.
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, module->encodings.count));
  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    const loom_encoding_t* encoding = &module->encodings.entries[i];

    // Find the family index for this encoding's name.
    uint32_t family_index = 0;
    for (iree_host_size_t k = 0; k < family_count; ++k) {
      if (family_entries[k].name_id == encoding->name_id) {
        family_index = (uint32_t)k;
        break;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(page_writer, family_index));

    // Alias string ID (0 = no alias).
    uint32_t alias_writer_id = 0;
    if (encoding->alias_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, encoding->alias_id, &alias_writer_id));
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(page_writer, alias_writer_id));

    // Parameters as structured named attributes, using the same
    // attribute serialization as the IR section.
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, encoding->attribute_count));
    for (uint8_t p = 0; p < encoding->attribute_count; ++p) {
      const loom_named_attr_t* attr = &encoding->attributes[p];
      uint32_t param_name_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr->name_id, &param_name_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, param_name_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
          page_writer, numbering, NULL, attr->value, NULL));
    }
  }

  return iree_ok_status();
}

// Writes the OPS section through the page writer.
static iree_status_t loom_bytecode_write_ops_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->op_count));
  for (uint32_t i = 0; i < numbering->op_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, numbering->op_entries[i].string_writer_id));
  }
  return iree_ok_status();
}

// Writes the LOCATIONS section: location_count + per-entry serialization.
static iree_status_t loom_bytecode_write_locations_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module) {
  iree_host_size_t location_count = module->locations.count;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, location_count));
  for (iree_host_size_t i = 0; i < location_count; ++i) {
    const loom_location_entry_t* entry = &module->locations.entries[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, (uint8_t)entry->kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, entry->flags));
    switch (entry->kind) {
      case LOOM_LOCATION_NONE:
        break;
      case LOOM_LOCATION_FILE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.start_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.start_col));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.end_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.end_col));
        break;
      }
      case LOOM_LOCATION_FUSED: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->fused.count));
        for (uint32_t c = 0; c < entry->fused.count; ++c) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, entry->fused.children[c]));
        }
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->opaque.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->opaque.data_length));
        if (entry->opaque.data_length > 0) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
              page_writer, entry->opaque.data, entry->opaque.data_length));
        }
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown location kind %d", (int)entry->kind);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Top-level writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_validate_module(
    const loom_module_t* module) {
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (needed for op vtables)");
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->kind == LOOM_SYMBOL_GLOBAL) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "GLOBAL symbols not yet supported");
    }
    if (symbol->kind == LOOM_SYMBOL_EXECUTABLE) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "EXECUTABLE symbols not yet supported");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_module(
    const loom_module_t* module, iree_io_stream_t* stream,
    const loom_bytecode_write_options_t* options,
    iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, loom_bytecode_validate_module(module));

  // Check stream capabilities.
  iree_io_stream_mode_t mode = iree_io_stream_mode(stream);
  if (!(mode & IREE_IO_STREAM_MODE_WRITABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not writable");
  }
  if (!(mode & IREE_IO_STREAM_MODE_SEEKABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not seekable (needed for directory "
                            "patching)");
  }

  iree_string_view_t producer = (options && options->producer.size > 0)
                                    ? options->producer
                                    : IREE_SV(LOOM_BYTECODE_DEFAULT_PRODUCER);
  loom_bytecode_file_flags_t file_flags = options ? options->flags : 0;

  // Temporary arena for all working memory. All numbering tables,
  // value maps, and scratch allocations come from here. Deinitialized
  // at the end, returning all blocks to the shared pool in O(1).
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  // Initialize the page writer.
  loom_bytecode_page_writer_t page_writer;
  loom_bytecode_page_writer_initialize(&page_writer, stream);

  // Initialize numbering context.
  loom_bytecode_numbering_t numbering;
  iree_status_t status =
      loom_bytecode_numbering_initialize(&numbering, module, &arena);

  // Pass 1: Number module metadata (names, sources, symbol names).
  // Function signatures and bodies are numbered during IR section writing.
  if (iree_status_is_ok(status)) {
    uint32_t unused_id = 0;
    status = loom_bytecode_numbering_intern_module_string(
        &numbering, module->name_id, &unused_id);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < module->symbols.count && iree_status_is_ok(status); ++i) {
      uint32_t unused_id = 0;
      status = loom_bytecode_numbering_intern_module_string(
          &numbering, module->symbols.entries[i].name_id, &unused_id);
    }
  }

  // File header: magic, version, flags, module count, producer string.
  iree_string_view_t module_name = module->strings.entries[module->name_id];

  if (iree_status_is_ok(status)) {
    // Magic.
    status = loom_bytecode_page_writer_write(&page_writer, LOOM_BYTECODE_MAGIC,
                                             LOOM_BYTECODE_MAGIC_LENGTH);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer,
                                                LOOM_BYTECODE_FORMAT_VERSION);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer, file_flags);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    1);  // module_count
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u32_le(
        &page_writer, (uint32_t)module_name.size);  // string_pool_length
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // reserved
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_null_terminated_string(
        &page_writer, producer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module directory entry. module_offset and module_length are
  // written as placeholders and patched after all sections are written.
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // name_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    (uint16_t)module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    0);  // module_flags
  }
  // module_offset placeholder: we'll patch this as part of the directory entry.
  iree_host_size_t module_dir_offset_position = 0;
  if (iree_status_is_ok(status)) {
    module_dir_offset_position = page_writer.total_written;
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_length
  }

  // File string pool: module name(s).
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write(&page_writer, module_name.data,
                                             module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module data starts at this offset.
  iree_host_size_t module_start = page_writer.total_written;

  // Section directory placeholder — patched after all sections are written.
  iree_host_size_t section_dir_patch_position = page_writer.total_written;
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_zeros(
        &page_writer, LOOM_BYTECODE_SECTION_COUNT *
                          sizeof(loom_bytecode_section_dir_entry_t));
  }

  // Track section offsets and lengths.
  uint64_t section_offsets[LOOM_BYTECODE_SECTION_COUNT] = {0};
  uint64_t section_lengths[LOOM_BYTECODE_SECTION_COUNT] = {0};

  // Allocate IR offset tracking from the arena.
  loom_bytecode_ir_offset_t* ir_offsets = NULL;
  if (iree_status_is_ok(status) && module->symbols.count > 0) {
    status = iree_arena_allocate_array(&arena, module->symbols.count,
                                       sizeof(loom_bytecode_ir_offset_t),
                                       (void**)&ir_offsets);
    if (iree_status_is_ok(status)) {
      memset(ir_offsets, 0,
             module->symbols.count * sizeof(loom_bytecode_ir_offset_t));
    }
  }

  // IR section: function bodies streamed through the page writer.
  // Written first so the numbering tables grow as entities are encountered.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_IR] =
        page_writer.total_written - module_start;
    status =
        loom_bytecode_write_ir_section(&page_writer, &numbering, ir_offsets);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_IR] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_IR];
    }
  }

  // Symbols section: buffered in a string builder because the import/export
  // offset tables at the start reference entry positions that come later.
  // The SYMBOLS section uses a string_builder (which needs realloc, so it
  // can't use the arena). Use the module's context allocator.
  iree_string_builder_t symbols_builder;
  iree_string_builder_initialize(module->context->allocator, &symbols_builder);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_write_symbols_section(&symbols_builder, &numbering,
                                                 ir_offsets);
  }
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SYMBOLS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_page_writer_write(
        &page_writer, iree_string_builder_buffer(&symbols_builder),
        iree_string_builder_size(&symbols_builder));
    section_lengths[LOOM_BYTECODE_SECTION_SYMBOLS] =
        iree_string_builder_size(&symbols_builder);
  }
  iree_string_builder_deinitialize(&symbols_builder);

  // Strings section: all interned strings from the numbering context.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_STRINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_strings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_STRINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_STRINGS];
    }
  }

  // Sources section: context-level source identifiers (filenames, tags).
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SOURCES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_sources_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SOURCES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SOURCES];
    }
  }

  // Types section: interned type table in topological order.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_TYPES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_types_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_TYPES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_TYPES];
    }
  }

  // Encodings section: kind registry + parameterized instances.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_encodings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_ENCODINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS];
    }
  }

  // Ops section: op kind name registry.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_OPS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_ops_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_OPS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_OPS];
    }
  }

  // Locations section: empty until the location table is on the module.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_locations_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_LOCATIONS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS];
    }
  }

  // Flush remaining page buffer, then seek back to patch the section
  // directory and module directory with correct offsets and lengths.
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_flush(&page_writer);
  }

  // Patch section directory.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)section_dir_patch_position);
  }
  if (iree_status_is_ok(status)) {
    for (int i = 0;
         i < LOOM_BYTECODE_SECTION_COUNT && iree_status_is_ok(status); ++i) {
      loom_bytecode_section_dir_entry_t entry = {0};
      entry.section_kind = (uint16_t)i;
      entry.offset = section_offsets[i];
      entry.length = section_lengths[i];
      status = iree_io_stream_write(stream, sizeof(entry), &entry);
    }
  }

  // Patch module directory: module_offset and module_length.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)module_dir_offset_position);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_offset = module_start;
    status = iree_io_stream_write(stream, 8, &module_offset);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_length = page_writer.total_written - module_start;
    status = iree_io_stream_write(stream, 8, &module_length);
  }

  // All numbering tables, value maps, and ir_offsets were arena-allocated.
  // One call returns all blocks to the shared pool.
  iree_arena_deinitialize(&arena);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
