// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/format/bytecode/diagnostic.h"
#include "loom/format/bytecode/varint.h"
#include "loom/ir/attribute.h"
#include "loom/ir/types.h"

// Keep reader allocation guards aligned with the bytecode format comment.
#define LOOM_BYTECODE_MAX_SECTION_COUNT 256
#define LOOM_BYTECODE_MAX_STRING_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_STRING_LENGTH (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_TYPE_COUNT (UINT64_C(1) << 16)
#define LOOM_BYTECODE_MAX_OP_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_SYMBOL_COUNT (UINT64_C(1) << 16)
#define LOOM_BYTECODE_MAX_LOCATION_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_ENCODING_COUNT (UINT64_C(1) << 16)

enum {
  LOOM_BYTECODE_ATTR_I64 = 0,
  LOOM_BYTECODE_ATTR_F64 = 1,
  LOOM_BYTECODE_ATTR_STRING = 2,
  LOOM_BYTECODE_ATTR_BOOL = 3,
  LOOM_BYTECODE_ATTR_ENUM = 4,
  LOOM_BYTECODE_ATTR_I64_ARRAY = 5,
  LOOM_BYTECODE_ATTR_SYMBOL = 6,
  LOOM_BYTECODE_ATTR_TYPE = 7,
  LOOM_BYTECODE_ATTR_PREDICATE_LIST = 8,
  LOOM_BYTECODE_ATTR_DICT = 9,
  LOOM_BYTECODE_ATTR_ENCODING = 10,
};

typedef struct loom_bytecode_reader_cursor_t {
  loom_bytecode_cursor_t cursor;  // Bounded cursor over the current byte range.
  uint64_t absolute_offset;       // Absolute file offset for cursor byte 0.
  iree_string_view_t range_name;  // Human-readable range name for diagnostics.
} loom_bytecode_reader_cursor_t;

typedef struct loom_bytecode_reader_module_t {
  uint32_t name_offset;     // Offset into the file string pool.
  uint16_t name_length;     // Module name byte length.
  uint16_t flags;           // Module flags from the directory entry.
  uint64_t offset;          // Absolute module byte offset.
  uint64_t length;          // Module byte length.
  iree_string_view_t name;  // Name view into the file string pool.
} loom_bytecode_reader_module_t;

typedef struct loom_bytecode_reader_section_t {
  uint16_t kind;                 // Wire section kind.
  uint16_t flags;                // Section flags.
  uint64_t offset;               // Module-relative byte offset.
  uint64_t length;               // Section byte length.
  uint64_t absolute_offset;      // Absolute file byte offset.
  iree_const_byte_span_t bytes;  // Section payload bytes.
} loom_bytecode_reader_section_t;

typedef struct loom_bytecode_reader_state_t {
  iree_const_byte_span_t bytecode;  // Full bytecode file bytes.
  iree_string_view_t filename;      // Logical input name for diagnostics.
  loom_context_t* context;          // Dialect and encoding registry context.
  iree_arena_allocator_t* arena;    // Transient metadata arena.
  // Diagnostic context used by bytecode/diagnostic helpers.
  loom_bytecode_reader_diagnostic_context_t diagnostic_context;
  loom_bytecode_read_result_t result;  // Public result accumulator.

  iree_string_view_t file_string_pool;     // File-level module-name pool.
  loom_bytecode_reader_module_t* modules;  // Module directory entries.
  iree_host_size_t module_count;           // Number of module entries.

  iree_string_view_t* strings;    // Current module STRINGS entries.
  iree_host_size_t string_count;  // Number of current module strings.
  iree_string_view_t* sources;    // Current module SOURCES entries.
  iree_host_size_t source_count;  // Number of current module sources.
  loom_type_t* types;             // Current module TYPES entries.
  iree_host_size_t type_count;    // Number of current module types.
  const loom_op_vtable_t** ops;   // Current module OPS resolved vtables.
  iree_host_size_t op_count;      // Number of current module OPS entries.
  // Current module ENCODINGS family vtables.
  const loom_encoding_vtable_t** encoding_families;
  iree_host_size_t encoding_family_count;  // Number of encoding families.
  iree_host_size_t encoding_count;         // Number of encoding instances.
  iree_host_size_t location_count;         // Number of location entries.
  iree_host_size_t symbol_count;           // Number of symbol entries.
} loom_bytecode_reader_state_t;

static void loom_bytecode_reader_cursor_initialize(
    const uint8_t* data, iree_host_size_t length, uint64_t absolute_offset,
    iree_string_view_t range_name, loom_bytecode_reader_cursor_t* out_cursor) {
  loom_bytecode_cursor_initialize(data, length, &out_cursor->cursor);
  out_cursor->absolute_offset = absolute_offset;
  out_cursor->range_name = range_name;
}

static uint64_t loom_bytecode_reader_cursor_absolute_position(
    const loom_bytecode_reader_cursor_t* cursor) {
  return cursor->absolute_offset + (uint64_t)cursor->cursor.position;
}

static const char* loom_bytecode_section_name(uint16_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_SECTION_STRINGS:
      return "STRINGS";
    case LOOM_BYTECODE_SECTION_SOURCES:
      return "SOURCES";
    case LOOM_BYTECODE_SECTION_TYPES:
      return "TYPES";
    case LOOM_BYTECODE_SECTION_ENCODINGS:
      return "ENCODINGS";
    case LOOM_BYTECODE_SECTION_OPS:
      return "OPS";
    case LOOM_BYTECODE_SECTION_LOCATIONS:
      return "LOCATIONS";
    case LOOM_BYTECODE_SECTION_SYMBOLS:
      return "SYMBOLS";
    case LOOM_BYTECODE_SECTION_IR:
      return "IR";
    case LOOM_BYTECODE_SECTION_RESOURCES:
      return "RESOURCES";
    default:
      return "UNKNOWN";
  }
}

static bool loom_bytecode_reader_has_errors(
    const loom_bytecode_reader_state_t* reader) {
  return reader->result.error_count > 0;
}

static iree_status_t loom_bytecode_reader_emit(
    loom_bytecode_reader_state_t* reader, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    uint64_t offset, uint64_t length) {
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++reader->result.error_count;
  } else if (error->severity == LOOM_DIAGNOSTIC_WARNING) {
    ++reader->result.warning_count;
  }
  return loom_bytecode_reader_emit_diagnostic(
      &reader->diagnostic_context, error, params, param_count,
      loom_bytecode_reader_byte_range(offset, length));
}

static iree_status_t loom_bytecode_reader_emit_unexpected_end(
    loom_bytecode_reader_state_t* reader, uint64_t offset, uint64_t needed,
    uint64_t available) {
  loom_diagnostic_param_t params[] = {
      loom_param_u64(offset),
      loom_param_u64(needed),
      loom_param_u64(available),
  };
  return loom_bytecode_reader_emit(reader, &loom_err_bytecode_003, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_emit_invalid_field(
    loom_bytecode_reader_state_t* reader, iree_string_view_t section_name,
    iree_string_view_t table_name, uint64_t record_index,
    iree_string_view_t field_name, uint64_t offset, iree_string_view_t reason) {
  ++reader->result.error_count;
  return loom_bytecode_reader_emit_invalid_record_field(
      &reader->diagnostic_context, section_name, table_name, record_index,
      field_name, offset, reason);
}

static iree_status_t loom_bytecode_reader_emit_range_error(
    loom_bytecode_reader_state_t* reader, iree_string_view_t range_name,
    uint64_t offset, uint64_t length, uint64_t container_length) {
  ++reader->result.error_count;
  return loom_bytecode_reader_emit_invalid_range(&reader->diagnostic_context,
                                                 range_name, offset, length,
                                                 container_length);
}

static iree_status_t loom_bytecode_reader_emit_count_exceeds(
    loom_bytecode_reader_state_t* reader, iree_string_view_t table_name,
    uint64_t count, uint64_t limit, uint64_t offset) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(table_name),
      loom_param_u64(count),
      loom_param_u64(limit),
  };
  return loom_bytecode_reader_emit(reader, &loom_err_bytecode_009, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_emit_table_ref(
    loom_bytecode_reader_state_t* reader, iree_string_view_t table_name,
    uint64_t ref_id, uint64_t table_count, uint64_t offset) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(table_name),
      loom_param_u64(ref_id),
      loom_param_u64(table_count),
  };
  return loom_bytecode_reader_emit(reader, &loom_err_bytecode_012, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_emit_enum_value(
    loom_bytecode_reader_state_t* reader, iree_string_view_t field_name,
    uint64_t actual_value, uint64_t case_count, uint64_t offset) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u64(actual_value),
      loom_param_u64(case_count),
  };
  return loom_bytecode_reader_emit(reader, &loom_err_bytecode_011, params,
                                   IREE_ARRAYSIZE(params), offset, 1);
}

static iree_status_t loom_bytecode_reader_read_u8(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint8_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 1)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 1,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u8(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_u16_le(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint16_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 2)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 2,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u16_le(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_u32_le(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint32_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 4)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 4,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u32_le(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_u64_le(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 8)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 8,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u64_le(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_uvarint(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t* out_value) {
  uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  iree_status_t status = loom_uvarint_decode(&cursor->cursor, out_value);
  if (iree_status_is_ok(status)) return iree_ok_status();

  iree_status_code_t code = iree_status_code(status);
  iree_status_ignore(status);
  iree_string_view_t reason =
      code == IREE_STATUS_OUT_OF_RANGE
          ? IREE_SV("continuation bit reached the end of the byte range")
          : IREE_SV("encoding is non-canonical or exceeds the uint64 range");
  loom_diagnostic_param_t params[] = {
      loom_param_u64(offset),
      loom_param_string(reason),
  };
  return loom_bytecode_reader_emit(reader, &loom_err_bytecode_008, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_read_svarint(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    int64_t* out_value) {
  uint64_t zigzag = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &zigzag));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  *out_value = (int64_t)((zigzag >> 1) ^ -(zigzag & 1));
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_span(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t length, iree_const_byte_span_t* out_span) {
  if (length > (uint64_t)IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_range_error(
        reader, cursor->range_name,
        loom_bytecode_reader_cursor_absolute_position(cursor), length,
        cursor->absolute_offset + cursor->cursor.length);
  }
  iree_host_size_t host_length = (iree_host_size_t)length;
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, host_length)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), length,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_span(&cursor->cursor, host_length, out_span);
}

static iree_status_t loom_bytecode_reader_expect_empty(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    iree_string_view_t table_name) {
  if (loom_bytecode_cursor_is_empty(&cursor->cursor)) return iree_ok_status();
  return loom_bytecode_reader_emit_invalid_field(
      reader, cursor->range_name, table_name, 0, IREE_SV("trailing_bytes"),
      loom_bytecode_reader_cursor_absolute_position(cursor),
      IREE_SV("section has unread trailing bytes"));
}

static bool loom_bytecode_reader_string_is_valid_utf8(iree_string_view_t text) {
  return iree_unicode_utf8_validate(text);
}

static iree_status_t loom_bytecode_reader_validate_string_ref(
    loom_bytecode_reader_state_t* reader, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  if (string_id >= reader->string_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(reader->string_count),
    };
    return loom_bytecode_reader_emit(reader, &loom_err_bytecode_010, params,
                                     IREE_ARRAYSIZE(params), offset, 0);
  }
  *out_string = reader->strings[string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_type_ref_bounded(
    loom_bytecode_reader_state_t* reader, uint64_t type_id, uint64_t type_count,
    iree_string_view_t field_name, uint64_t offset, loom_type_t* out_type) {
  (void)field_name;
  if (type_id >= type_count) {
    return loom_bytecode_reader_emit_table_ref(reader, IREE_SV("TYPES"),
                                               type_id, type_count, offset);
  }
  *out_type = reader->types[type_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_type_ref(
    loom_bytecode_reader_state_t* reader, uint64_t type_id,
    iree_string_view_t field_name, uint64_t offset, loom_type_t* out_type) {
  return loom_bytecode_reader_validate_type_ref_bounded(
      reader, type_id, reader->type_count, field_name, offset, out_type);
}

static iree_status_t loom_bytecode_reader_validate_encoding_ref(
    loom_bytecode_reader_state_t* reader, uint64_t encoding_id,
    uint64_t offset) {
  if (encoding_id == 0 || encoding_id > reader->encoding_count) {
    return loom_bytecode_reader_emit_table_ref(reader, IREE_SV("ENCODINGS"),
                                               encoding_id,
                                               reader->encoding_count, offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_location_ref(
    loom_bytecode_reader_state_t* reader, uint64_t location_id,
    uint64_t offset) {
  if (location_id >= reader->location_count) {
    return loom_bytecode_reader_emit_table_ref(reader, IREE_SV("LOCATIONS"),
                                               location_id,
                                               reader->location_count, offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_op_ref(
    loom_bytecode_reader_state_t* reader, uint64_t op_table_index_plus1,
    uint64_t offset, const loom_op_vtable_t** out_vtable) {
  if (op_table_index_plus1 == 0 || op_table_index_plus1 > reader->op_count) {
    return loom_bytecode_reader_emit_table_ref(
        reader, IREE_SV("OPS"), op_table_index_plus1, reader->op_count, offset);
  }
  *out_vtable = reader->ops[op_table_index_plus1 - 1];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_range(
    loom_bytecode_reader_state_t* reader, iree_string_view_t range_name,
    uint64_t offset, uint64_t length, uint64_t container_length) {
  if (length > UINT64_MAX - offset || offset + length > container_length) {
    return loom_bytecode_reader_emit_range_error(reader, range_name, offset,
                                                 length, container_length);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_string_table(
    loom_bytecode_reader_state_t* reader,
    loom_bytecode_reader_section_t section, iree_string_view_t table_name,
    uint64_t count_limit, iree_string_view_t** out_strings,
    iree_host_size_t* out_count) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section.bytes.data, section.bytes.data_length, section.absolute_offset,
      table_name, &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > count_limit || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(reader, table_name, count,
                                                   count_limit, count_offset);
  }

  iree_string_view_t* strings = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)count, sizeof(iree_string_view_t),
        (void**)&strings));
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t string_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t length = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (length > LOOM_BYTECODE_MAX_STRING_LENGTH) {
      return loom_bytecode_reader_emit_count_exceeds(
          reader, IREE_SV("string_length"), length,
          LOOM_BYTECODE_MAX_STRING_LENGTH, string_offset);
    }
    iree_const_byte_span_t bytes = {0};
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_span(reader, &cursor, length, &bytes));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t text =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!loom_bytecode_reader_string_is_valid_utf8(text)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, table_name, IREE_SV("string"), i, IREE_SV("utf8_data"),
          string_offset, IREE_SV("string payload is not valid UTF-8"));
    }
    strings[i] = text;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_expect_empty(reader, &cursor, table_name));
  *out_strings = strings;
  *out_count = (iree_host_size_t)count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_predicate_list(
    loom_bytecode_reader_state_t* reader,
    loom_bytecode_reader_cursor_t* cursor) {
  uint64_t predicate_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &predicate_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  for (uint64_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    uint8_t predicate_kind = 0;
    uint8_t arg_count = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &predicate_kind));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (predicate_kind > 9) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("predicate_kind"), predicate_kind, 10, kind_offset);
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &arg_count));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    for (uint8_t arg_index = 0; arg_index < arg_count; ++arg_index) {
      uint8_t tag = 0;
      uint64_t tag_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &tag));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (tag == 1) {
        uint64_t name_id = 0;
        uint64_t name_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, cursor, &name_id));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        iree_string_view_t unused = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, name_id, IREE_SV("predicate_value_name"), name_offset,
            &unused));
      } else if (tag == 2) {
        int64_t value = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_svarint(reader, cursor, &value));
        (void)value;
      } else {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("predicate_arg_tag"), tag, 3, tag_offset);
      }
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_attr_value(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint8_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_ATTR_I64: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_svarint(reader, cursor, &value));
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_F64: {
      uint64_t bits = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u64_le(reader, cursor, &bits));
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_STRING:
    case LOOM_BYTECODE_ATTR_ENUM: {
      uint64_t string_id = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &string_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t unused = {0};
      return loom_bytecode_reader_validate_string_ref(
          reader, string_id, IREE_SV("attribute_string"), offset, &unused);
    }
    case LOOM_BYTECODE_ATTR_BOOL: {
      uint8_t value = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, cursor, &value));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (value > 1) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("bool_attribute"), value, 2, offset);
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_I64_ARRAY: {
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &count));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      for (uint64_t i = 0; i < count; ++i) {
        int64_t value = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_svarint(reader, cursor, &value));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SYMBOL: {
      uint64_t module_id = 0;
      uint64_t symbol_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &module_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &symbol_id));
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_TYPE: {
      uint64_t type_id = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &type_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      loom_type_t unused = {0};
      return loom_bytecode_reader_validate_type_ref(
          reader, type_id, IREE_SV("attribute_type"), offset, &unused);
    }
    case LOOM_BYTECODE_ATTR_PREDICATE_LIST:
      return loom_bytecode_reader_skip_predicate_list(reader, cursor);
    case LOOM_BYTECODE_ATTR_DICT: {
      uint64_t entry_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &entry_count));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t previous_key = {0};
      for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
        uint64_t key_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t key_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        iree_string_view_t key = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, key_id, IREE_SV("dict_key"), key_offset, &key));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (entry_index > 0 &&
            iree_string_view_compare(key, previous_key) <= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, cursor->range_name, IREE_SV("dict"), entry_index,
              IREE_SV("key_id"), key_offset,
              IREE_SV("dictionary keys are not in canonical order"));
        }
        previous_key = key;
        uint8_t value_kind = 0;
        uint64_t value_kind_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (value_kind > LOOM_BYTECODE_ATTR_ENCODING) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("attribute_kind"), value_kind,
              LOOM_BYTECODE_ATTR_ENCODING + 1, value_kind_offset);
        }
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_skip_attr_value(reader, cursor, value_kind));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENCODING: {
      uint64_t encoding_id = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &encoding_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      return loom_bytecode_reader_validate_encoding_ref(reader, encoding_id,
                                                        offset);
    }
    default:
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), kind,
          LOOM_BYTECODE_ATTR_ENCODING + 1,
          loom_bytecode_reader_cursor_absolute_position(cursor));
  }
}

static iree_status_t loom_bytecode_reader_read_encodings(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("ENCODINGS"), &cursor);

  uint64_t family_count = 0;
  uint64_t family_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &family_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (family_count > LOOM_BYTECODE_MAX_ENCODING_COUNT ||
      family_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("encoding_families"), family_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, family_count_offset);
  }
  if (family_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, (iree_host_size_t)family_count,
                                  sizeof(const loom_encoding_vtable_t*),
                                  (void**)&reader->encoding_families));
  }
  reader->encoding_family_count = (iree_host_size_t)family_count;
  for (uint64_t i = 0; i < family_count; ++i) {
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t name = {0};
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("encoding_family_name"), name_offset, &name));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    const loom_encoding_vtable_t* vtable =
        loom_context_lookup_encoding_vtable(reader->context, name);
    if (!vtable) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("ENCODINGS"), IREE_SV("family"), i,
          IREE_SV("name_id"), name_offset,
          IREE_SV("encoding family is not registered in the context"));
    }
    reader->encoding_families[i] = vtable;
  }

  uint64_t instance_count = 0;
  uint64_t instance_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &instance_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (instance_count > LOOM_BYTECODE_MAX_ENCODING_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("encoding_instances"), instance_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, instance_count_offset);
  }
  reader->encoding_count = (iree_host_size_t)instance_count;
  for (uint64_t i = 0; i < instance_count; ++i) {
    uint64_t family_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t family_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &family_index));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (family_index >= family_count) {
      return loom_bytecode_reader_emit_table_ref(
          reader, IREE_SV("encoding_families"), family_index, family_count,
          family_offset);
    }
    uint64_t alias_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t alias_plus_one = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &alias_plus_one));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (alias_plus_one > reader->string_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("alias_string_id_plus1")),
          loom_param_u64(alias_plus_one - 1),
          loom_param_u64(reader->string_count),
      };
      return loom_bytecode_reader_emit(reader, &loom_err_bytecode_010, params,
                                       IREE_ARRAYSIZE(params), alias_offset, 0);
    }
    uint64_t param_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &param_count));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t previous_param = {0};
    for (uint64_t param_index = 0; param_index < param_count; ++param_index) {
      uint64_t name_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t name_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t param_name = {0};
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, name_id, IREE_SV("encoding_param_name"), name_offset,
          &param_name));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (param_index > 0 &&
          iree_string_view_compare(param_name, previous_param) <= 0) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("ENCODINGS"), IREE_SV("instance"), i,
            IREE_SV("param_name"), name_offset,
            IREE_SV("encoding parameters are not in canonical order"));
      }
      previous_param = param_name;
      uint8_t value_kind = 0;
      uint64_t kind_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &value_kind));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (value_kind > LOOM_BYTECODE_ATTR_ENCODING) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("attribute_kind"), value_kind,
            LOOM_BYTECODE_ATTR_ENCODING + 1, kind_offset);
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_skip_attr_value(reader, &cursor, value_kind));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    }
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor,
                                           IREE_SV("ENCODINGS"));
}

static iree_status_t loom_bytecode_reader_decode_type_kind(
    loom_bytecode_reader_state_t* reader, uint8_t kind_byte, uint64_t offset,
    loom_type_kind_t* out_kind) {
  switch (kind_byte) {
    case LOOM_BYTECODE_TYPE_NONE:
      *out_kind = LOOM_TYPE_NONE;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_SCALAR:
      *out_kind = LOOM_TYPE_SCALAR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_TILE:
      *out_kind = LOOM_TYPE_TILE;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_TENSOR:
      *out_kind = LOOM_TYPE_TENSOR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_GROUP:
      *out_kind = LOOM_TYPE_GROUP;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_FUNCTION:
      *out_kind = LOOM_TYPE_FUNCTION;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_DIALECT:
      *out_kind = LOOM_TYPE_DIALECT;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_ENCODING:
      *out_kind = LOOM_TYPE_ENCODING;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_POOL:
      *out_kind = LOOM_TYPE_POOL;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_VECTOR:
      *out_kind = LOOM_TYPE_VECTOR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_VIEW:
      *out_kind = LOOM_TYPE_VIEW;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_BUFFER:
      *out_kind = LOOM_TYPE_BUFFER;
      return iree_ok_status();
    default: {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(kind_byte),
          loom_param_u64(offset),
      };
      return loom_bytecode_reader_emit(reader, &loom_err_bytecode_004, params,
                                       IREE_ARRAYSIZE(params), offset, 1);
    }
  }
}

static iree_status_t loom_bytecode_reader_build_shaped_type(
    loom_bytecode_reader_state_t* reader, loom_type_kind_t kind,
    loom_scalar_type_t element_type, uint8_t rank, uint8_t attachment,
    uint64_t encoding_instance, const uint64_t* dims, loom_type_t* out_type,
    uint64_t offset) {
  if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
    return loom_bytecode_reader_emit_enum_value(
        reader, IREE_SV("element_type"), element_type, LOOM_SCALAR_TYPE_COUNT_,
        offset);
  }
  if (rank > LOOM_TYPE_MAX_RANK) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("TYPES"), IREE_SV("type"), 0, IREE_SV("rank"), offset,
        IREE_SV("rank exceeds loom type maximum"));
  }
  if (kind == LOOM_TYPE_VECTOR && rank == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("TYPES"), IREE_SV("type"), 0, IREE_SV("rank"), offset,
        IREE_SV("vector types must have rank >= 1"));
  }

  uint16_t encoding_id = 0;
  loom_encoding_flags_t encoding_flags = 0;
  switch (attachment) {
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE:
      if (encoding_instance != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("none encoding attachment must have id 0"));
      }
      break;
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC: {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_encoding_ref(
          reader, encoding_instance, offset));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (encoding_instance > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("encoding id exceeds runtime type field width"));
      }
      encoding_id = (uint16_t)encoding_instance;
      break;
    }
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA:
      if (encoding_instance != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("dynamic encoding attachment must have id 0"));
      }
      encoding_flags = LOOM_ENCODING_FLAG_SSA;
      break;
    default:
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("encoding_attachment"), attachment,
          LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA + 1, offset);
  }
  if (kind == LOOM_TYPE_VECTOR && (encoding_id != 0 || encoding_flags != 0)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
        IREE_SV("encoding_attachment"), offset,
        IREE_SV("vector types must not carry encoding or layout attachments"));
  }

  loom_type_t type = {0};
  if (rank == 0) {
    type = loom_type_shaped_0d(kind, element_type, encoding_id);
  } else if (rank == 1) {
    type = loom_type_shaped_1d(kind, element_type, dims[0], encoding_id);
  } else if (rank == 2) {
    type =
        loom_type_shaped_2d(kind, element_type, dims[0], dims[1], encoding_id);
  } else {
    loom_overflow_dim_t* overflow_dims = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(reader->arena, rank,
                                                   sizeof(loom_overflow_dim_t),
                                                   (void**)&overflow_dims));
    bool all_static = true;
    for (uint8_t i = 0; i < rank; ++i) {
      overflow_dims[i] = dims[i];
      if (loom_dim_is_dynamic(dims[i])) all_static = false;
    }
    uint8_t flags = all_static ? LOOM_TYPE_FLAG_ALL_STATIC : 0;
    type.header = loom_type_make_header(kind, element_type, rank, flags);
    type.encoding_id = encoding_id;
    type.dims[0] = (uint64_t)(uintptr_t)overflow_dims;
  }
  type.encoding_flags = encoding_flags;
  *out_type = type;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_types(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("TYPES"), &cursor);

  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_TYPE_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("TYPES"), count, LOOM_BYTECODE_MAX_TYPE_COUNT,
        count_offset);
  }
  if (count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, (iree_host_size_t)count,
                                  sizeof(loom_type_t), (void**)&reader->types));
  }
  reader->type_count = (iree_host_size_t)count;
  for (uint64_t type_index = 0; type_index < count; ++type_index) {
    uint64_t type_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, &cursor, &kind_byte));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    loom_type_kind_t kind = LOOM_TYPE_NONE;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_decode_type_kind(
        reader, kind_byte, type_offset, &kind));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

    loom_type_t type = {0};
    switch (kind) {
      case LOOM_TYPE_NONE:
        type = loom_type_none();
        break;
      case LOOM_TYPE_SCALAR: {
        uint8_t element_type = 0;
        uint64_t element_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &element_type));
        if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("scalar_type"), element_type,
              LOOM_SCALAR_TYPE_COUNT_, element_offset);
        }
        type = loom_type_scalar((loom_scalar_type_t)element_type);
        break;
      }
      case LOOM_TYPE_TILE:
      case LOOM_TYPE_TENSOR:
      case LOOM_TYPE_VECTOR:
      case LOOM_TYPE_VIEW: {
        uint8_t element_type = 0;
        uint8_t rank = 0;
        uint8_t attachment = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &element_type));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &rank));
        uint64_t attachment_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &attachment));
        uint64_t encoding_instance = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            reader, &cursor, &encoding_instance));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        uint64_t dims[LOOM_TYPE_MAX_RANK] = {0};
        if (rank > LOOM_TYPE_MAX_RANK) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("rank"), type_offset,
              IREE_SV("rank exceeds loom type maximum"));
        }
        for (uint8_t i = 0; i < rank; ++i) {
          uint8_t is_dynamic = 0;
          uint64_t dim_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_u8(reader, &cursor, &is_dynamic));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          if (is_dynamic == 0) {
            uint64_t size = 0;
            IREE_RETURN_IF_ERROR(
                loom_bytecode_reader_read_uvarint(reader, &cursor, &size));
            if (size > LOOM_DIM_MAX_STATIC_SIZE) {
              return loom_bytecode_reader_emit_invalid_field(
                  reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
                  IREE_SV("dim_size"), dim_offset,
                  IREE_SV("static dimension exceeds loom maximum"));
            }
            dims[i] = loom_dim_pack_static((int64_t)size);
          } else if (is_dynamic == 1) {
            dims[i] = loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID);
          } else {
            return loom_bytecode_reader_emit_enum_value(
                reader, IREE_SV("is_dynamic"), is_dynamic, 2, dim_offset);
          }
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_build_shaped_type(
            reader, kind, (loom_scalar_type_t)element_type, rank, attachment,
            encoding_instance, dims, &type, attachment_offset));
        break;
      }
      case LOOM_TYPE_GROUP: {
        uint8_t scope = 0;
        uint64_t scope_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &scope));
        if (scope >= LOOM_GROUP_SCOPE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("group_scope"), scope, LOOM_GROUP_SCOPE_COUNT_,
              scope_offset);
        }
        type.header = loom_type_make_header(
            LOOM_TYPE_GROUP, (loom_scalar_type_t)scope, 0,
            LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
        break;
      }
      case LOOM_TYPE_FUNCTION: {
        uint64_t arg_count = 0;
        uint64_t result_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &arg_count));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &result_count));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (arg_count > UINT16_MAX || result_count > UINT16_MAX ||
            arg_count + result_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("signature_count"), type_offset,
              IREE_SV("function signature exceeds runtime field width"));
        }
        iree_host_size_t total_count =
            (iree_host_size_t)(arg_count + result_count);
        iree_host_size_t total_size =
            sizeof(loom_func_type_data_t) + total_count * sizeof(loom_type_t);
        loom_func_type_data_t* data = NULL;
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(reader->arena, total_size, (void**)&data));
        data->arg_count = (uint16_t)arg_count;
        data->result_count = (uint16_t)result_count;
        data->reserved = 0;
        for (iree_host_size_t i = 0; i < total_count; ++i) {
          uint64_t ref_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t type_id = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &type_id));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref_bounded(
              reader, type_id, type_index, IREE_SV("signature_type"),
              ref_offset, &data->types[i]));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        }
        type = loom_type_function(data);
        break;
      }
      case LOOM_TYPE_DIALECT: {
        uint64_t name_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t name_id = 0;
        uint64_t param_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &param_count));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        iree_string_view_t unused_name = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, name_id, IREE_SV("dialect_type_name"), name_offset,
            &unused_name));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (param_count > UINT16_MAX || param_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              reader, IREE_SV("dialect_type_params"), param_count, UINT16_MAX,
              type_offset);
        }
        loom_type_t* params = NULL;
        if (param_count > 0) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              reader->arena, (iree_host_size_t)param_count, sizeof(loom_type_t),
              (void**)&params));
        }
        for (uint64_t i = 0; i < param_count; ++i) {
          uint64_t ref_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t type_id = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &type_id));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref_bounded(
              reader, type_id, type_index, IREE_SV("dialect_type_param"),
              ref_offset, &params[i]));
        }
        type = param_count == 0
                   ? loom_type_dialect_opaque((loom_string_id_t)name_id)
                   : loom_type_dialect((loom_string_id_t)name_id,
                                       (uint16_t)param_count, params);
        break;
      }
      case LOOM_TYPE_ENCODING: {
        uint8_t role = 0;
        uint64_t role_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &role));
        if (role >= LOOM_ENCODING_ROLE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("encoding_role"), role, LOOM_ENCODING_ROLE_COUNT_,
              role_offset);
        }
        type = loom_type_encoding_with_role((loom_encoding_role_t)role);
        break;
      }
      case LOOM_TYPE_POOL: {
        uint8_t is_dynamic = 0;
        uint64_t dim_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &is_dynamic));
        if (is_dynamic == 0) {
          uint64_t size = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &size));
          if (size > LOOM_DIM_MAX_STATIC_SIZE) {
            return loom_bytecode_reader_emit_invalid_field(
                reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
                IREE_SV("block_size"), dim_offset,
                IREE_SV("static pool block size exceeds loom maximum"));
          }
          type = loom_type_pool(loom_dim_pack_static((int64_t)size));
        } else if (is_dynamic == 1) {
          type = loom_type_pool(loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID));
        } else {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("is_dynamic"), is_dynamic, 2, dim_offset);
        }
        break;
      }
      case LOOM_TYPE_BUFFER:
        type = loom_type_buffer();
        break;
      default:
        break;
    }
    reader->types[type_index] = type;
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("TYPES"));
}

static iree_status_t loom_bytecode_reader_read_ops(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("OPS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_OP_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("OPS"), count, LOOM_BYTECODE_MAX_OP_COUNT,
        count_offset);
  }
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)count, sizeof(const loom_op_vtable_t*),
        (void**)&reader->ops));
  }
  reader->op_count = (iree_host_size_t)count;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t op_name = {0};
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("op_name"), name_offset, &op_name));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
    const loom_op_vtable_t* vtable =
        loom_context_lookup_op_by_name(reader->context, op_name, &kind);
    if (!vtable) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("OPS"), IREE_SV("op"), i, IREE_SV("name_id"),
          name_offset, IREE_SV("op name is not registered in the context"));
    }
    reader->ops[i] = vtable;
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("OPS"));
}

static iree_status_t loom_bytecode_reader_read_locations(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("LOCATIONS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_LOCATION_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("LOCATIONS"), count, LOOM_BYTECODE_MAX_LOCATION_COUNT,
        count_offset);
  }
  reader->location_count = (iree_host_size_t)count;
  for (uint64_t i = 0; i < count; ++i) {
    uint8_t kind = 0;
    uint8_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &flags));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("flags"), kind_offset + 1,
          IREE_SV("location has unsupported flag bits"));
    }
    switch (kind) {
      case 0:
        break;
      case 1: {
        uint64_t source_id = 0;
        uint64_t source_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &source_id));
        if (source_id >= reader->source_count) {
          return loom_bytecode_reader_emit_table_ref(
              reader, IREE_SV("SOURCES"), source_id, reader->source_count,
              source_offset);
        }
        for (int field = 0; field < 4; ++field) {
          uint64_t value = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &value));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        }
        break;
      }
      case 2: {
        uint64_t child_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &child_count));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        for (uint64_t child_index = 0; child_index < child_count;
             ++child_index) {
          uint64_t child_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t child = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &child));
          if (child >= i) {
            return loom_bytecode_reader_emit_table_ref(
                reader, IREE_SV("LOCATIONS"), child, i, child_offset);
          }
        }
        break;
      }
      case 3: {
        uint64_t source_id = 0;
        uint64_t source_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &source_id));
        if (source_id >= reader->source_count) {
          return loom_bytecode_reader_emit_table_ref(
              reader, IREE_SV("SOURCES"), source_id, reader->source_count,
              source_offset);
        }
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &data_length));
        iree_const_byte_span_t unused = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            reader, &cursor, data_length, &unused));
        break;
      }
      default:
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("location_kind"), kind, 4, kind_offset);
    }
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor,
                                           IREE_SV("LOCATIONS"));
}

static iree_status_t loom_bytecode_reader_read_symbols(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* symbols_section,
    const loom_bytecode_reader_section_t* ir_section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      symbols_section->bytes.data, symbols_section->bytes.data_length,
      symbols_section->absolute_offset, IREE_SV("SYMBOLS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_SYMBOL_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("SYMBOLS"), count, LOOM_BYTECODE_MAX_SYMBOL_COUNT,
        count_offset);
  }
  reader->symbol_count = (iree_host_size_t)count;

  uint64_t import_count = 0;
  uint64_t export_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &import_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &export_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (import_count > count || export_count > count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("header"), 0,
        IREE_SV("import_export_count"),
        loom_bytecode_reader_cursor_absolute_position(&cursor),
        IREE_SV("import/export counts must not exceed symbol count"));
  }
  uint64_t offset_table_count = import_count + export_count;
  uint64_t* offset_table = NULL;
  if (offset_table_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)offset_table_count, sizeof(uint64_t),
        (void**)&offset_table));
  }
  for (uint64_t i = 0; i < offset_table_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, &cursor, &offset_table[i]));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  uint64_t entries_base = cursor.absolute_offset + cursor.cursor.position;
  uint64_t entries_payload_length =
      symbols_section->absolute_offset + symbols_section->length - entries_base;
  for (uint64_t i = 0; i < offset_table_count; ++i) {
    if (offset_table[i] >= entries_payload_length) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("symbol_offset_table"), offset_table[i], 1,
          entries_payload_length);
    }
  }

  for (uint64_t symbol_index = 0; symbol_index < count; ++symbol_index) {
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t unused_name = {0};
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("symbol_name"), name_offset, &unused_name));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

    uint8_t kind = 0;
    uint8_t visibility = 0;
    uint16_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, &cursor, &visibility));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &flags));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (kind >= LOOM_BYTECODE_SYMBOL_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("symbol_kind"), kind, LOOM_BYTECODE_SYMBOL_COUNT_,
          kind_offset);
    }
    if (visibility >= LOOM_BYTECODE_SYMBOL_VISIBILITY_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("symbol_visibility"), visibility,
          LOOM_BYTECODE_SYMBOL_VISIBILITY_COUNT_, kind_offset + 1);
    }
    if (flags & ~(LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC |
                  LOOM_BYTECODE_SYMBOL_FLAG_IMPORT)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("flags"), kind_offset + 2,
          IREE_SV("symbol has unsupported flag bits"));
    }
    if (flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT) {
      for (int i = 0; i < 2; ++i) {
        uint64_t ref_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t ref_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &ref_id));
        iree_string_view_t unused = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, ref_id,
            i == 0 ? IREE_SV("source_module_id") : IREE_SV("source_symbol_id"),
            ref_offset, &unused));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
    }

    if (kind <= LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
      uint64_t op_ref_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t op_table_index_plus1 = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          reader, &cursor, &op_table_index_plus1));
      const loom_op_vtable_t* unused_vtable = NULL;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
          reader, op_table_index_plus1, op_ref_offset, &unused_vtable));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

      uint8_t calling_convention = 0;
      uint64_t cc_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &calling_convention));
      if (calling_convention > 2) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("calling_convention"), calling_convention, 3,
            cc_offset);
      }
      uint64_t arg_count = 0;
      uint64_t result_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &arg_count));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &result_count));
      if (arg_count > UINT16_MAX || result_count > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
            IREE_SV("signature_count"), cc_offset,
            IREE_SV("function signature exceeds runtime field width"));
      }
      for (uint64_t i = 0; i < arg_count; ++i) {
        uint64_t ref_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t type_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &type_id));
        loom_type_t unused = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref(
            reader, type_id, IREE_SV("arg_type"), ref_offset, &unused));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      for (uint64_t i = 0; i < result_count; ++i) {
        uint8_t is_tied = 0;
        uint64_t tie_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &is_tied));
        if (is_tied > 1) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("is_tied"), is_tied, 2, tie_offset);
        }
        uint64_t type_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &type_id));
        loom_type_t unused = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref(
            reader, type_id, IREE_SV("result_type"), tie_offset, &unused));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (is_tied) {
          uint64_t operand_index = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              reader, &cursor, &operand_index));
          if (operand_index >= arg_count) {
            return loom_bytecode_reader_emit_invalid_field(
                reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
                IREE_SV("tied_operand_index"), tie_offset,
                IREE_SV("tied result references an argument out of range"));
          }
        }
      }
      uint64_t tied_result_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          reader, &cursor, &tied_result_count));
      (void)tied_result_count;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_skip_predicate_list(reader, &cursor));
      if (kind == LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE ||
          kind == LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
        for (int i = 0; i < 2; ++i) {
          uint64_t value_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t value = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &value));
          if (i == 0) {
            iree_string_view_t unused = {0};
            IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
                reader, value, IREE_SV("implements_op_name"), value_offset,
                &unused));
            if (loom_bytecode_reader_has_errors(reader))
              return iree_ok_status();
          }
        }
      }
      uint8_t has_body = 0;
      uint64_t has_body_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &has_body));
      if (has_body > 1) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("has_body"), has_body, 2, has_body_offset);
      }
      if (has_body) {
        uint64_t ir_offset = 0;
        uint32_t ir_length = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u64_le(reader, &cursor, &ir_offset));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u32_le(reader, &cursor, &ir_length));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
            reader, IREE_SV("IR body"), ir_offset, ir_length,
            ir_section ? ir_section->length : 0));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
    } else if (kind == LOOM_BYTECODE_SYMBOL_GLOBAL) {
      uint64_t type_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t type_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &type_id));
      loom_type_t unused = {0};
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref(
          reader, type_id, IREE_SV("global_type"), type_offset, &unused));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      uint8_t is_mutable = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &is_mutable));
      uint64_t initializer_offset = 0;
      uint32_t initializer_length = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
          reader, &cursor, &initializer_offset));
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u32_le(
          reader, &cursor, &initializer_length));
      (void)is_mutable;
      (void)initializer_offset;
      (void)initializer_length;
    }
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

    uint64_t next_entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor) - entries_base;
    if (next_entry_offset > entries_payload_length) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("symbol_entry"), next_entry_offset, 0,
          entries_payload_length);
    }
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("SYMBOLS"));
}

static iree_status_t loom_bytecode_reader_validate_file_header(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t* out_string_pool_length) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 16)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, 0, 16, loom_bytecode_cursor_remaining(&cursor->cursor));
  }

  uint8_t magic[LOOM_BYTECODE_MAGIC_LENGTH] = {0};
  iree_const_byte_span_t magic_span = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
      reader, cursor, LOOM_BYTECODE_MAGIC_LENGTH, &magic_span));
  memcpy(magic, magic_span.data, sizeof(magic));
  if (memcmp(magic, LOOM_BYTECODE_MAGIC, LOOM_BYTECODE_MAGIC_LENGTH) != 0) {
    char actual_magic[9] = {0};
    static const char kHex[] = "0123456789ABCDEF";
    for (int i = 0; i < LOOM_BYTECODE_MAGIC_LENGTH; ++i) {
      actual_magic[i * 2 + 0] = kHex[magic[i] >> 4];
      actual_magic[i * 2 + 1] = kHex[magic[i] & 0xF];
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV(LOOM_BYTECODE_MAGIC)),
        loom_param_string(iree_make_cstring_view(actual_magic)),
    };
    return loom_bytecode_reader_emit(reader, &loom_err_bytecode_001, params,
                                     IREE_ARRAYSIZE(params), 0,
                                     LOOM_BYTECODE_MAGIC_LENGTH);
  }

  uint8_t version = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &version));
  if (version != LOOM_BYTECODE_FORMAT_VERSION) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(version),
        loom_param_u32(LOOM_BYTECODE_FORMAT_VERSION),
    };
    return loom_bytecode_reader_emit(reader, &loom_err_bytecode_002, params,
                                     IREE_ARRAYSIZE(params), 4, 1);
  }
  uint8_t location_mode = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(reader, cursor, &location_mode));
  if (location_mode > LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    return loom_bytecode_reader_emit_enum_value(
        reader, IREE_SV("location_mode"), location_mode,
        LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS + 1, 5);
  }
  if (location_mode == LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("location_mode"),
        5, IREE_SV("FULL_LOCATIONS bytecode requires field-span support"));
  }
  reader->result.location_mode = (loom_bytecode_location_mode_t)location_mode;

  uint16_t module_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u16_le(reader, cursor, &module_count));
  if (module_count == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("module_count"),
        6, IREE_SV("bytecode files must contain at least one module"));
  }
  reader->module_count = module_count;
  reader->result.module_count = module_count;

  uint32_t string_pool_length = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u32_le(reader, cursor, &string_pool_length));
  *out_string_pool_length = string_pool_length;
  uint32_t reserved = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u32_le(reader, cursor, &reserved));
  if (reserved != 0) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("reserved"), 12,
        IREE_SV("reserved header field must be zero"));
  }

  uint64_t producer_start = cursor->cursor.position;
  while (cursor->cursor.position < cursor->cursor.length &&
         cursor->cursor.data[cursor->cursor.position] != 0) {
    ++cursor->cursor.position;
  }
  if (cursor->cursor.position >= cursor->cursor.length) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, producer_start, 1,
        cursor->cursor.length >= producer_start
            ? cursor->cursor.length - producer_start
            : 0);
  }
  iree_string_view_t producer =
      iree_make_string_view((const char*)cursor->cursor.data + producer_start,
                            cursor->cursor.position - producer_start);
  if (!loom_bytecode_reader_string_is_valid_utf8(producer)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("producer"),
        producer_start, IREE_SV("producer string is not valid UTF-8"));
  }
  ++cursor->cursor.position;
  while ((cursor->cursor.position & 7) != 0) {
    uint64_t padding_offset = cursor->cursor.position;
    uint8_t padding = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &padding));
    if (padding != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("FILE"), IREE_SV("header"), 0,
          IREE_SV("producer_padding"), padding_offset,
          IREE_SV("header alignment padding must be zero"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_module_directory(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t string_pool_length) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      reader->arena, reader->module_count,
      sizeof(loom_bytecode_reader_module_t), (void**)&reader->modules));
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    loom_bytecode_reader_module_t* module = &reader->modules[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(reader, cursor, &module->name_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, cursor, &module->name_length));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, cursor, &module->flags));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &module->offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &module->length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (module->flags & ~LOOM_BYTECODE_MODULE_FLAG_DECLARATIONS_ONLY) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("FILE"), IREE_SV("module_directory"), i,
          IREE_SV("flags"), entry_offset + 6,
          IREE_SV("module has unsupported flag bits"));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("file_string_pool"), module->name_offset,
        module->name_length, string_pool_length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("module"), module->offset, module->length,
        reader->bytecode.data_length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  uint64_t string_pool_end = 0;
  if (string_pool_length > UINT64_MAX - cursor->cursor.position) {
    return loom_bytecode_reader_emit_range_error(
        reader, IREE_SV("file_string_pool"), cursor->cursor.position,
        string_pool_length, reader->bytecode.data_length);
  }
  string_pool_end = cursor->cursor.position + string_pool_length;
  if (string_pool_end > UINT64_MAX - 7) {
    return loom_bytecode_reader_emit_range_error(
        reader, IREE_SV("file_string_pool"), cursor->cursor.position,
        string_pool_length, reader->bytecode.data_length);
  }
  uint64_t previous_end = (string_pool_end + 7) & ~UINT64_C(7);
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    const loom_bytecode_reader_module_t* module = &reader->modules[i];
    if (module->offset < previous_end) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("module"), module->offset, module->length,
          reader->bytecode.data_length);
    }
    previous_end = module->offset + module->length;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_file_string_pool(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t string_pool_length) {
  iree_const_byte_span_t pool = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
      reader, cursor, string_pool_length, &pool));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  reader->file_string_pool =
      iree_make_string_view((const char*)pool.data, pool.data_length);
  if (!loom_bytecode_reader_string_is_valid_utf8(reader->file_string_pool)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("string_pool"), 0,
        IREE_SV("utf8_data"), cursor->absolute_offset,
        IREE_SV("file string pool is not valid UTF-8"));
  }
  while ((cursor->cursor.position & 7) != 0) {
    uint64_t padding_offset = cursor->cursor.position;
    uint8_t padding = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &padding));
    if (padding != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("FILE"), IREE_SV("string_pool"), 0,
          IREE_SV("padding"), padding_offset,
          IREE_SV("file string pool alignment padding must be zero"));
    }
  }
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    loom_bytecode_reader_module_t* module = &reader->modules[i];
    module->name = iree_make_string_view(
        reader->file_string_pool.data + module->name_offset,
        module->name_length);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_section_directory(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_module_t* module,
    loom_bytecode_reader_section_t** out_sections,
    iree_host_size_t* out_section_count) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      reader->bytecode.data + module->offset, (iree_host_size_t)module->length,
      module->offset, IREE_SV("MODULE"), &cursor);

  uint64_t section_count = 0;
  uint64_t section_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &section_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (section_count > LOOM_BYTECODE_MAX_SECTION_COUNT ||
      section_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("SECTIONS"), section_count,
        LOOM_BYTECODE_MAX_SECTION_COUNT, section_count_offset);
  }

  uint64_t* summary_fields[] = {
      &reader->result.first_module.value_count,
      &reader->result.first_module.region_count,
      &reader->result.first_module.block_count,
      &reader->result.first_module.op_count,
  };
  uint64_t ignored_summary_fields[4] = {0};
  if (module != &reader->modules[0]) {
    for (int i = 0; i < 4; ++i) summary_fields[i] = &ignored_summary_fields[i];
  }
  for (int i = 0; i < 4; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, summary_fields[i]));
  }
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  loom_bytecode_reader_section_t* sections = NULL;
  if (section_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)section_count,
        sizeof(loom_bytecode_reader_section_t), (void**)&sections));
  }
  uint64_t minimum_section_offset = cursor.cursor.position + section_count * 32;
  uint64_t previous_end = minimum_section_offset;
  for (uint64_t i = 0; i < section_count; ++i) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    loom_bytecode_reader_section_t* section = &sections[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &section->kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &section->flags));
    uint32_t reserved = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(reader, &cursor, &reserved));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, &cursor, &section->offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, &cursor, &section->length));
    uint64_t uncompressed_length = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        reader, &cursor, &uncompressed_length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (section->flags != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("flags"), entry_offset + 2,
          IREE_SV("section has unsupported flag bits"));
    }
    if (reserved != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("reserved"), entry_offset + 4,
          IREE_SV("section reserved field must be zero"));
    }
    if (uncompressed_length != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("uncompressed_length"), entry_offset + 24,
          IREE_SV(
              "uncompressed length must be zero for uncompressed sections"));
    }
    for (uint64_t j = 0; j < i; ++j) {
      if (sections[j].kind == section->kind) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
            IREE_SV("kind"), entry_offset,
            IREE_SV("section kind appears more than once"));
      }
    }
    if (section->offset < previous_end) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("section"), section->offset, section->length,
          module->length);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("section"), section->offset, section->length,
        module->length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    section->absolute_offset = module->offset + section->offset;
    section->bytes = iree_make_const_byte_span(
        reader->bytecode.data + section->absolute_offset,
        (iree_host_size_t)section->length);
    previous_end = section->offset + section->length;
  }
  *out_sections = sections;
  *out_section_count = (iree_host_size_t)section_count;
  return iree_ok_status();
}

static const loom_bytecode_reader_section_t* loom_bytecode_reader_find_section(
    const loom_bytecode_reader_section_t* sections, iree_host_size_t count,
    uint16_t kind) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (sections[i].kind == kind) return &sections[i];
  }
  return NULL;
}

static iree_status_t loom_bytecode_reader_require_section(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* sections, iree_host_size_t count,
    uint16_t kind, const loom_bytecode_reader_section_t** out_section) {
  const loom_bytecode_reader_section_t* section =
      loom_bytecode_reader_find_section(sections, count, kind);
  if (!section) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
        IREE_SV("required_section"), 0,
        iree_make_cstring_view(loom_bytecode_section_name(kind)));
  }
  *out_section = section;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_module_metadata(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_module_t* module) {
  reader->strings = NULL;
  reader->string_count = 0;
  reader->sources = NULL;
  reader->source_count = 0;
  reader->types = NULL;
  reader->type_count = 0;
  reader->ops = NULL;
  reader->op_count = 0;
  reader->encoding_families = NULL;
  reader->encoding_family_count = 0;
  reader->encoding_count = 0;
  reader->location_count = 0;
  reader->symbol_count = 0;

  loom_bytecode_reader_section_t* sections = NULL;
  iree_host_size_t section_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_section_directory(
      reader, module, &sections, &section_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  const loom_bytecode_reader_section_t* strings_section = NULL;
  const loom_bytecode_reader_section_t* sources_section = NULL;
  const loom_bytecode_reader_section_t* encodings_section = NULL;
  const loom_bytecode_reader_section_t* types_section = NULL;
  const loom_bytecode_reader_section_t* ops_section = NULL;
  const loom_bytecode_reader_section_t* symbols_section = NULL;
  const loom_bytecode_reader_section_t* ir_section = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_STRINGS,
      &strings_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_SOURCES,
      &sources_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_ENCODINGS,
      &encodings_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_TYPES,
      &types_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_OPS,
      &ops_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_SYMBOLS,
      &symbols_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_IR, &ir_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  const loom_bytecode_reader_section_t* locations_section =
      loom_bytecode_reader_find_section(sections, section_count,
                                        LOOM_BYTECODE_SECTION_LOCATIONS);
  if (reader->result.location_mode ==
      LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    if (locations_section) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
          IREE_SV("LOCATIONS"), locations_section->absolute_offset,
          IREE_SV("NO_LOCATIONS bytecode must not contain LOCATIONS"));
    }
  } else if (!locations_section) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
        IREE_SV("LOCATIONS"), 0,
        IREE_SV("source-location bytecode must contain LOCATIONS"));
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_string_table(
      reader, *strings_section, IREE_SV("STRINGS"),
      LOOM_BYTECODE_MAX_STRING_COUNT, &reader->strings, &reader->string_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_string_table(
      reader, *sources_section, IREE_SV("SOURCES"),
      LOOM_BYTECODE_MAX_STRING_COUNT, &reader->sources, &reader->source_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_encodings(reader, encodings_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_types(reader, types_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_ops(reader, ops_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (locations_section) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_locations(reader, locations_section));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_symbols(reader, symbols_section, ir_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  if (module == &reader->modules[0]) {
    reader->result.first_module.string_count = reader->string_count;
    reader->result.first_module.source_count = reader->source_count;
    reader->result.first_module.type_count = reader->type_count;
    reader->result.first_module.encoding_count = reader->encoding_count;
    reader->result.first_module.op_name_count = reader->op_count;
    reader->result.first_module.location_count = reader->location_count;
    reader->result.first_module.symbol_count = reader->symbol_count;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_read_metadata(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result) {
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  if (!block_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "block_pool is NULL");
  }
  if (!out_result) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_result is NULL");
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_bytecode_reader_state_t reader = {
      .bytecode = bytecode,
      .filename = filename,
      .context = context,
      .arena = &arena,
      .diagnostic_context =
          {
              .sink = options ? options->diagnostic_sink
                              : (loom_diagnostic_sink_t){0},
              .filename = filename,
          },
  };

  loom_bytecode_reader_cursor_t file_cursor;
  loom_bytecode_reader_cursor_initialize(bytecode.data, bytecode.data_length, 0,
                                         IREE_SV("FILE"), &file_cursor);
  uint64_t string_pool_length = 0;
  iree_status_t status = loom_bytecode_reader_validate_file_header(
      &reader, &file_cursor, &string_pool_length);
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_module_directory(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_file_string_pool(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    for (iree_host_size_t i = 0; i < reader.module_count; ++i) {
      status = loom_bytecode_reader_read_module_metadata(&reader,
                                                         &reader.modules[i]);
      if (!iree_status_is_ok(status) ||
          loom_bytecode_reader_has_errors(&reader)) {
        break;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_result = reader.result;
  }
  iree_arena_deinitialize(&arena);
  return status;
}
