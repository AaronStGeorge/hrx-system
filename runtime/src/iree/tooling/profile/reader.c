// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/reader.h"

#include <inttypes.h>
#include <string.h>

typedef struct iree_profile_session_state_t {
  // Process-local identity declared by the session-begin record.
  iree_hal_profile_session_id_t session_id;
  // Source record index of the unique session-begin record.
  iree_host_size_t begin_record_index;
  // True until the matching session-end record is observed.
  bool is_active;
} iree_profile_session_state_t;

typedef struct iree_profile_session_lookup_t {
  // Session state rows indexed by the active lookup.
  const iree_profile_session_state_t* sessions;
  // Process-local session identity being resolved.
  iree_hal_profile_session_id_t session_id;
} iree_profile_session_lookup_t;

static bool iree_profile_session_matches(const void* user_data,
                                         iree_host_size_t value) {
  const iree_profile_session_lookup_t* lookup =
      (const iree_profile_session_lookup_t*)user_data;
  return lookup->sessions[value].session_id == lookup->session_id;
}

static iree_profile_session_state_t* iree_profile_find_session_state(
    const iree_profile_index_t* session_index,
    iree_profile_session_state_t* sessions,
    iree_hal_profile_session_id_t session_id) {
  const iree_profile_session_lookup_t lookup = {
      .sessions = sessions,
      .session_id = session_id,
  };
  iree_host_size_t session_state_index = 0;
  if (!iree_profile_index_find(
          session_index, iree_profile_index_mix_u64(session_id),
          iree_profile_session_matches, &lookup, &session_state_index)) {
    return NULL;
  }
  return &sessions[session_state_index];
}

static iree_status_t iree_profile_validate_record_session(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    iree_allocator_t host_allocator, iree_profile_index_t* session_index,
    iree_profile_session_state_t** sessions, iree_host_size_t* session_count,
    iree_host_size_t* session_capacity) {
  const iree_hal_profile_file_record_type_t record_type =
      record->header.record_type;
  if (record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN &&
      record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK &&
      record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END) {
    return iree_ok_status();
  }

  const iree_hal_profile_session_id_t session_id = record->header.session_id;
  if (session_id == IREE_HAL_PROFILE_SESSION_ID_NONE) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile record %" PRIhsz
                            " has no profiling session identifier",
                            record_index);
  }

  iree_profile_session_state_t* session =
      iree_profile_find_session_state(session_index, *sessions, session_id);
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      if (session) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "profile session %" PRIu64
            " begins more than once at records %" PRIhsz " and %" PRIhsz,
            session_id, session->begin_record_index, record_index);
      }
      if (*session_count + 1 > *session_capacity) {
        IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
            host_allocator, iree_max((iree_host_size_t)4, *session_count + 1),
            sizeof((*sessions)[0]), session_capacity, (void**)sessions));
      }
      const iree_host_size_t session_state_index = *session_count;
      IREE_RETURN_IF_ERROR(iree_profile_index_insert(
          session_index, host_allocator, iree_profile_index_mix_u64(session_id),
          session_state_index));
      session = &(*sessions)[session_state_index];
      *session = (iree_profile_session_state_t){
          .session_id = session_id,
          .begin_record_index = record_index,
          .is_active = true,
      };
      ++*session_count;
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      if (!session) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "profile chunk record %" PRIhsz
                                " belongs to session %" PRIu64
                                " with no matching begin",
                                record_index, session_id);
      }
      if (!session->is_active) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "profile chunk record %" PRIhsz
                                " belongs to completed session %" PRIu64,
                                record_index, session_id);
      }
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      if (!session) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "profile session end record %" PRIhsz
                                " names session %" PRIu64
                                " with no matching begin",
                                record_index, session_id);
      }
      if (!session->is_active) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "profile session %" PRIu64
            " ends more than once; repeated end is record %" PRIhsz,
            session_id, record_index);
      }
      session->is_active = false;
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

static iree_status_t iree_profile_payload_record_length(
    iree_string_view_t content_type, iree_const_byte_span_t payload,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t* out_record_length) {
  *out_record_length = 0;

  if (payload_offset > payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile chunk '%.*s' has a typed record offset past the payload",
        (int)content_type.size, content_type.data);
  }

  const iree_host_size_t remaining_length =
      payload.data_length - payload_offset;
  if (remaining_length < minimum_record_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile chunk '%.*s' has a truncated typed record",
                            (int)content_type.size, content_type.data);
  }

  uint32_t record_length = 0;
  memcpy(&record_length, payload.data + payload_offset, sizeof(record_length));
  if (record_length < minimum_record_length ||
      record_length > remaining_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile chunk '%.*s' has invalid typed record length %u",
        (int)content_type.size, content_type.data, record_length);
  }

  *out_record_length = record_length;
  return iree_ok_status();
}

iree_status_t iree_profile_typed_record_parse(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t record_index, iree_profile_typed_record_t* out_record) {
  memset(out_record, 0, sizeof(*out_record));

  if (minimum_record_length < sizeof(uint32_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "minimum typed record length must include record_length");
  }

  iree_host_size_t record_length = 0;
  IREE_RETURN_IF_ERROR(iree_profile_payload_record_length(
      chunk->content_type, chunk->payload, payload_offset,
      minimum_record_length, &record_length));

  const uint8_t* record_data = chunk->payload.data + payload_offset;
  out_record->chunk = chunk;
  out_record->record_index = record_index;
  out_record->payload_offset = payload_offset;
  out_record->minimum_record_length = minimum_record_length;
  out_record->record_length = record_length;
  out_record->contents = iree_make_const_byte_span(record_data, record_length);
  out_record->inline_payload =
      iree_make_const_byte_span(record_data + minimum_record_length,
                                record_length - minimum_record_length);
  out_record->following_payload = iree_make_const_byte_span(
      record_data + record_length,
      chunk->payload.data_length - payload_offset - record_length);
  return iree_ok_status();
}

iree_status_t iree_profile_executable_trace_record_parse(
    const iree_hal_profile_file_record_t* chunk,
    iree_hal_profile_executable_trace_record_t* out_record,
    iree_const_byte_span_t* out_trace_data) {
  memset(out_record, 0, sizeof(*out_record));
  if (out_trace_data) {
    *out_trace_data = iree_const_byte_span_empty();
  }

  iree_profile_typed_record_t typed_record;
  IREE_RETURN_IF_ERROR(iree_profile_typed_record_parse(
      chunk, 0, sizeof(*out_record), 0, &typed_record));
  if (typed_record.record_length != sizeof(*out_record)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile executable trace chunk record length must exclude raw trace "
        "bytes");
  }

  memcpy(out_record, typed_record.contents.data, sizeof(*out_record));
  if ((iree_host_size_t)out_record->data_length !=
      typed_record.following_payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile executable trace chunk data length is inconsistent with "
        "payload length");
  }

  if (out_trace_data) {
    *out_trace_data = typed_record.following_payload;
  }
  return iree_ok_status();
}

void iree_profile_typed_record_iterator_initialize(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t minimum_record_length,
    iree_profile_typed_record_iterator_t* out_iterator) {
  memset(out_iterator, 0, sizeof(*out_iterator));
  out_iterator->chunk = chunk;
  out_iterator->minimum_record_length = minimum_record_length;
}

iree_status_t iree_profile_typed_record_iterator_next(
    iree_profile_typed_record_iterator_t* iterator,
    iree_profile_typed_record_t* out_record, bool* out_has_record) {
  *out_has_record = false;
  memset(out_record, 0, sizeof(*out_record));

  if (iterator->payload_offset >= iterator->chunk->payload.data_length) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_profile_typed_record_parse(
      iterator->chunk, iterator->payload_offset,
      iterator->minimum_record_length, iterator->record_index, out_record));
  iterator->payload_offset += out_record->record_length;
  ++iterator->record_index;
  *out_has_record = true;
  return iree_ok_status();
}

iree_status_t iree_profile_file_open(iree_string_view_t path,
                                     iree_allocator_t host_allocator,
                                     iree_profile_file_t* out_profile_file) {
  memset(out_profile_file, 0, sizeof(*out_profile_file));
  out_profile_file->host_allocator = host_allocator;

  iree_status_t status =
      iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ, host_allocator,
                                &out_profile_file->contents);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(
        out_profile_file->contents->const_buffer, &out_profile_file->header,
        &out_profile_file->first_record_offset);
  }
  if (iree_status_is_ok(status)) {
    uint64_t file_length = out_profile_file->header.file_length;
    if (file_length == 0) {
      file_length = out_profile_file->contents->const_buffer.data_length;
    }
    if (IREE_UNLIKELY(file_length > IREE_HOST_SIZE_MAX)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "profile file length exceeds host size");
    } else {
      out_profile_file->file_length = (iree_host_size_t)file_length;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_profile_file_close(out_profile_file);
  }
  return status;
}

void iree_profile_file_close(iree_profile_file_t* profile_file) {
  iree_io_file_contents_free(profile_file->contents);
  memset(profile_file, 0, sizeof(*profile_file));
}

iree_status_t iree_profile_file_for_each_record(
    iree_profile_file_t* profile_file,
    iree_profile_file_record_callback_t callback) {
  if (!callback.fn) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "profile file record callback must have a function");
  }

  iree_host_size_t file_length = profile_file->file_length;
  if (file_length == 0) {
    file_length = profile_file->contents->const_buffer.data_length;
  }
  iree_host_size_t record_offset = profile_file->first_record_offset;
  iree_host_size_t record_index = 0;
  iree_profile_session_state_t* sessions = NULL;
  iree_host_size_t session_count = 0;
  iree_host_size_t session_capacity = 0;
  iree_profile_index_t session_index = {0};
  const bool validate_session_lifecycle =
      !profile_file->session_lifecycle_validated;
  iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      profile_file->contents->const_buffer.data, file_length);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && record_offset < file_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents, record_offset,
                                                &record, &next_record_offset);
    if (iree_status_is_ok(status) && validate_session_lifecycle) {
      status = iree_profile_validate_record_session(
          &record, record_index, profile_file->host_allocator, &session_index,
          &sessions, &session_count, &session_capacity);
    }
    if (iree_status_is_ok(status)) {
      status = callback.fn(callback.user_data, &record, record_index);
    }
    if (iree_status_is_ok(status)) {
      record_offset = next_record_offset;
      ++record_index;
    }
  }
  if (iree_status_is_ok(status) && validate_session_lifecycle) {
    for (iree_host_size_t i = 0; i < session_count; ++i) {
      if (sessions[i].is_active) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "profile session %" PRIu64 " beginning at record %" PRIhsz
            " has no matching end",
            sessions[i].session_id, sessions[i].begin_record_index);
        break;
      }
    }
  }
  if (iree_status_is_ok(status) && validate_session_lifecycle) {
    profile_file->session_lifecycle_validated = true;
  }
  iree_profile_index_deinitialize(&session_index, profile_file->host_allocator);
  iree_allocator_free(profile_file->host_allocator, sessions);
  return status;
}
