// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/http/live.h"

#include <inttypes.h>

#include "loom/serve/runtime/host/timer.h"
#include "loom/serve/server/http/response.h"
#include "loom/serve/server/observation.h"
#include "loom/serve/transport/websocket/connection.h"
#include "loom/serve/transport/websocket/handshake.h"

typedef struct loom_serve_http_live_reader_t {
  // Owned claim, or NULL for a vacant reader slot.
  loom_serve_http_connection_t* connection;
  // Protocol storage borrowed through the connection claim.
  loom_serve_websocket_t* socket;
  // Next current-state page; never a pointer into a model or event stream.
  iree_host_size_t cursor;
  // Sampling generation at the start of the current sweep.
  uint64_t generation;
  // Last delivered sequence, echoed verbatim by the client after application.
  uint64_t sequence;
  // A delivered page has not yet been applied by the client.
  bool awaiting_ack;
} loom_serve_http_live_reader_t;

struct loom_serve_http_live_t {
  // Allocator owning this object and its fixed reader table.
  iree_allocator_t host_allocator;
  // Existing HTTP owner, borrowed through final observer retirement.
  loom_serve_http_server_t* server;
  // Registration truth, consulted only on the serialized serving turn.
  loom_serve_registry_t* registry;
  // Externally visible origins, borrowed from application configuration.
  iree_string_view_list_t origins;
  // Shared proactor alarm, enabled only while readers exist.
  loom_serve_timer_t timer;
  // Monotonic sampling generation, independent of reader acknowledgement.
  uint64_t generation;
  // Total slots, one fewer than transport capacity to reserve control credit.
  iree_host_size_t capacity;
  // Occupied reader slots.
  iree_host_size_t count;
  // Rotating reader priority under the router's per-turn quota.
  iree_host_size_t cursor;
  // Shared bounded formatting scratch, not an output queue. Framing plus this
  // payload stays inside the carrier's ordinary 16 KiB generated prefix slot.
  char storage[16368];
  // Fixed metadata, with no per-source or per-event reader allocation.
  loom_serve_http_live_reader_t readers[];
};

iree_status_t loom_serve_http_live_allocate(loom_serve_http_server_t* server,
                                            loom_serve_registry_t* registry,
                                            iree_string_view_list_t origins,
                                            loom_serve_wake_callback_t observer,
                                            iree_allocator_t host_allocator,
                                            loom_serve_http_live_t** out_live) {
  *out_live = NULL;
  const iree_host_size_t capacity =
      loom_serve_http_server_connection_capacity(server) - 1;
  loom_serve_http_live_t* live = NULL;
  // Transport already allocated its larger per-connection table for this
  // capacity. Reader metadata and bounded shared scratch share one allocation.
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*live) + capacity * sizeof(*live->readers),
      (void**)&live));
  live->host_allocator = host_allocator;
  live->server = server;
  live->registry = registry;
  live->origins = origins;
  live->capacity = capacity;
  loom_serve_timer_initialize(&live->timer);
  iree_async_proactor_t* proactor = NULL;
  iree_status_t status = iree_async_proactor_pool_get(
      loom_serve_http_server_proactor_pool(server), 0, &proactor);
  if (iree_status_is_ok(status)) {
    loom_serve_timer_configure(&live->timer, proactor,
                               iree_make_duration_ms(250));
    loom_serve_timer_set_observer(&live->timer, observer);
    *out_live = live;
  } else {
    loom_serve_timer_deinitialize(&live->timer);
    iree_allocator_free(host_allocator, live);
  }
  return status;
}

static void loom_serve_http_live_failure(
    loom_serve_http_live_t* live, loom_serve_http_connection_t* connection,
    iree_status_t status) {
  loom_serve_http_server_record_error(live->server, status);
  iree_status_free(status);
  loom_serve_http_connection_abort(connection);
}

void loom_serve_http_live_accept(loom_serve_http_live_t* live,
                                 loom_serve_http_connection_t* connection,
                                 const loom_serve_http_request_t* request) {
  iree_string_builder_t output;
  iree_string_builder_initialize_with_storage(live->storage,
                                              sizeof(live->storage), &output);
  iree_status_t status = loom_serve_websocket_handshake(
      request, live->origins, IREE_SV("loom.observe.v1"), &output);
  if (!iree_status_is_ok(status)) {
    const bool denied = iree_status_is_permission_denied(status);
    iree_status_t response = loom_serve_http_response_reject(
        live->server, connection, denied ? 403 : 400,
        denied ? "Forbidden" : "Bad Request", iree_status_message(status),
        &output);
    iree_status_free(status);
    if (!iree_status_is_ok(response)) {
      // The response helper consumed the claim on both success and failure.
      loom_serve_http_server_record_error(live->server, response);
      iree_status_free(response);
    }
  } else {
    loom_serve_websocket_t* socket = NULL;
    status = loom_serve_websocket_upgrade(
        connection,
        (loom_serve_websocket_limits_t){
            .message_byte_capacity = sizeof(live->storage),
            .retained_byte_capacity = sizeof(live->storage),
            .message_capacity = 4},
        live->host_allocator, &socket);
    if (iree_status_is_ok(status)) {
      status = loom_serve_http_connection_send(
          connection, iree_string_builder_view(&output));
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < live->capacity; ++i) {
        if (!live->readers[i].connection) {
          live->readers[i] = (loom_serve_http_live_reader_t){
              .connection = connection,
              .socket = socket,
              .generation = live->generation - 1};
          ++live->count;
          break;
        }
      }
    } else {
      loom_serve_http_live_failure(live, connection, status);
    }
  }
  iree_string_builder_deinitialize(&output);
}

static iree_status_t loom_serve_http_live_receive(
    loom_serve_http_live_reader_t* reader, bool* out_progress) {
  loom_serve_websocket_message_type_t type;
  iree_const_byte_span_t payload;
  loom_serve_websocket_message_t* message =
      loom_serve_websocket_take(reader->socket, &type, &payload);
  if (!message) {
    return iree_ok_status();
  }
  *out_progress = true;
  iree_status_t status = iree_ok_status();
  const bool closing = type == LOOM_SERVE_WEBSOCKET_CLOSE;
  if (type == LOOM_SERVE_WEBSOCKET_TEXT) {
    uint64_t sequence = 0;
    const iree_string_view_t text =
        iree_make_string_view((const char*)payload.data, payload.data_length);
    if (!reader->awaiting_ack ||
        !iree_string_view_atoi_uint64(text, &sequence) ||
        sequence != reader->sequence) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "expected acknowledgement of the delivered sequence");
    } else {
      reader->awaiting_ack = false;
    }
  } else if (type == LOOM_SERVE_WEBSOCKET_PING) {
    status = loom_serve_websocket_send(reader->socket,
                                       LOOM_SERVE_WEBSOCKET_PONG, payload);
  } else if (type == LOOM_SERVE_WEBSOCKET_CLOSE) {
    status = loom_serve_websocket_send(reader->socket,
                                       LOOM_SERVE_WEBSOCKET_CLOSE, payload);
  } else if (type != LOOM_SERVE_WEBSOCKET_PONG) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "observation acknowledgements must be text");
  }
  loom_serve_websocket_message_release(reader->socket, message);
  if (closing && iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(reader->connection);
    reader->connection = NULL;
  }
  return status;
}

iree_status_t loom_serve_http_live_advance(loom_serve_http_live_t* live,
                                           bool stopping,
                                           iree_host_size_t quota,
                                           bool* out_progress) {
  if (!live) {
    return iree_ok_status();
  }
  bool tick = false;
  bool timer_progress = false;
  iree_status_t timer_status = loom_serve_timer_advance(
      &live->timer, stopping, live->count != 0, &timer_progress, &tick);
  *out_progress |= timer_progress;
  stopping |= !iree_status_is_ok(timer_status);
  if (tick) {
    ++live->generation;
  }
  const iree_host_size_t start = live->cursor;
  for (iree_host_size_t i = 0; i < live->capacity; ++i) {
    const iree_host_size_t index = (start + i) % live->capacity;
    loom_serve_http_live_reader_t* reader = &live->readers[index];
    if (!reader->connection) {
      continue;
    }
    if (stopping || loom_serve_http_connection_failed(reader->connection)) {
      loom_serve_http_connection_abort(reader->connection);
      reader->connection = NULL;
      --live->count;
      *out_progress = true;
      continue;
    }
    if (!quota || !loom_serve_http_connection_can_send(reader->connection)) {
      continue;
    }
    --quota;
    live->cursor = (index + 1) % live->capacity;
    iree_status_t status = loom_serve_http_live_receive(reader, out_progress);
    if (reader->connection && iree_status_is_ok(status) &&
        !reader->awaiting_ack &&
        loom_serve_http_connection_can_send(reader->connection) &&
        (reader->cursor || reader->generation != live->generation)) {
      if (!reader->cursor) {
        reader->generation = live->generation;
      }
      iree_string_builder_t output;
      iree_string_builder_initialize_with_storage(
          live->storage, sizeof(live->storage), &output);
      const loom_serve_telemetry_identity_t identity =
          loom_serve_telemetry_stream_identity(
              loom_serve_http_server_diagnostics(live->server));
      status = iree_string_builder_append_format(
          &output,
          "{\"version\":1,\"server\":\"%" PRIu64 ":%" PRIu64
          "\",\"sequence\":\"%" PRIu64 "\",\"time_ns\":\"%" PRId64
          "\",\"page\":",
          identity.created_ns, identity.ordinal, ++reader->sequence,
          iree_time_now());
      if (iree_status_is_ok(status)) {
        status = loom_serve_observation_write_page(live->registry,
                                                   &reader->cursor, &output);
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(&output, "}");
      }
      if (iree_status_is_ok(status)) {
        const iree_string_view_t view = iree_string_builder_view(&output);
        status = loom_serve_websocket_send(
            reader->socket, LOOM_SERVE_WEBSOCKET_TEXT,
            iree_make_const_byte_span(view.data, view.size));
        reader->awaiting_ack = iree_status_is_ok(status);
      }
      iree_string_builder_deinitialize(&output);
      *out_progress = true;
    }
    if (!iree_status_is_ok(status)) {
      loom_serve_http_live_failure(live, reader->connection, status);
      reader->connection = NULL;
    }
    if (!reader->connection) {
      --live->count;
    }
  }
  return timer_status;
}

iree_host_size_t loom_serve_http_live_count(
    const loom_serve_http_live_t* live) {
  return live ? live->count : 0;
}

bool loom_serve_http_live_is_pending(loom_serve_http_live_t* live) {
  return live && (live->count || loom_serve_timer_is_pending(&live->timer));
}

void loom_serve_http_live_free(loom_serve_http_live_t* live) {
  if (!live) {
    return;
  }
  loom_serve_timer_set_observer(&live->timer, (loom_serve_wake_callback_t){0});
  loom_serve_timer_deinitialize(&live->timer);
  iree_allocator_free(live->host_allocator, live);
}
