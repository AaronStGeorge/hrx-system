// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/dashboard/assets.h"

#include <string.h>

#include "loom/serve/dashboard/files.h"
#include "loom/serve/server/http/response.h"

static const iree_file_toc_t* loom_serve_dashboard_lookup(
    iree_string_view_t target) {
  if (!iree_string_view_consume_prefix(&target, IREE_SV("/dashboard/"))) {
    return NULL;
  }
  if (!target.size) {
    target = IREE_SV("index.html");
  }
  const iree_file_toc_t* files = loom_serve_dashboard_files_create();
  for (iree_host_size_t i = 0; i < loom_serve_dashboard_files_size(); ++i) {
    if (iree_string_view_equal(target, iree_make_cstring_view(files[i].name))) {
      return &files[i];
    }
  }
  return NULL;
}

bool loom_serve_dashboard_handles(iree_string_view_t target) {
  return iree_string_view_equal(target, IREE_SV("/")) ||
         iree_string_view_equal(target, IREE_SV("/dashboard")) ||
         loom_serve_dashboard_lookup(target) != NULL;
}

typedef struct loom_serve_dashboard_response_t {
  // Stack header borrowed by the synchronous generated-send writer.
  iree_string_view_t header;
  // Immutable embedded file with process lifetime.
  const iree_file_toc_t* file;
} loom_serve_dashboard_response_t;

static iree_status_t loom_serve_dashboard_write(void* self,
                                                iree_byte_span_t storage) {
  const loom_serve_dashboard_response_t* response = self;
  memcpy(storage.data, response->header.data, response->header.size);
  memcpy(storage.data + response->header.size, response->file->data,
         response->file->size);
  return iree_ok_status();
}

void loom_serve_dashboard_send(loom_serve_http_server_t* server,
                               loom_serve_http_connection_t* connection,
                               iree_string_view_t target) {
  if (iree_string_view_equal(target, IREE_SV("/")) ||
      iree_string_view_equal(target, IREE_SV("/dashboard"))) {
    loom_serve_http_response_send(
        server, connection,
        IREE_SV("HTTP/1.1 302 Found\r\nLocation: ./dashboard/\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n"));
    return;
  }
  const iree_file_toc_t* file = loom_serve_dashboard_lookup(target);
  const char* content_type = "text/html; charset=utf-8";
  if (iree_string_view_equal(target, IREE_SV("/dashboard/style.css"))) {
    content_type = "text/css; charset=utf-8";
  } else if (iree_string_view_equal(target, IREE_SV("/dashboard/app.js"))) {
    content_type = "text/javascript; charset=utf-8";
  }
  char header[512];
  iree_string_builder_t output;
  iree_string_builder_initialize_with_storage(header, sizeof(header), &output);
  iree_status_t status = iree_string_builder_append_format(
      &output,
      "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
      "Cache-Control: no-cache\r\nX-Content-Type-Options: nosniff\r\n"
      "Content-Security-Policy: default-src 'none'; script-src 'self'; "
      "style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors "
      "'none'\r\n"
      "Connection: close\r\n\r\n",
      content_type, file->size);
  if (iree_status_is_ok(status) &&
      loom_serve_http_connection_can_send(connection)) {
    const loom_serve_dashboard_response_t response = {
        iree_string_builder_view(&output), file};
    status = loom_serve_http_connection_send_generated(
        connection,
        (iree_net_send_prefix_t){response.header.size + file->size,
                                 loom_serve_dashboard_write, (void*)&response});
  } else if (iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    iree_string_builder_deinitialize(&output);
    return;
  }
  iree_string_builder_deinitialize(&output);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
  } else {
    loom_serve_http_server_record_error(server, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(connection);
  }
}
