// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TRANSPORT_HTTP_ORIGIN_H_
#define LOOM_SERVE_TRANSPORT_HTTP_ORIGIN_H_

#include "loom/serve/transport/http/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Established by the accepting transport, never by a request header.
typedef enum loom_serve_http_origin_source_e {
  // Plain HTTP. Automatic browser admission requires a numeric IP or localhost.
  LOOM_SERVE_HTTP_ORIGIN_DIRECT = 0,
  // A trusted relay validates public Host names and controls access to its
  // backend carrier. It preserves Host/Origin and replaces X-Forwarded-Proto
  // with the public http/https scheme; absence means http.
  LOOM_SERVE_HTTP_ORIGIN_RELAY = 1,
} loom_serve_http_origin_source_t;

typedef struct loom_serve_http_origin_policy_t {
  // Immutable transport provenance for this connection.
  loom_serve_http_origin_source_t source;
  // Additional exact serialized origins deliberately authorized by the owner.
  // Borrowed, prevalidated configuration, not authentication or a CORS policy.
  iree_string_view_list_t allowed_origins;
} loom_serve_http_origin_policy_t;

// Validates configuration before accepting connections. Entries are http/https
// origins with no userinfo, path, query, fragment, wildcard, or opaque value.
iree_status_t loom_serve_http_origin_validate_allowed(
    iree_string_view_list_t allowed_origins);

// Checks browser admission at an external request boundary. Missing Origin
// permits native clients. Otherwise an exact configured origin or automatic
// same-origin admission is required. Automatic admission compares the public
// scheme and Host authority (including the effective port), not the listener's
// port, and rejects conflicting Sec-Fetch-Site metadata. Direct connections
// never trust forwarded headers or infer arbitrary DNS names to be safe.
// No allocation, DNS lookup, or connection-owned state is needed.
iree_status_t loom_serve_http_origin_check(
    const loom_serve_http_request_t* request,
    loom_serve_http_origin_policy_t policy);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TRANSPORT_HTTP_ORIGIN_H_
