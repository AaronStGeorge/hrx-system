// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/http/origin.h"

#include "iree/async/address.h"

typedef struct loom_serve_http_origin_t {
  // Borrowed host, retaining IPv6 brackets for unambiguous comparison.
  iree_string_view_t host;
  // Explicit port or the scheme's default.
  uint16_t port;
  // HTTPS rather than HTTP.
  bool secure;
  // Numeric IP literal or the exact localhost name; no DNS resolution required.
  bool dns_independent;
} loom_serve_http_origin_t;

static iree_status_t loom_serve_http_origin_parse_authority(
    iree_string_view_t authority, loom_serve_http_origin_t* origin) {
  iree_string_view_t port = iree_string_view_empty();
  bool has_port = false;
  if (iree_string_view_starts_with(authority, IREE_SV("["))) {
    const iree_host_size_t end = iree_string_view_find_char(authority, ']', 1);
    if (end == IREE_STRING_VIEW_NPOS || end == 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP authority requires a bracketed IPv6 host");
    }
    origin->host = iree_string_view_substr(authority, 0, end + 1);
    if (iree_string_view_find_char(origin->host, '%', 0) !=
        IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP authority cannot contain an IPv6 zone");
    }
    iree_async_address_t address;
    IREE_RETURN_IF_ERROR(iree_async_address_from_ipv6(
        iree_string_view_substr(authority, 1, end - 1), 0, &address));
    origin->dns_independent = true;
    if (end + 1 < authority.size) {
      if (authority.data[end + 1] != ':') {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HTTP authority has trailing IPv6 host data");
      }
      has_port = true;
      port = iree_string_view_substr(authority, end + 2, IREE_HOST_SIZE_MAX);
    }
  } else {
    const iree_host_size_t colon =
        iree_string_view_find_char(authority, ':', 0);
    origin->host = iree_string_view_substr(authority, 0, colon);
    has_port = colon != IREE_STRING_VIEW_NPOS;
    if (has_port) {
      port = iree_string_view_substr(authority, colon + 1, IREE_HOST_SIZE_MAX);
    }
    if (iree_string_view_is_empty(origin->host)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP authority requires a host");
    }
    bool numeric = true;
    for (iree_host_size_t i = 0; i < origin->host.size; ++i) {
      const char c = origin->host.data[i];
      const bool digit = c >= '0' && c <= '9';
      const bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
      if (!digit && !letter && c != '-' && c != '.' && c != '_') {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HTTP authority contains an invalid host");
      }
      numeric &= digit || c == '.';
    }
    origin->dns_independent =
        iree_string_view_equal_case(origin->host, IREE_SV("localhost"));
    if (numeric) {
      iree_async_address_t address;
      IREE_RETURN_IF_ERROR(
          iree_async_address_from_ipv4(origin->host, 0, &address));
      origin->dns_independent = true;
    }
  }
  origin->port = origin->secure ? 443 : 80;
  if (has_port) {
    uint32_t number = 0;
    if (iree_string_view_is_empty(port)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP authority has an empty port");
    }
    for (iree_host_size_t i = 0; i < port.size; ++i) {
      const char c = port.data[i];
      if (c < '0' || c > '9') {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HTTP authority port must contain digits");
      }
      number = number * 10 + (uint32_t)(c - '0');
      if (number > 65535) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HTTP authority port must be in [0, 65535]");
      }
    }
    origin->port = (uint16_t)number;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_origin_parse(
    iree_string_view_t value, loom_serve_http_origin_t* origin) {
  *origin = (loom_serve_http_origin_t){0};
  if (iree_string_view_consume_prefix(&value, IREE_SV("https://"))) {
    origin->secure = true;
  } else if (!iree_string_view_consume_prefix(&value, IREE_SV("http://"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP Origin requires http:// or https://");
  }
  return loom_serve_http_origin_parse_authority(value, origin);
}

iree_status_t loom_serve_http_origin_validate_allowed(
    iree_string_view_list_t allowed_origins) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < allowed_origins.count && iree_status_is_ok(status); ++i) {
    loom_serve_http_origin_t origin;
    status = loom_serve_http_origin_parse(allowed_origins.values[i], &origin);
  }
  return status;
}

iree_status_t loom_serve_http_origin_check(
    const loom_serve_http_request_t* request,
    loom_serve_http_origin_policy_t policy) {
  iree_string_view_t value;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Origin"),
                                             &value)) {
    return iree_ok_status();
  }
  loom_serve_http_origin_t origin;
  IREE_RETURN_IF_ERROR(loom_serve_http_origin_parse(value, &origin));
  for (iree_host_size_t i = 0; i < policy.allowed_origins.count; ++i) {
    if (iree_string_view_equal(value, policy.allowed_origins.values[i])) {
      return iree_ok_status();
    }
  }

  loom_serve_http_origin_t target = {0};
  if (policy.source == LOOM_SERVE_HTTP_ORIGIN_RELAY &&
      loom_serve_http_request_lookup_header(
          request, IREE_SV("X-Forwarded-Proto"), &value)) {
    if (iree_string_view_equal(value, IREE_SV("https"))) {
      target.secure = true;
    } else if (!iree_string_view_equal(value, IREE_SV("http"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "relay X-Forwarded-Proto must be http or https");
    }
  }
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Host"),
                                             &value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP Origin admission requires Host");
  }
  IREE_RETURN_IF_ERROR(loom_serve_http_origin_parse_authority(value, &target));
  if (origin.secure != target.secure || origin.port != target.port ||
      !iree_string_view_equal_case(origin.host, target.host) ||
      (policy.source == LOOM_SERVE_HTTP_ORIGIN_DIRECT &&
       !target.dns_independent)) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "browser Origin is not an allowed server origin");
  }
  if (loom_serve_http_request_lookup_header(request, IREE_SV("Sec-Fetch-Site"),
                                            &value) &&
      !iree_string_view_equal(value, IREE_SV("same-origin"))) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "browser request is not same-origin");
  }
  return iree_ok_status();
}
