// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/device_configuration.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"

static loom_serve_device_options_t loom_serve_device_configuration_defaults(
    void) {
  return (loom_serve_device_options_t){
      .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
      .weight_slab_size = UINT64_C(1) << 30,
      .state_slab_size = UINT64_C(64) << 20,
  };
}

typedef struct loom_serve_device_configuration_parser_t {
  // Output under construction; deinitialization owns partial string storage.
  loom_serve_device_configuration_t* configuration;
  // Next unused byte of input-sized decoded string storage.
  char* next_string;
} loom_serve_device_configuration_parser_t;

static iree_status_t loom_serve_device_configuration_string(
    loom_serve_device_configuration_parser_t* parser,
    iree_string_view_t escaped, iree_string_view_t* out_value) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(escaped, escaped.size,
                                                 parser->next_string, &length));
  const iree_string_view_t value =
      iree_make_string_view(parser->next_string, length);
  if (!length || memchr(value.data, 0, length) ||
      !iree_unicode_utf8_validate(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device strings must be nonempty UTF-8 without NUL");
  }
  parser->next_string += length;
  *out_value = value;
  return iree_ok_status();
}

typedef struct loom_serve_device_configuration_fields_t {
  // Decoder owning storage for all string values in this device entry.
  loom_serve_device_configuration_parser_t* parser;
  // Decoded strings or raw JSON numbers; null data marks omission.
  iree_string_view_t values[7];
} loom_serve_device_configuration_fields_t;

static const char* const loom_serve_device_configuration_field_names[] = {
    "uri",
    "pool_backing",
    "weight_slab_bytes",
    "state_slab_bytes",
    "memory_bytes",
    "workspace_idle_ms",
    "parameter_idle_ms"};

static iree_status_t loom_serve_device_configuration_field(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_fields_t* fields = self;
  iree_host_size_t i = 0;
  while (i < IREE_ARRAYSIZE(loom_serve_device_configuration_field_names) &&
         !iree_string_view_equal(
             key, iree_make_cstring_view(
                      loom_serve_device_configuration_field_names[i]))) {
    ++i;
  }
  if (i == IREE_ARRAYSIZE(loom_serve_device_configuration_field_names)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown device field '%.*s'", (int)key.size,
                            key.data);
  }
  const bool type_allowed = (type == IREE_JSON_VALUE_TYPE_STRING && i < 5) ||
                            (type == IREE_JSON_VALUE_TYPE_NUMBER && i >= 2);
  if (fields->values[i].data || !type_allowed) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate or wrongly typed device field '%s'",
                            loom_serve_device_configuration_field_names[i]);
  }
  if (type == IREE_JSON_VALUE_TYPE_STRING) {
    IREE_RETURN_IF_ERROR(loom_serve_device_configuration_string(
                             fields->parser, value, &fields->values[i]),
                         "device field '%s'",
                         loom_serve_device_configuration_field_names[i]);
  } else {
    fields->values[i] = value;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_device_configuration_entry(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_parser_t* parser = self;
  loom_serve_device_configuration_t* configuration = parser->configuration;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT ||
      configuration->count == IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "devices requires at most %d named objects",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  const iree_host_size_t index = configuration->count;
  loom_serve_device_options_t* options = &configuration->options[index];
  *options = loom_serve_device_configuration_defaults();
  IREE_RETURN_IF_ERROR(
      loom_serve_device_configuration_string(parser, key, &options->name));
  for (iree_host_size_t c = 0; c < options->name.size; ++c) {
    if ((uint8_t)options->name.data[c] < 0x21 ||
        (uint8_t)options->name.data[c] > 0x7e) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device keys must be visible ASCII");
    }
  }
  for (iree_host_size_t i = 0; i < index; ++i) {
    if (iree_string_view_equal(configuration->options[i].name, options->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate device key '%.*s'",
                              (int)options->name.size, options->name.data);
    }
  }
  loom_serve_device_configuration_fields_t fields = {.parser = parser};
  IREE_RETURN_IF_ERROR(
      iree_json_enumerate_object_typed(
          value, loom_serve_device_configuration_field, &fields),
      "device '%.*s'", (int)options->name.size, options->name.data);
  if (!fields.values[0].data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device '%.*s' requires 'uri'",
                            (int)options->name.size, options->name.data);
  }
  options->uri = fields.values[0];
  if (fields.values[1].data) {
    if (iree_string_view_equal(fields.values[1], IREE_SV("fixed"))) {
      options->backing = LOOM_SERVE_DEVICE_BACKING_FIXED;
    } else if (!iree_string_view_equal(fields.values[1], IREE_SV("elastic"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pool_backing must be elastic or fixed");
    }
  }
  uint64_t numbers[] = {options->weight_slab_size, options->state_slab_size, 0,
                        0, 0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(numbers) && iree_status_is_ok(status); ++i) {
    if (fields.values[i + 2].data) {
      if (i < 3) {
        iree_device_size_t size = 0;
        status =
            iree_string_view_parse_device_size(fields.values[i + 2], &size);
        numbers[i] = size;
      } else {
        status = iree_json_parse_uint64(fields.values[i + 2], &numbers[i]);
      }
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            status, "device '%.*s' field '%s'", (int)options->name.size,
            options->name.data,
            loom_serve_device_configuration_field_names[i + 2]);
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (numbers[3] > INT64_MAX / 1000000 || numbers[4] > INT64_MAX / 1000000) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device idle interval is out of range");
  }
  if (options->backing == LOOM_SERVE_DEVICE_BACKING_FIXED && numbers[2]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a physical budget requires elastic backing");
  }
  options->weight_slab_size = (iree_device_size_t)numbers[0];
  options->state_slab_size = (iree_device_size_t)numbers[1];
  options->memory_limit = numbers[2];
  configuration->idle[index] = (loom_serve_device_idle_policy_t){
      .workspace_timeout = (iree_duration_t)numbers[3] * 1000000,
      .parameter_timeout = (iree_duration_t)numbers[4] * 1000000};
  ++configuration->count;
  return iree_ok_status();
}

static iree_status_t loom_serve_device_configuration_root(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  iree_string_view_t* devices = self;
  if (devices->data || !iree_string_view_equal(key, IREE_SV("devices")) ||
      type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device configuration requires one 'devices' object");
  }
  *devices = value;
  return iree_ok_status();
}

iree_status_t loom_serve_device_configuration_initialize(
    iree_string_view_t json, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration) {
  *out_configuration = (loom_serve_device_configuration_t){0};
  const iree_host_size_t storage_size = json.size;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&json, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&json));
  if (json.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing device configuration JSON");
  }
  iree_string_view_t devices = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_device_configuration_root, &devices));
  if (!devices.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device configuration requires 'devices'");
  }
  char* strings = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, storage_size, (void**)&strings));
  out_configuration->host_allocator = host_allocator;
  out_configuration->strings = strings;
  loom_serve_device_configuration_parser_t parser = {
      .configuration = out_configuration,
      .next_string = out_configuration->strings};
  iree_status_t status = iree_json_enumerate_object_typed(
      devices, loom_serve_device_configuration_entry, &parser);
  if (iree_status_is_ok(status) && !out_configuration->count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "devices must contain at least one named device");
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_device_configuration_deinitialize(out_configuration);
  }
  return status;
}

iree_status_t loom_serve_device_configuration_initialize_uri(
    iree_string_view_t uri, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration) {
  *out_configuration = (loom_serve_device_configuration_t){0};
  if (!uri.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device is required; specify a HAL device URI");
  }
  if (memchr(uri.data, 0, uri.size) || !iree_unicode_utf8_validate(uri)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device URI must be UTF-8 without NUL");
  }
  char* strings = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      host_allocator, iree_make_const_byte_span(uri.data, uri.size),
      (void**)&strings));
  out_configuration->host_allocator = host_allocator;
  out_configuration->strings = strings;
  out_configuration->count = 1;
  out_configuration->options[0] = loom_serve_device_configuration_defaults();
  out_configuration->options[0].name = IREE_SV("device");
  out_configuration->options[0].uri =
      iree_make_string_view(out_configuration->strings, uri.size);
  return iree_ok_status();
}

void loom_serve_device_configuration_deinitialize(
    loom_serve_device_configuration_t* configuration) {
  iree_allocator_free(configuration->host_allocator, configuration->strings);
  memset(configuration, 0, sizeof(*configuration));
}
