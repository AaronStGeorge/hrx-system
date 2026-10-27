// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_DEVICE_CONFIGURATION_H_
#define LOOM_SERVE_SERVER_DEVICE_CONFIGURATION_H_

#include "loom/serve/runtime/device/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// Machine-local device declarations, independent of portable model programs.
// Native drivers are resolved by the application after parsing succeeds.
typedef struct loom_serve_device_configuration_t {
  // Allocator owning decoded strings.
  iree_allocator_t host_allocator;
  // Single allocation holding all decoded strings.
  char* strings;
  // Number of populated entries in both arrays below.
  iree_host_size_t count;
  // Native creation inputs; driver remains NULL until application resolution.
  loom_serve_device_options_t options[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  // Independent owner-side idle policies in the same order as options.
  loom_serve_pool_idle_policy_t idle[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
} loom_serve_device_configuration_t;

// Parses {"devices":{"key":{"uri":"...",...}}}. Device entries support
// pool_backing, weight_slab_bytes, state_slab_bytes, memory_bytes,
// workspace_idle_ms, and parameter_idle_ms. Defaults are elastic backing,
// 1 GiB weight slabs, 64 MiB state slabs, no byte limit, and disabled expiry.
// Byte counts accept integer JSON numbers or strings parsed by
// iree_string_view_parse_device_size, such as "64gib". Idle durations remain
// integer JSON numbers in milliseconds.
// Limits are per record, not a combined physical-memory-domain budget.
// Rejects duplicate/unknown keys, invalid types and unsupported combinations
// before creating a driver or device. Copies strings and zeros output on error.
iree_status_t loom_serve_device_configuration_initialize(
    iree_string_view_t json, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration);

// Constructs the single-device convenience declaration, named "device", with
// the same defaults as JSON. Copies the URI; does not resolve or open hardware.
iree_status_t loom_serve_device_configuration_initialize_uri(
    iree_string_view_t uri, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration);

void loom_serve_device_configuration_deinitialize(
    loom_serve_device_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_DEVICE_CONFIGURATION_H_
