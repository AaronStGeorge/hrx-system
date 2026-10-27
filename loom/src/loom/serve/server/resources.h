// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_RESOURCES_H_
#define LOOM_SERVE_SERVER_RESOURCES_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/telemetry/sample.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed application resources available to management operations.
// The view and every referenced owner remain live through the router's join.
// Membership is immutable during routing; models can independently come and go.
typedef struct loom_serve_resources_t {
  // Application's single cooperating group, or NULL for externally owned work.
  loom_serve_device_group_t* device_group;
  // Zero disables idle reclamation; otherwise equals the group's pool count.
  iree_host_size_t idle_policy_count;
  // Borrowed immutable policies, in pool-inventory order, through the router
  // join.
  const loom_serve_pool_idle_policy_t* idle_policies;
  // Number of bound listeners published by this application.
  iree_host_size_t listener_count;
  // Borrowed array; listener storage survives stop until routing has joined.
  loom_serve_http_listener_t* const* listeners;
} loom_serve_resources_t;

// Registers the router's nonblocking scheduling wake with native capacity work
// and final output retirement, including cold workspace requests.
// An empty observer detaches and joins callback access before router teardown.
void loom_serve_resources_set_observer(const loom_serve_resources_t* resources,
                                       loom_serve_wake_callback_t observer);

// Advances accepted workspace growth/trim, publishes completed native capacity
// work, and retires parameter admission interests before services advance.
// Continue advancing during error draining until accepted work retires.
// A terminal platform failure stops admission and initiates ordinary draining;
// it never destroys shared owners.
iree_status_t loom_serve_resources_advance(
    const loom_serve_resources_t* resources);

// Whether cold workspace, native capacity work, or accepted outputs borrow
// application resources. Router shutdown joins this independently of readiness
// or semaphore failure.
bool loom_serve_resources_is_pending(const loom_serve_resources_t* resources);

// Writes a JSON object with bound listeners, immutable device identities and
// serving-managed memory observations. Called on the serialized serving owner;
// never waits for devices, polls hardware or changes residency. Retained pool
// accounting is null for fixed backing; an elastic limit is null when
// unbounded. Storage reports ordinary native backing. Workspace is a subset of
// storage for fixed backing or retained for elastic backing, not an additional
// amount. Pool values are independent atomic observations, not a coherent
// total. Acquisition/release counters accumulate across models and retired pool
// generations for the device owner's lifetime.
// wait_count counts reservations needing a reuse prerequisite, not host waits
// or elapsed time. These values exclude host snapshots, loader/driver storage,
// and hidden padding; they cannot be interpreted as device-wide usage or
// available allocation bytes.
iree_status_t loom_serve_resources_write_json(
    const loom_serve_resources_t* resources, loom_output_stream_t* output);

// Immutable device-source count, independent of model registration/residency.
iree_host_size_t loom_serve_resources_observation_count(
    const loom_serve_resources_t* resources);

// Copies one device's pool facts to a transport-independent numeric sink.
// Index comes from observation_count. Runs on the serialized serving owner
// without hardware polling or waiting; only sink failure can propagate.
// Descriptors and values are borrowed only through the synchronous emit call.
iree_status_t loom_serve_resources_sample(
    const loom_serve_resources_t* resources, iree_host_size_t device_index,
    loom_serve_telemetry_sample_sink_t sink);

// Copies one device's identity and source-owned numeric pool topics. Index is
// from observation_count and identifies this source for the resources lifetime.
// Deployment keys are bounded display labels; indices identify owners.
// Native storage and elastic backing are distinct pools. Workspace borrows
// storage with fixed backing or elastic backing otherwise; its bytes are not
// additive to that parent. These are serving-pool facts, not device-wide usage.
// Runs on the serialized serving owner without hardware polling or waiting.
// Failure is only output/allocation failure; no resource borrow escapes.
iree_status_t loom_serve_resources_write_observation(
    const loom_serve_resources_t* resources, iree_host_size_t device_index,
    iree_string_builder_t* output);

// Parses exactly {"device": <key>, "target_bytes": <unsigned integer>} and
// resolves the device key against immutable membership. Performs no mutation.
// Outputs are valid only on success and through this resource view's lifetime.
iree_status_t loom_serve_resources_parse_trim(
    const loom_serve_resources_t* resources, iree_string_view_t value,
    iree_allocator_t host_allocator, iree_host_size_t* out_device_index,
    uint64_t* out_target_bytes);

// Executes a validated trim request on the serialized serving owner and writes
// {"target_bytes": ..., "device": <resource object>} with current accounting.
// Schedules release of eligible idle workspace and unpinned parameters without
// waiting for live work, evicting mutable state, or changing the configured
// budget. The target may remain unreachable. Output is copied and retains no
// model borrow. A platform failure may leave partially released backing: the
// caller must stop admission and drain accepted work, not retry inference
// against that backing.
iree_status_t loom_serve_resources_trim(const loom_serve_resources_t* resources,
                                        iree_host_size_t device_index,
                                        uint64_t target_bytes,
                                        loom_output_stream_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_RESOURCES_H_
