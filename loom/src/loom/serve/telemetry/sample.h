// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TELEMETRY_SAMPLE_H_
#define LOOM_SERVE_TELEMETRY_SAMPLE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_telemetry_metric_kind_e {
  // Current ownership or occupancy; samples may decrease.
  LOOM_SERVE_TELEMETRY_GAUGE = 0,
  // Cumulative work within the source incarnation, independent of readers.
  LOOM_SERVE_TELEMETRY_COUNTER = 1,
} loom_serve_telemetry_metric_kind_t;

// Immutable source-owned numeric semantics. Names are stable lowercase ASCII
// identifiers; units describe the integer value, without implicit conversion.
// A topic's field order and meaning are stable for its schema version.
typedef struct loom_serve_telemetry_metric_t {
  // Field identifier, unique within the topic.
  const char* name;
  // Unit such as requests, tokens, bytes, pixels, or nanoseconds.
  const char* unit;
  // Gauge versus cumulative counter; neither implies a retirement reduction.
  loom_serve_telemetry_metric_kind_t kind;
} loom_serve_telemetry_metric_t;

typedef struct loom_serve_telemetry_topic_t {
  // Stable source-owned namespace, independent of a transport or model name.
  const char* name;
  // Schema revision for the field meanings and order.
  uint32_t version;
  // Number of fields and corresponding sampled values.
  iree_host_size_t count;
  // Immutable field descriptors, live through the synchronous sink call.
  const loom_serve_telemetry_metric_t* fields;
} loom_serve_telemetry_topic_t;

// Synchronous consumer of copied owner facts. Descriptors and values are
// borrowed only during emit; an exporter copies or formats before returning.
// The callback can fail only for its output/allocation, not model execution.
// Providers sample on their serialized owner, never wait for a device, and
// never expose task-borrowed state. No history is synthesized from samples.
typedef struct loom_serve_telemetry_sample_sink_t {
  // Consumer context, borrowed through the sampling call.
  void* self;
  // Consumes exactly topic->count unsigned 64-bit values.
  iree_status_t (*emit)(void* self, const loom_serve_telemetry_topic_t* topic,
                        const uint64_t* values);
} loom_serve_telemetry_sample_sink_t;

// Appends one self-describing topic object. Integer values are decimal strings
// so JavaScript clients can difference counters without losing low bits.
// This is consumer-side projection; producers have no JSON dependency.
iree_status_t loom_serve_telemetry_sample_write_json(
    const loom_serve_telemetry_topic_t* topic, const uint64_t* values,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_TELEMETRY_SAMPLE_H_
