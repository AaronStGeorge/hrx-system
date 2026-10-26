// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/observation.h"

#include <inttypes.h>

#include "loom/serve/service/kind.h"
#include "loom/serve/telemetry/sample.h"
#include "loom/util/json.h"

static iree_status_t loom_serve_observation_counters(
    const loom_serve_telemetry_counters_t* counters,
    iree_string_builder_t* output) {
  if (!counters) {
    return iree_string_builder_append_cstring(output, "null");
  }
  return iree_string_builder_append_format(
      output,
      "{\"requests\":\"%" PRIu64 "\",\"completed\":\"%" PRIu64
      "\",\"rejected\":\"%" PRIu64 "\",\"cancelled\":\"%" PRIu64
      "\",\"failed\":\"%" PRIu64 "\",\"input_tokens\":\"%" PRIu64
      "\",\"output_tokens\":\"%" PRIu64 "\",\"batches\":\"%" PRIu64 "\"}",
      counters->requests, counters->completed, counters->rejected,
      counters->cancelled, counters->failed, counters->input_tokens,
      counters->output_tokens, counters->batches);
}

static iree_status_t loom_serve_observation_topic(
    void* self, const loom_serve_telemetry_topic_t* topic,
    const uint64_t* values) {
  iree_string_builder_t* output = self;
  // Each extension follows the common topic's array entry.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ","));
  return loom_serve_telemetry_sample_write_json(topic, values, output);
}

static iree_status_t loom_serve_observation_model(
    const loom_serve_registered_model_t* model, iree_host_size_t index,
    iree_string_builder_t* output) {
  const loom_serve_service_t* service = &model->service;
  const loom_serve_service_snapshot_t snapshot =
      loom_serve_registry_snapshot(model);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"slot\":%zu,\"incarnation\":\"%" PRIu64 "\",\"name\":", index,
      model->incarnation));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_string_view_substr(service->name, 0, 256)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"name_truncated\":%s,\"kind\":\"%s\",\"phase\":",
      service->name.size > 256 ? "true" : "false",
      loom_serve_model_kind_name(service->kind)));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_make_cstring_view(snapshot.phase)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"queued\":%zu,\"active\":%zu,\"held\":%zu,\"admission_open\":%s,"
      "\"counters\":",
      snapshot.queued_requests, snapshot.active_requests, model->held_count,
      model->closed ? "false" : "true"));
  IREE_RETURN_IF_ERROR(
      loom_serve_observation_counters(service->telemetry.counters, output));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"topics\":["));
  static const loom_serve_telemetry_metric_t fields[] = {
      {"oldest_queued_age", "nanoseconds", LOOM_SERVE_TELEMETRY_GAUGE},
      {"oldest_active_age", "nanoseconds", LOOM_SERVE_TELEMETRY_GAUGE},
  };
  const loom_serve_telemetry_topic_t topic = {"activity", 1,
                                              IREE_ARRAYSIZE(fields), fields};
  const uint64_t values[] = {(uint64_t)snapshot.oldest_queued_age,
                             (uint64_t)snapshot.oldest_active_age};
  IREE_RETURN_IF_ERROR(
      loom_serve_telemetry_sample_write_json(&topic, values, output));
  if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
      service->telemetry.sample) {
    IREE_RETURN_IF_ERROR(service->telemetry.sample(
        service->self, (loom_serve_telemetry_sample_sink_t){
                           output, loom_serve_observation_topic}));
  }
  return iree_string_builder_append_cstring(output, "]}");
}

iree_status_t loom_serve_observation_write_page(loom_serve_registry_t* registry,
                                                iree_host_size_t* cursor,
                                                iree_string_builder_t* output) {
  if (!*cursor) {
    loom_serve_telemetry_counters_t counters;
    const bool available = loom_serve_registry_counters(registry, &counters);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "{\"type\":\"server\",\"model_capacity\":%zu,\"counters\":",
        loom_serve_registry_capacity(registry)));
    IREE_RETURN_IF_ERROR(
        loom_serve_observation_counters(available ? &counters : NULL, output));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
    *cursor = 1;
    return iree_ok_status();
  }
  const iree_host_size_t first = *cursor - 1;
  const iree_host_size_t capacity = loom_serve_registry_capacity(registry);
  const iree_host_size_t end = first + iree_min(capacity - first, 64);
  iree_host_size_t next = first;
  const loom_serve_registered_model_t* model = NULL;
  while (next < end && !model) {
    const loom_serve_registered_model_t* candidate =
        loom_serve_registry_at(registry, next++);
    if (candidate->service.name.size) {
      model = candidate;
    }
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "{\"type\":\"models\",\"first\":%zu,\"next\":%zu,\"done\":%s,\"model\":",
      first, next, next == capacity ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      model ? loom_serve_observation_model(model, next - 1, output)
            : iree_string_builder_append_cstring(output, "null"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  *cursor = next == capacity ? 0 : next + 1;
  return iree_ok_status();
}
