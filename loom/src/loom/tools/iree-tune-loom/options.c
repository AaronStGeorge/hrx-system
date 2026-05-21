// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-tune-loom/options.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>

#include "loom/util/json.h"

iree_string_view_t iree_tune_loom_artifact_bundle_policy_name(
    iree_tune_loom_artifact_bundle_policy_t policy) {
  switch (policy) {
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_NONE:
      return IREE_SV("none");
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL:
      return IREE_SV("minimal");
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG:
      return IREE_SV("debug");
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_FULL:
      return IREE_SV("full");
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t iree_tune_loom_parse_artifact_bundle_policy(
    iree_string_view_t value,
    iree_tune_loom_artifact_bundle_policy_t* out_policy) {
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("minimal"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("debug"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("full"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_FULL;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--artifact_bundle_policy must be one of minimal, debug, full, or none; "
      "got '%.*s'",
      (int)value.size, value.data);
}

iree_status_t iree_tune_loom_parse_sample_compilation_mode(
    iree_string_view_t value,
    iree_tune_loom_sample_compilation_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(value, IREE_SV("once"))) {
    *out_mode = IREE_TUNE_LOOM_SAMPLE_COMPILATION_ONCE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("per_sample"))) {
    *out_mode = IREE_TUNE_LOOM_SAMPLE_COMPILATION_PER_SAMPLE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("both"))) {
    *out_mode = IREE_TUNE_LOOM_SAMPLE_COMPILATION_BOTH;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--sample_compilation must be one of once, per_sample, or both; got "
      "'%.*s'",
      (int)value.size, value.data);
}

iree_string_view_t iree_tune_loom_sample_compilation_mode_name(
    iree_tune_loom_sample_compilation_mode_t mode) {
  switch (mode) {
    case IREE_TUNE_LOOM_SAMPLE_COMPILATION_ONCE:
      return IREE_SV("once");
    case IREE_TUNE_LOOM_SAMPLE_COMPILATION_PER_SAMPLE:
      return IREE_SV("per_sample");
    case IREE_TUNE_LOOM_SAMPLE_COMPILATION_BOTH:
      return IREE_SV("both");
    default:
      return IREE_SV("unknown");
  }
}

bool iree_tune_loom_sample_compilation_runs_once(
    iree_tune_loom_sample_compilation_mode_t mode) {
  return mode == IREE_TUNE_LOOM_SAMPLE_COMPILATION_ONCE ||
         mode == IREE_TUNE_LOOM_SAMPLE_COMPILATION_BOTH;
}

bool iree_tune_loom_sample_compilation_runs_per_sample(
    iree_tune_loom_sample_compilation_mode_t mode) {
  return mode == IREE_TUNE_LOOM_SAMPLE_COMPILATION_PER_SAMPLE ||
         mode == IREE_TUNE_LOOM_SAMPLE_COMPILATION_BOTH;
}

iree_status_t iree_tune_loom_parse_interleave_mode(
    iree_string_view_t value, iree_tune_loom_interleave_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("ABABA")) ||
      iree_string_view_equal(value, IREE_SV("ababa"))) {
    *out_mode = IREE_TUNE_LOOM_INTERLEAVE_ABABA;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("round_robin")) ||
      iree_string_view_equal(value, IREE_SV("round-robin"))) {
    *out_mode = IREE_TUNE_LOOM_INTERLEAVE_ROUND_ROBIN;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--interleave must be one of ABABA or round_robin; got '%.*s'",
      (int)value.size, value.data);
}

iree_string_view_t iree_tune_loom_interleave_mode_name(
    iree_tune_loom_interleave_mode_t mode) {
  switch (mode) {
    case IREE_TUNE_LOOM_INTERLEAVE_NONE:
      return IREE_SV("none");
    case IREE_TUNE_LOOM_INTERLEAVE_ABABA:
      return IREE_SV("ABABA");
    case IREE_TUNE_LOOM_INTERLEAVE_ROUND_ROBIN:
      return IREE_SV("round_robin");
    default:
      return IREE_SV("unknown");
  }
}

typedef struct iree_tune_loom_profile_family_name_t {
  // HAL profiling data-family bit represented by these names.
  iree_hal_device_profiling_data_families_t bit;
  // Command-line family name accepted by --profile_data.
  const char* flag_name;
  // Stable JSON string used for this family.
  const char* json_name;
} iree_tune_loom_profile_family_name_t;

static const iree_tune_loom_profile_family_name_t
    iree_tune_loom_profile_family_names[] = {
        {IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS, "queue-events",
         "queue_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS, "host-execution",
         "host_execution_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
         "device-queue-events", "device_queue_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS, "dispatch-events",
         "dispatch_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES, "counters",
         "counter_samples"},
        {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
         "executable-metadata", "executable_metadata"},
        {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES, "executable-traces",
         "executable_traces"},
        {IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS, "memory-events",
         "memory_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS, "device-metrics",
         "device_metrics"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS,
         "command-region-events", "command_region_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES, "counter-ranges",
         "counter_ranges"},
};

bool iree_tune_loom_profile_data_has_counter_data(
    iree_hal_device_profiling_data_families_t profile_data_families) {
  return iree_any_bit_set(profile_data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                              IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES);
}

bool iree_tune_loom_profile_data_needs_artifact_data(
    iree_hal_device_profiling_data_families_t profile_data_families) {
  return iree_tune_loom_profile_data_has_counter_data(profile_data_families) ||
         iree_any_bit_set(profile_data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS |
                              IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES);
}

iree_status_t iree_tune_loom_parse_profile_data_families(
    iree_string_view_t value,
    iree_hal_device_profiling_data_families_t* out_profile_data_families) {
  *out_profile_data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  iree_string_view_t remaining = iree_string_view_trim(value);
  if (iree_string_view_is_empty(remaining)) {
    return iree_ok_status();
  }
  *out_profile_data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t family_part = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &family_part, &remaining);
    family_part = iree_string_view_trim(family_part);
    if (iree_string_view_is_empty(family_part)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--profile_data contains an empty data family");
    }
    if (iree_string_view_equal(family_part, IREE_SV("none"))) {
      if (*out_profile_data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE ||
          !iree_string_view_is_empty(iree_string_view_trim(remaining))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--profile_data=none cannot be combined with "
                                "other data families");
      }
      return iree_ok_status();
    }
    bool matched = false;
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_tune_loom_profile_family_names); ++i) {
      const iree_tune_loom_profile_family_name_t* family =
          &iree_tune_loom_profile_family_names[i];
      if (iree_string_view_equal(family_part,
                                 iree_make_cstring_view(family->flag_name)) ||
          iree_string_view_equal(family_part,
                                 iree_make_cstring_view(family->json_name))) {
        *out_profile_data_families |= family->bit;
        matched = true;
        break;
      }
    }
    if (!matched &&
        iree_string_view_equal(family_part, IREE_SV("pmc-ranges"))) {
      *out_profile_data_families |=
          IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
      matched = true;
    }
    if (!matched) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported --profile_data family '%.*s'",
                              (int)family_part.size, family_part.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

iree_status_t iree_tune_loom_write_profile_family_names_json(
    iree_hal_device_profiling_data_families_t profile_data_families,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool first = true;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_tune_loom_profile_family_names); ++i) {
    const iree_tune_loom_profile_family_name_t* family =
        &iree_tune_loom_profile_family_names[i];
    if (!iree_all_bits_set(profile_data_families, family->bit)) {
      continue;
    }
    if (!first) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_cstring(stream, family->json_name));
    first = false;
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

iree_status_t iree_tune_loom_parse_i32_flag(iree_string_view_t flag_name,
                                            void* storage,
                                            iree_string_view_t value) {
  iree_tune_loom_i32_flag_t* flag = (iree_tune_loom_i32_flag_t*)storage;
  char* value_end = NULL;
  int64_t parsed_value = 0;
  if (!iree_string_view_is_empty(value)) {
    errno = 0;
    parsed_value = strtoll(value.data, &value_end, 10);
    if (errno == ERANGE) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "flag '--%.*s' value '%.*s' is outside int64 "
                              "range",
                              (int)flag_name.size, flag_name.data,
                              (int)value.size, value.data);
    }
    if (value_end != value.data + value.size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "flag '--%.*s' must be an int32 value; got '%.*s'",
          (int)flag_name.size, flag_name.data, (int)value.size, value.data);
    }
  }
  if (parsed_value < INT32_MIN || parsed_value > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "flag '--%.*s' value %" PRIi64
                            " is outside int32 range",
                            (int)flag_name.size, flag_name.data, parsed_value);
  }
  flag->value = (int32_t)parsed_value;
  flag->specified = true;
  return iree_ok_status();
}

void iree_tune_loom_print_i32_flag(iree_string_view_t flag_name, void* storage,
                                   FILE* file) {
  const iree_tune_loom_i32_flag_t* flag =
      (const iree_tune_loom_i32_flag_t*)storage;
  fprintf(file, "--%.*s=%" PRId32 "\n", (int)flag_name.size, flag_name.data,
          flag->value);
}

iree_status_t iree_tune_loom_parse_bool_flag(iree_string_view_t flag_name,
                                             void* storage,
                                             iree_string_view_t value) {
  iree_tune_loom_bool_flag_t* flag = (iree_tune_loom_bool_flag_t*)storage;
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("true")) ||
      iree_string_view_equal(value, IREE_SV("1"))) {
    flag->value = true;
  } else if (iree_string_view_equal(value, IREE_SV("false")) ||
             iree_string_view_equal(value, IREE_SV("0"))) {
    flag->value = false;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flag '--%.*s' must be a bool value; got '%.*s'",
                            (int)flag_name.size, flag_name.data,
                            (int)value.size, value.data);
  }
  flag->specified = true;
  return iree_ok_status();
}

void iree_tune_loom_print_bool_flag(iree_string_view_t flag_name, void* storage,
                                    FILE* file) {
  const iree_tune_loom_bool_flag_t* flag =
      (const iree_tune_loom_bool_flag_t*)storage;
  fprintf(file, "--%.*s=%s\n", (int)flag_name.size, flag_name.data,
          flag->value ? "true" : "false");
}
