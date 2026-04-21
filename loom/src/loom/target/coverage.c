// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/coverage.h"

#include <inttypes.h>

#include "loom/util/json.h"
#include "loom/util/stream.h"

typedef struct loom_target_coverage_phase_name_t {
  // Phase bit represented by |name|.
  loom_target_coverage_phase_flags_t bit;
  // Stable manifest token for |bit|.
  iree_string_view_t name;
} loom_target_coverage_phase_name_t;

static const loom_target_coverage_phase_name_t kLoomTargetCoveragePhaseNames[] =
    {
        {LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR, IREE_SVL("descriptor")},
        {LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE, IREE_SVL("low-parse")},
        {LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER, IREE_SVL("source-lower")},
        {LOOM_TARGET_COVERAGE_PHASE_SCHEDULE, IREE_SVL("schedule")},
        {LOOM_TARGET_COVERAGE_PHASE_ALLOCATE, IREE_SVL("allocate")},
        {LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT, IREE_SVL("text-emit")},
        {LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE, IREE_SVL("binary-encode")},
        {LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP, IREE_SVL("artifact-wrap")},
        {LOOM_TARGET_COVERAGE_PHASE_INSPECT, IREE_SVL("inspect")},
        {LOOM_TARGET_COVERAGE_PHASE_RUN, IREE_SVL("run")},
};

static iree_string_view_t loom_target_coverage_trim(iree_string_view_t value) {
  return iree_string_view_trim(value);
}

static iree_status_t loom_target_coverage_verify_string(
    iree_string_view_t value, iree_string_view_t provider_name,
    iree_host_size_t row_index, const char* field_name) {
  if (!iree_string_view_is_empty(loom_target_coverage_trim(value))) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "target coverage provider '%.*s' row %" PRIhsz " has empty %s",
      (int)provider_name.size, provider_name.data, row_index, field_name);
}

static iree_status_t loom_target_coverage_verify_row(
    const loom_target_coverage_provider_t* provider,
    iree_host_size_t row_index) {
  const loom_target_coverage_row_t* row = &provider->rows[row_index];
  IREE_RETURN_IF_ERROR(loom_target_coverage_verify_string(
      row->target_key, provider->name, row_index, "target key"));
  IREE_RETURN_IF_ERROR(loom_target_coverage_verify_string(
      row->descriptor_set_key, provider->name, row_index,
      "descriptor set key"));
  IREE_RETURN_IF_ERROR(loom_target_coverage_verify_string(
      row->category, provider->name, row_index, "category"));
  IREE_RETURN_IF_ERROR(loom_target_coverage_verify_string(
      row->semantic_tag, provider->name, row_index, "semantic tag"));
  if (row->expected_phases == LOOM_TARGET_COVERAGE_PHASE_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage provider '%.*s' row %" PRIhsz
                            " has no expected phases",
                            (int)provider->name.size, provider->name.data,
                            row_index);
  }
  if (iree_any_bit_set(row->expected_phases | row->supported_phases,
                       ~LOOM_TARGET_COVERAGE_PHASE_ALL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage provider '%.*s' row %" PRIhsz
                            " uses unknown phase bits",
                            (int)provider->name.size, provider->name.data,
                            row_index);
  }
  if (iree_any_bit_set(row->supported_phases, ~row->expected_phases)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage provider '%.*s' row %" PRIhsz
                            " supports phases it does not expect",
                            (int)provider->name.size, provider->name.data,
                            row_index);
  }

  const loom_target_coverage_phase_flags_t missing_phases =
      row->expected_phases & ~row->supported_phases;
  const bool has_gap_key =
      !iree_string_view_is_empty(loom_target_coverage_trim(row->gap_key));
  if (missing_phases != LOOM_TARGET_COVERAGE_PHASE_NONE && !has_gap_key) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage provider '%.*s' row %" PRIhsz
                            " is missing phases but has no gap key",
                            (int)provider->name.size, provider->name.data,
                            row_index);
  }
  if (missing_phases == LOOM_TARGET_COVERAGE_PHASE_NONE && has_gap_key) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage provider '%.*s' row %" PRIhsz
                            " is complete but still has a gap key",
                            (int)provider->name.size, provider->name.data,
                            row_index);
  }
  return iree_ok_status();
}

iree_status_t loom_target_coverage_provider_set_verify(
    const loom_target_coverage_provider_set_t* provider_set) {
  if (provider_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage provider set is required");
  }
  if (provider_set->provider_count != 0 && provider_set->providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage provider table is required");
  }
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_coverage_provider_t* provider =
        provider_set->providers[i];
    if (provider == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target coverage provider %" PRIhsz " is null",
                              i);
    }
    if (iree_string_view_is_empty(loom_target_coverage_trim(provider->name))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target coverage provider %" PRIhsz " has no name", i);
    }
    if (provider->row_count != 0 && provider->rows == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target coverage provider '%.*s' has no row "
                              "table",
                              (int)provider->name.size, provider->name.data);
    }
    for (iree_host_size_t row_index = 0; row_index < provider->row_count;
         ++row_index) {
      IREE_RETURN_IF_ERROR(
          loom_target_coverage_verify_row(provider, row_index));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_coverage_write_string_field(
    loom_output_stream_t* stream, const char* field_name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\":"));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_target_coverage_write_phase_array(
    loom_output_stream_t* stream,
    loom_target_coverage_phase_flags_t phase_flags) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  bool needs_comma = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomTargetCoveragePhaseNames); ++i) {
    const loom_target_coverage_phase_name_t* phase =
        &kLoomTargetCoveragePhaseNames[i];
    if (!iree_any_bit_set(phase_flags, phase->bit)) {
      continue;
    }
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, phase->name));
    needs_comma = true;
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_target_coverage_write_optional_gap(
    loom_output_stream_t* stream, iree_string_view_t gap_key) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"gap\":"));
  if (iree_string_view_is_empty(loom_target_coverage_trim(gap_key))) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, gap_key);
}

static iree_status_t loom_target_coverage_write_row(
    loom_output_stream_t* stream, iree_string_view_t provider_name,
    const loom_target_coverage_row_t* row) {
  const loom_target_coverage_phase_flags_t missing_phases =
      row->expected_phases & ~row->supported_phases;

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  IREE_RETURN_IF_ERROR(loom_target_coverage_write_string_field(
      stream, "provider", provider_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_coverage_write_string_field(
      stream, "target", row->target_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_coverage_write_string_field(
      stream, "descriptor_set", row->descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_coverage_write_string_field(
      stream, "category", row->category));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_coverage_write_string_field(
      stream, "semantic", row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"expected\":"));
  IREE_RETURN_IF_ERROR(
      loom_target_coverage_write_phase_array(stream, row->expected_phases));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"supported\":"));
  IREE_RETURN_IF_ERROR(
      loom_target_coverage_write_phase_array(stream, row->supported_phases));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"missing\":"));
  IREE_RETURN_IF_ERROR(
      loom_target_coverage_write_phase_array(stream, missing_phases));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(
      loom_target_coverage_write_optional_gap(stream, row->gap_key));
  return loom_output_stream_write_char(stream, '}');
}

iree_status_t loom_target_coverage_provider_set_format_manifest_json(
    const loom_target_coverage_provider_set_t* provider_set,
    iree_string_builder_t* builder) {
  if (builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target coverage manifest builder is required");
  }
  IREE_RETURN_IF_ERROR(loom_target_coverage_provider_set_verify(provider_set));

  iree_host_size_t row_count = 0;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_coverage_provider_t* provider =
        provider_set->providers[i];
    if (IREE_HOST_SIZE_MAX - row_count < provider->row_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "target coverage row count overflow");
    }
    row_count += provider->row_count;
  }

  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      "{\"provider_count\":%" PRIhsz ",\"row_count\":%" PRIhsz ",\"rows\":[",
      provider_set->provider_count, row_count));
  bool needs_comma = false;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_coverage_provider_t* provider =
        provider_set->providers[i];
    for (iree_host_size_t row_index = 0; row_index < provider->row_count;
         ++row_index) {
      if (needs_comma) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
      }
      IREE_RETURN_IF_ERROR(loom_target_coverage_write_row(
          &stream, provider->name, &provider->rows[row_index]));
      needs_comma = true;
    }
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
