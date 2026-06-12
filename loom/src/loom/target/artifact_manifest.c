// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_manifest.h"

#include <inttypes.h>
#include <stddef.h>

#include "loom/util/json.h"

void loom_target_artifact_manifest_format_options_initialize(
    loom_target_artifact_manifest_format_options_t* out_options) {
  *out_options = (loom_target_artifact_manifest_format_options_t){
      .mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE,
  };
}

iree_string_view_t loom_target_artifact_manifest_mode_name(
    loom_target_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY:
      return IREE_SV("summary");
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS:
      return IREE_SV("details");
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return IREE_SV("analysis");
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_target_artifact_manifest_mode_parse(
    iree_string_view_t value, loom_target_artifact_manifest_mode_t* out_mode) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("details"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("analysis"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported artifact manifest mode '%.*s'; "
                          "expected 'none', 'summary', 'details', or "
                          "'analysis'",
                          (int)value.size, value.data);
}

static iree_string_view_t loom_target_artifact_manifest_parameter_kind_name(
    loom_target_artifact_manifest_parameter_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_VALUE:
      return IREE_SV("value");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_POINTER:
      return IREE_SV("pointer");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_BINDING:
      return IREE_SV("binding");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_CONSTANT:
      return IREE_SV("constant");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static bool loom_target_artifact_manifest_mode_includes_details(
    loom_target_artifact_manifest_mode_t mode) {
  return mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS ||
         mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS;
}

static bool loom_target_artifact_manifest_mode_is_valid(
    loom_target_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_target_artifact_manifest_require_string(
    iree_string_view_t value, const char* field_name) {
  if (!iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "artifact manifest field '%s' must not be empty",
                          field_name);
}

static iree_status_t loom_target_artifact_manifest_require_array(
    const void* values, iree_host_size_t count, const char* field_name) {
  if (count == 0 || values != NULL) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "artifact manifest array '%s' has a count but no "
                          "storage",
                          field_name);
}

static iree_status_t loom_target_artifact_manifest_json_begin_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name) {
  if (!*inout_first_field) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  }
  *inout_first_field = false;
  return loom_output_stream_write_format(stream, "\"%s\":", name);
}

static iree_status_t loom_target_artifact_manifest_json_write_string_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, name));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t
loom_target_artifact_manifest_json_write_optional_string_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return loom_target_artifact_manifest_json_write_string_field(
      stream, inout_first_field, name, value);
}

static iree_status_t
loom_target_artifact_manifest_json_write_optional_source_name(
    loom_output_stream_t* stream, bool* inout_first_field,
    iree_string_view_t name, iree_string_view_t source_name) {
  if (iree_string_view_is_empty(source_name) ||
      iree_string_view_equal(name, source_name)) {
    return iree_ok_status();
  }
  return loom_target_artifact_manifest_json_write_string_field(
      stream, inout_first_field, "source", source_name);
}

static iree_status_t loom_target_artifact_manifest_json_write_u32_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    uint32_t value) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

static iree_status_t loom_target_artifact_manifest_json_write_u64_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    uint64_t value) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIu64, value);
}

static iree_status_t loom_target_artifact_manifest_json_write_workgroup_size(
    loom_output_stream_t* stream, bool* inout_first_field,
    const uint32_t workgroup_size[3]) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, "workgroup_size"));
  return loom_output_stream_write_format(
      stream, "[%" PRIu32 ",%" PRIu32 ",%" PRIu32 "]", workgroup_size[0],
      workgroup_size[1], workgroup_size[2]);
}

static iree_status_t loom_target_artifact_manifest_json_write_name_array_field(
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    const iree_string_view_t* values, iree_host_size_t count) {
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_require_array(values, count, name));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_artifact_manifest_require_string(values[i], name));
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, values[i]));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static bool loom_target_artifact_manifest_has_target_name(
    const loom_target_artifact_manifest_t* manifest, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < manifest->target_count; ++i) {
    if (iree_string_view_equal(manifest->targets[i].name, name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_target_artifact_manifest_validate_target_names(
    const loom_target_artifact_manifest_t* manifest,
    const iree_string_view_t* values, iree_host_size_t count,
    const char* field_name) {
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_require_array(values, count, field_name));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_artifact_manifest_require_string(values[i], field_name));
    if (!loom_target_artifact_manifest_has_target_name(manifest, values[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "artifact manifest target reference '%.*s' in '%s' must match a "
          "declared targets[].name",
          (int)values[i].size, values[i].data, field_name);
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_artifact_manifest_json_write_target_array_field(
    const loom_target_artifact_manifest_t* manifest,
    loom_output_stream_t* stream, bool* inout_first_field, const char* name,
    const iree_string_view_t* values, iree_host_size_t count) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_validate_target_names(
      manifest, values, count, name));
  return loom_target_artifact_manifest_json_write_name_array_field(
      stream, inout_first_field, name, values, count);
}

static bool loom_target_artifact_manifest_interface_has_data(
    const loom_target_artifact_manifest_interface_t* interface,
    loom_target_artifact_manifest_mode_t mode) {
  return interface->flags != 0 ||
         (loom_target_artifact_manifest_mode_includes_details(mode) &&
          interface->parameter_detail_count != 0);
}

static bool loom_target_artifact_manifest_execution_has_data(
    const loom_target_artifact_manifest_execution_t* execution) {
  return execution->flags != 0;
}

static iree_status_t loom_target_artifact_manifest_format_artifact_json(
    const loom_target_artifact_manifest_artifact_t* artifact,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      artifact->format, "artifact.format"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_string_field(
      stream, &first_field, "format", artifact->format));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "name", artifact->name));
  if (iree_any_bit_set(
          artifact->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_ARTIFACT_FLAG_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u64_field(
        stream, &first_field, "byte_length", artifact->byte_length));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_artifact_manifest_format_target_json(
    const loom_target_artifact_manifest_target_t* target,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      target->name, "target.name"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_string_field(
      stream, &first_field, "name", target->name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "family", target->family));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "processor", target->processor));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "triple", target->triple));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "profile", target->profile));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "code_object_target",
          target->code_object_target));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_name_array_field(
          stream, &first_field, "features", target->feature_names,
          target->feature_name_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_artifact_manifest_format_parameter_json(
    const loom_target_artifact_manifest_parameter_t* parameter,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "name", parameter->name));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_string_field(
      stream, &first_field, "kind",
      loom_target_artifact_manifest_parameter_kind_name(parameter->kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "type", parameter->type));
  if (iree_any_bit_set(parameter->flags,
                       LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u32_field(
        stream, &first_field, "index", parameter->index));
  }
  if (iree_any_bit_set(
          parameter->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_OFFSET)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u64_field(
        stream, &first_field, "byte_offset", parameter->byte_offset));
  }
  if (iree_any_bit_set(
          parameter->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u64_field(
        stream, &first_field, "byte_length", parameter->byte_length));
  }
  if (iree_any_bit_set(
          parameter->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_ALIGNMENT)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u64_field(
        stream, &first_field, "byte_alignment", parameter->byte_alignment));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_artifact_manifest_format_parameters_json(
    const loom_target_artifact_manifest_interface_t* interface,
    loom_output_stream_t* stream, bool* inout_first_field) {
  if (interface->parameter_detail_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      interface->parameters, interface->parameter_detail_count,
      "interface.parameters"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, "parameters"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < interface->parameter_detail_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_parameter_json(
        &interface->parameters[i], stream));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_target_artifact_manifest_format_interface_json(
    const loom_target_artifact_manifest_interface_t* interface,
    loom_target_artifact_manifest_mode_t mode, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  if (iree_any_bit_set(
          interface->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_PARAMETER_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u32_field(
        stream, &first_field, "parameter_count", interface->parameter_count));
  }
  if (iree_any_bit_set(
          interface->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_BINDING_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u32_field(
        stream, &first_field, "binding_count", interface->binding_count));
  }
  if (iree_any_bit_set(
          interface->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_CONSTANT_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u64_field(
        stream, &first_field, "constant_byte_length",
        interface->constant_byte_length));
  }
  if (loom_target_artifact_manifest_mode_includes_details(mode)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_parameters_json(
        interface, stream, &first_field));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_artifact_manifest_format_execution_json(
    const loom_target_artifact_manifest_execution_t* execution,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  if (iree_any_bit_set(
          execution->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(
        loom_target_artifact_manifest_json_write_workgroup_size(
            stream, &first_field, execution->workgroup_size));
  }
  if (iree_any_bit_set(
          execution->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_SUBGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u32_field(
        stream, &first_field, "subgroup_size", execution->subgroup_size));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_artifact_manifest_format_function_json(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_function_t* function,
    loom_target_artifact_manifest_mode_t mode, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      function->name, "function.name"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_string_field(
      stream, &first_field, "name", function->name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_source_name(
          stream, &first_field, function->name, function->source_name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_target_array_field(
          manifest, stream, &first_field, "targets", function->target_names,
          function->target_name_count));
  if (loom_target_artifact_manifest_interface_has_data(&function->interface,
                                                       mode)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
        stream, &first_field, "interface"));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_interface_json(
        &function->interface, mode, stream));
  }
  if (loom_target_artifact_manifest_execution_has_data(&function->execution)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
        stream, &first_field, "execution"));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_execution_json(
        &function->execution, stream));
  }
  if (loom_target_artifact_manifest_mode_includes_details(mode)) {
    IREE_RETURN_IF_ERROR(
        loom_target_artifact_manifest_json_write_name_array_field(
            stream, &first_field, "uses_globals", function->used_global_names,
            function->used_global_name_count));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_artifact_manifest_format_global_json(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_global_t* global,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      global->name, "global.name"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_string_field(
      stream, &first_field, "name", global->name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_source_name(
          stream, &first_field, global->name, global->source_name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          stream, &first_field, "type", global->type));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_target_array_field(
          manifest, stream, &first_field, "targets", global->target_names,
          global->target_name_count));
  if (iree_any_bit_set(global->flags,
                       LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u64_field(
        stream, &first_field, "byte_length", global->byte_length));
  }
  if (iree_any_bit_set(
          global->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_ALIGNMENT)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u64_field(
        stream, &first_field, "byte_alignment", global->byte_alignment));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_target_artifact_manifest_format_targets_json(
    const loom_target_artifact_manifest_t* manifest,
    loom_output_stream_t* stream, bool* inout_first_field) {
  if (manifest->target_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      manifest->targets, manifest->target_count, "targets"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, "targets"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < manifest->target_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_target_json(
        &manifest->targets[i], stream));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_target_artifact_manifest_format_functions_json(
    const loom_target_artifact_manifest_t* manifest,
    loom_target_artifact_manifest_mode_t mode, loom_output_stream_t* stream,
    bool* inout_first_field) {
  if (manifest->function_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      manifest->functions, manifest->function_count, "functions"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, "functions"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < manifest->function_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_function_json(
        manifest, &manifest->functions[i], mode, stream));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_target_artifact_manifest_format_globals_json(
    const loom_target_artifact_manifest_t* manifest,
    loom_output_stream_t* stream, bool* inout_first_field) {
  if (manifest->global_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      manifest->globals, manifest->global_count, "globals"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, inout_first_field, "globals"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < manifest->global_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_global_json(
        manifest, &manifest->globals[i], stream));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

iree_status_t loom_target_artifact_manifest_format_json(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_format_options_t* options,
    loom_output_stream_t* stream) {
  if (options == NULL || stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest format options and stream are "
                            "required");
  }
  if (options->mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_ok_status();
  }
  if (manifest == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest is required when formatting is "
                            "enabled");
  }
  if (!loom_target_artifact_manifest_mode_is_valid(options->mode)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported artifact manifest mode value %d",
                            (int)options->mode);
  }

  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_string_field(
      stream, &first_field, "kind", IREE_SV("loom.artifact_manifest")));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_u32_field(
      stream, &first_field, "schema_version",
      LOOM_TARGET_ARTIFACT_MANIFEST_SCHEMA_VERSION));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_string_field(
      stream, &first_field, "mode",
      loom_target_artifact_manifest_mode_name(options->mode)));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_begin_field(
      stream, &first_field, "artifact"));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_artifact_json(
      &manifest->artifact, stream));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_targets_json(
      manifest, stream, &first_field));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_functions_json(
      manifest, options->mode, stream, &first_field));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_globals_json(
      manifest, stream, &first_field));
  return loom_output_stream_write_cstring(stream, "}");
}
