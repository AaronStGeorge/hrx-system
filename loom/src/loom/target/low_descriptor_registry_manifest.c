// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry_manifest.h"

#include <inttypes.h>

#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_status_t loom_target_low_manifest_write_string_field(
    loom_output_stream_t* stream, const char* field_name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\":"));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_target_low_manifest_write_memory_spaces(
    loom_output_stream_t* stream,
    const loom_target_memory_space_map_t* memory_spaces) {
  return loom_output_stream_write_format(
      stream,
      "\"memory_spaces\":{\"generic\":%" PRIu32 ",\"global\":%" PRIu32
      ",\"workgroup\":%" PRIu32 ",\"constant\":%" PRIu32
      ",\"private_memory\":%" PRIu32 ",\"host\":%" PRIu32
      ",\"descriptor\":%" PRIu32 "}",
      memory_spaces->generic, memory_spaces->global, memory_spaces->workgroup,
      memory_spaces->constant, memory_spaces->private_memory,
      memory_spaces->host, memory_spaces->descriptor);
}

static iree_status_t loom_target_low_manifest_write_descriptor_set_summary(
    loom_output_stream_t* stream,
    const loom_low_descriptor_set_t* descriptor_set) {
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &key));
  iree_string_view_t target = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->target_key_string_offset, &target));
  iree_string_view_t feature_namespace = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->feature_key_string_offset,
      &feature_namespace));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  IREE_RETURN_IF_ERROR(
      loom_target_low_manifest_write_string_field(stream, "key", key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(
      loom_target_low_manifest_write_string_field(stream, "target", target));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "feature_namespace", feature_namespace));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"abi_version\":%" PRIu32 ",\"generator_version\":%" PRIu32
      ",\"table_counts\":{\"descriptors\":%" PRIu32
      ",\"descriptor_refs\":%" PRIu32 ",\"operands\":%" PRIu32
      ",\"immediates\":%" PRIu32 ",\"enum_domains\":%" PRIu32
      ",\"enum_values\":%" PRIu32 ",\"effects\":%" PRIu32
      ",\"constraints\":%" PRIu32 ",\"reg_classes\":%" PRIu32
      ",\"reg_class_alts\":%" PRIu32 ",\"schedule_classes\":%" PRIu32
      ",\"issue_uses\":%" PRIu32 ",\"resources\":%" PRIu32
      ",\"hazards\":%" PRIu32 ",\"pressure_deltas\":%" PRIu32
      ",\"feature_mask_words\":%" PRIu32 "}}",
      descriptor_set->abi_version, descriptor_set->generator_version,
      descriptor_set->descriptor_count, descriptor_set->descriptor_ref_count,
      descriptor_set->operand_count, descriptor_set->immediate_count,
      descriptor_set->enum_domain_count, descriptor_set->enum_value_count,
      descriptor_set->effect_count, descriptor_set->constraint_count,
      descriptor_set->reg_class_count, descriptor_set->reg_class_alt_count,
      descriptor_set->schedule_class_count, descriptor_set->issue_use_count,
      descriptor_set->resource_count, descriptor_set->hazard_count,
      descriptor_set->pressure_delta_count,
      descriptor_set->feature_mask_word_count));
  return iree_ok_status();
}

static iree_status_t loom_target_low_manifest_write_bundle_summary(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_target_bundle_t* bundle,
    loom_low_descriptor_requirement_flags_t requirements,
    loom_output_stream_t* stream) {
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      descriptor_registry, bundle, requirements, &descriptor_set));
  iree_string_view_t descriptor_set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &descriptor_set_key));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  IREE_RETURN_IF_ERROR(
      loom_target_low_manifest_write_string_field(stream, "key", bundle->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"snapshot\":{"));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "name", bundle->snapshot->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"codegen_format\":%u,",
      (unsigned)bundle->snapshot->codegen_format));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "target_triple", bundle->snapshot->target_triple));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "data_layout", bundle->snapshot->data_layout));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"artifact_format\":%u,",
      (unsigned)bundle->snapshot->artifact_format));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "target_cpu", bundle->snapshot->target_cpu));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "target_features", bundle->snapshot->target_features));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"default_pointer_bitwidth\":%" PRIu32 ",\"index_bitwidth\":%" PRIu32
      ",\"offset_bitwidth\":%" PRIu32 ",\"subgroup_size\":%" PRIu32 ",",
      bundle->snapshot->default_pointer_bitwidth,
      bundle->snapshot->index_bitwidth, bundle->snapshot->offset_bitwidth,
      bundle->snapshot->subgroup_size));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_memory_spaces(
      stream, &bundle->snapshot->memory_spaces));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "},\"export\":{"));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "name", bundle->export_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "export_symbol", bundle->export_plan->export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, ",\"abi_kind\":%u,\"linkage\":%u",
                                      (unsigned)bundle->export_plan->abi_kind,
                                      (unsigned)bundle->export_plan->linkage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"hal_kernel\":{\"binding_alignment\":%" PRIu32
      ",\"required_workgroup_size\":{\"x\":%" PRIu32 ",\"y\":%" PRIu32
      ",\"z\":%" PRIu32 "},\"flat_workgroup_size_min\":%" PRIu32
      ",\"flat_workgroup_size_max\":%" PRIu32
      ",\"buffer_resource_flags\":%" PRIu32 "}",
      bundle->export_plan->hal_kernel.binding_alignment,
      bundle->export_plan->hal_kernel.required_workgroup_size.x,
      bundle->export_plan->hal_kernel.required_workgroup_size.y,
      bundle->export_plan->hal_kernel.required_workgroup_size.z,
      bundle->export_plan->hal_kernel.flat_workgroup_size_min,
      bundle->export_plan->hal_kernel.flat_workgroup_size_max,
      bundle->export_plan->hal_kernel.buffer_resource_flags));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "},\"config\":{"));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "name", bundle->config->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
  IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_string_field(
      stream, "descriptor_set", descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"contract_feature_bits\":%" PRIu64 "}}",
      bundle->config->contract_feature_bits));
  return iree_ok_status();
}

iree_status_t loom_target_low_descriptor_registry_format_manifest_json(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements,
    iree_string_builder_t* builder) {
  if (builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry manifest builder is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_target_low_descriptor_registry_verify(registry, requirements));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(&registry->registry);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      "{\"descriptor_set_count\":%" PRIhsz ",\"bundle_count\":%" PRIhsz
      ",\"descriptor_sets\":[",
      descriptor_set_count, registry->target_bundle_count));
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(&registry->registry, i);
    IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_descriptor_set_summary(
        &stream, descriptor_set));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "],\"bundles\":["));
  for (iree_host_size_t i = 0; i < registry->target_bundle_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_target_low_manifest_write_bundle_summary(
        &registry->registry, registry->target_bundles[i], requirements,
        &stream));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
