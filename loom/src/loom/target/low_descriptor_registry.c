// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/util/json.h"
#include "loom/util/stream.h"

void loom_target_low_descriptor_registry_initialize_from_tables(
    loom_target_low_descriptor_registry_t* out_registry,
    const loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_count,
    const loom_target_bundle_t* const* target_bundles,
    iree_host_size_t target_bundle_count) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_target_low_descriptor_registry_t){
      .descriptor_set_providers = descriptor_set_providers,
      .descriptor_set_provider_count = descriptor_set_provider_count,
      .target_bundles = target_bundles,
      .target_bundle_count = target_bundle_count,
      .registry =
          {
              .descriptor_set_providers = descriptor_set_providers,
              .descriptor_set_provider_count = descriptor_set_provider_count,
              .target_bundles = target_bundles,
              .target_bundle_count = target_bundle_count,
          },
  };
}

iree_status_t loom_target_low_descriptor_registry_append_to_tables(
    const loom_target_low_descriptor_registry_t* source,
    loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_capacity,
    iree_host_size_t* descriptor_set_provider_count,
    const loom_target_bundle_t** target_bundles,
    iree_host_size_t target_bundle_capacity,
    iree_host_size_t* target_bundle_count) {
  IREE_ASSERT_ARGUMENT(descriptor_set_provider_count);
  IREE_ASSERT_ARGUMENT(target_bundle_count);
  if (source == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry source is required");
  }
  if (*descriptor_set_provider_count > descriptor_set_provider_capacity ||
      *target_bundle_count > target_bundle_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry destination counts exceed destination capacity");
  }
  if (source->descriptor_set_provider_count != 0 &&
      source->descriptor_set_providers == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry source descriptor-set providers are required");
  }
  if (source->target_bundle_count != 0 && source->target_bundles == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry source bundles are required");
  }
  if (source->descriptor_set_provider_count != 0 &&
      descriptor_set_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry destination "
                            "descriptor-set provider table is required");
  }
  if (source->target_bundle_count != 0 && target_bundles == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry destination bundle table is required");
  }
  if (source->descriptor_set_provider_count >
      descriptor_set_provider_capacity - *descriptor_set_provider_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target-low registry descriptor-set provider table capacity exceeded");
  }
  if (source->target_bundle_count >
      target_bundle_capacity - *target_bundle_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target-low registry target-bundle table capacity exceeded");
  }

  for (iree_host_size_t i = 0; i < source->descriptor_set_provider_count; ++i) {
    descriptor_set_providers[*descriptor_set_provider_count + i] =
        source->descriptor_set_providers[i];
  }
  *descriptor_set_provider_count += source->descriptor_set_provider_count;
  for (iree_host_size_t i = 0; i < source->target_bundle_count; ++i) {
    target_bundles[*target_bundle_count + i] = source->target_bundles[i];
  }
  *target_bundle_count += source->target_bundle_count;
  return iree_ok_status();
}

static iree_status_t loom_target_low_verify_bundle_record(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_target_bundle_t* bundle,
    loom_low_descriptor_requirement_flags_t requirements,
    iree_host_size_t row) {
  if (bundle == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry bundle row %" PRIhsz " is null", row);
  }
  if (iree_string_view_is_empty(iree_string_view_trim(bundle->name))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry bundle row %" PRIhsz " has no name", row);
  }
  if (bundle->snapshot == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low bundle '%.*s' has no target snapshot",
                            (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->export_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low bundle '%.*s' has no export plan",
                            (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low bundle '%.*s' has no config",
                            (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low bundle '%.*s' snapshot has unknown codegen format",
        (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->snapshot->artifact_format ==
      LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low bundle '%.*s' snapshot has unknown artifact format",
        (int)bundle->name.size, bundle->name.data);
  }
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low bundle '%.*s' export plan has unknown ABI",
        (int)bundle->name.size, bundle->name.data);
  }

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  return loom_target_low_descriptor_set_select_for_bundle(
      descriptor_registry, bundle, requirements, &descriptor_set);
}

iree_status_t loom_target_low_descriptor_registry_verify(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry is required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor-set providers are required");
  }
  if (registry->target_bundle_count != 0 && registry->target_bundles == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry bundles are required");
  }
  if (registry->registry.descriptor_set_count != 0 ||
      registry->registry.descriptor_sets != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor view must use provider tables");
  }
  if (registry->registry.descriptor_set_providers !=
          registry->descriptor_set_providers ||
      registry->registry.descriptor_set_provider_count !=
          registry->descriptor_set_provider_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor view does not match provider table");
  }

  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_registry_verify(&registry->registry));

  for (iree_host_size_t i = 0; i < registry->target_bundle_count; ++i) {
    const loom_target_bundle_t* bundle = registry->target_bundles[i];
    IREE_RETURN_IF_ERROR(loom_target_low_verify_bundle_record(
        &registry->registry, bundle, requirements, i));

    for (iree_host_size_t j = i + 1; j < registry->target_bundle_count; ++j) {
      const loom_target_bundle_t* other_bundle = registry->target_bundles[j];
      if (other_bundle == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "target-low registry bundle row %" PRIhsz " is null", j);
      }
      if (iree_string_view_equal(bundle->name, other_bundle->name)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "target-low registry has duplicate bundle key '%.*s'",
            (int)bundle->name.size, bundle->name.data);
      }
    }
  }
  return iree_ok_status();
}

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
      ",\"offset_bitwidth\":%" PRIu32 ",",
      bundle->snapshot->default_pointer_bitwidth,
      bundle->snapshot->index_bitwidth, bundle->snapshot->offset_bitwidth));
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

iree_status_t loom_target_low_descriptor_registry_lookup_bundle(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_view_t key, const loom_target_bundle_t** out_bundle) {
  if (registry == NULL) {
    if (out_bundle != NULL) *out_bundle = NULL;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry is required");
  }
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(registry);
  return loom_target_preset_registry_lookup_bundle(&preset_registry, key,
                                                   out_bundle);
}

iree_status_t loom_target_low_descriptor_set_select_for_bundle(
    const loom_low_descriptor_registry_t* registry,
    const loom_target_bundle_t* bundle,
    loom_low_descriptor_requirement_flags_t requirements,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  if (out_descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set output is required");
  }
  *out_descriptor_set = NULL;
  if (registry == NULL || bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry and target bundle are "
                            "required");
  }
  if (bundle->config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle '%.*s' has no config",
                            (int)bundle->name.size, bundle->name.data);
  }
  iree_string_view_t descriptor_set_key =
      iree_string_view_trim(bundle->config->contract_set_key);
  if (iree_string_view_is_empty(descriptor_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' config '%.*s' does not name a low descriptor set",
        (int)bundle->name.size, bundle->name.data,
        (int)bundle->config->name.size, bundle->config->name.data);
  }

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_registry_lookup(
      registry, descriptor_set_key, &descriptor_set));
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target bundle '%.*s' selected low descriptor set '%.*s' that is not "
        "linked",
        (int)bundle->name.size, bundle->name.data, (int)descriptor_set_key.size,
        descriptor_set_key.data);
  }
  if (requirements != 0) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_verify_requirements(
        descriptor_set, requirements));
  }
  *out_descriptor_set = descriptor_set;
  return iree_ok_status();
}
