// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/testing/low_descriptor_registry_verify.h"

#include <inttypes.h>
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
  if (iree_string_view_is_empty(
          iree_string_view_trim(bundle->config->contract_set_key))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low bundle '%.*s' config '%.*s' has no descriptor set key",
        (int)bundle->name.size, bundle->name.data,
        (int)bundle->config->name.size, bundle->config->name.data);
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
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      descriptor_registry, bundle, &descriptor_set));
  if (requirements != 0) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_verify_requirements(
        descriptor_set, requirements));
  }
  return iree_ok_status();
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
