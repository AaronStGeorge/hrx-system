// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry.h"

void loom_target_low_descriptor_registry_initialize_from_tables(
    loom_target_low_descriptor_registry_t* out_registry,
    const loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_count,
    const loom_target_bundle_t* const* target_bundles,
    iree_host_size_t target_bundle_count) {
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
  if (*descriptor_set_provider_count > descriptor_set_provider_capacity ||
      *target_bundle_count > target_bundle_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target-low registry destination counts exceed destination capacity");
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

iree_status_t loom_target_low_descriptor_registry_lookup_bundle(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_view_t key, const loom_target_bundle_t** out_bundle) {
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(registry);
  return loom_target_preset_registry_lookup_bundle(&preset_registry, key,
                                                   out_bundle);
}

iree_status_t loom_target_low_descriptor_set_select_for_bundle(
    const loom_low_descriptor_registry_t* registry,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  iree_string_view_t descriptor_set_key =
      iree_string_view_trim(bundle->config->contract_set_key);

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(registry, descriptor_set_key);
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target bundle '%.*s' selected low descriptor set '%.*s' that is not "
        "linked",
        (int)bundle->name.size, bundle->name.data, (int)descriptor_set_key.size,
        descriptor_set_key.data);
  }
  *out_descriptor_set = descriptor_set;
  return iree_ok_status();
}
