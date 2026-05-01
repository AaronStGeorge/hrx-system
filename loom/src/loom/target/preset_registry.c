// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/preset_registry.h"

iree_status_t loom_target_preset_registry_lookup_bundle(
    const loom_target_preset_registry_t* registry, iree_string_view_t key,
    const loom_target_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  key = iree_string_view_trim(key);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target preset key is required");
  }
  for (iree_host_size_t i = 0; i < registry->target_bundle_count; ++i) {
    const loom_target_bundle_t* bundle = registry->target_bundles[i];
    if (iree_string_view_equal(bundle->name, key)) {
      *out_bundle = bundle;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "target preset '%.*s' is not linked", (int)key.size,
                          key.data);
}
