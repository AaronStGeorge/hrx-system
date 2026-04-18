// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/preset_registry.h"

#include <inttypes.h>

iree_status_t loom_target_preset_registry_lookup_bundle(
    const loom_target_preset_registry_t* registry, iree_string_view_t key,
    const loom_target_bundle_t** out_bundle) {
  if (out_bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle output is required");
  }
  *out_bundle = NULL;
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target preset registry is required");
  }
  key = iree_string_view_trim(key);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target preset key is required");
  }
  if (registry->target_bundle_count > 0 && registry->target_bundles == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target preset registry has no bundle table");
  }
  for (iree_host_size_t i = 0; i < registry->target_bundle_count; ++i) {
    const loom_target_bundle_t* bundle = registry->target_bundles[i];
    if (bundle == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target preset registry bundle row %" PRIhsz " is null", i);
    }
    if (iree_string_view_equal(bundle->name, key)) {
      *out_bundle = bundle;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "target preset '%.*s' is not linked", (int)key.size,
                          key.data);
}
