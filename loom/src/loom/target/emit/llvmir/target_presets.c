// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_presets.h"

#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#include "loom/target/emit/llvmir/x86/target_env.h"

enum { LOOM_LLVMIR_BUILTIN_PROFILE_COUNT = 3 };

static const loom_llvmir_target_profile_t* loom_llvmir_builtin_profile(
    iree_host_size_t ordinal) {
  switch (ordinal) {
    case 0:
      return loom_llvmir_target_profile_x86_64_object();
    case 1:
      return loom_llvmir_target_profile_x86_64_packed_dot_object();
    case 2:
      return loom_llvmir_target_profile_amdgpu_hal();
    default:
      return NULL;
  }
}

iree_status_t loom_llvmir_target_profile_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  if (out_profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile output is required");
  }
  *out_profile = NULL;
  profile_name = iree_string_view_trim(profile_name);
  if (iree_string_view_is_empty(profile_name)) {
    *out_profile = loom_llvmir_target_profile_x86_64_object();
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < LOOM_LLVMIR_BUILTIN_PROFILE_COUNT; ++i) {
    const loom_llvmir_target_profile_t* profile =
        loom_llvmir_builtin_profile(i);
    if (iree_string_view_equal(profile_name, profile->name)) {
      *out_profile = profile;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown LLVMIR target profile '%.*s'",
                          (int)profile_name.size, profile_name.data);
}
