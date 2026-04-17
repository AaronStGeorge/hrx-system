// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_env.h"

#include <stdio.h>

iree_status_t loom_llvmir_target_env_module_config(
    const loom_llvmir_target_env_t* target_env, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  if (out_config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target config output is required");
  }
  *out_config = (loom_llvmir_target_config_t){0};
  if (target_env == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target environment is required");
  }
  *out_config = (loom_llvmir_target_config_t){
      .source_name = source_name,
      .target_triple = target_env->target_triple,
      .data_layout = target_env->data_layout,
      .default_pointer_bitwidth = target_env->default_pointer_bitwidth,
      .index_bitwidth = target_env->index_bitwidth,
      .offset_bitwidth = target_env->offset_bitwidth,
  };
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_profile_module_config(
    const loom_llvmir_target_profile_t* profile, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  if (profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is required");
  }
  return loom_llvmir_target_env_module_config(profile->target_env, source_name,
                                              out_config);
}

static iree_status_t loom_llvmir_target_profile_append_llc_argument(
    iree_string_view_t prefix, iree_string_view_t value,
    loom_llvmir_target_profile_llc_arguments_t* arguments) {
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  if (arguments->count >= IREE_ARRAYSIZE(arguments->values)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM llc argument storage is too small");
  }
  char* storage = arguments->storage[arguments->count];
  iree_host_size_t storage_length =
      IREE_ARRAYSIZE(arguments->storage[arguments->count]);
  int length = snprintf(storage, storage_length, "%.*s%.*s", (int)prefix.size,
                        prefix.data, (int)value.size, value.data);
  if (length <= 0 || (iree_host_size_t)length >= storage_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM llc argument exceeds storage");
  }
  arguments->values[arguments->count] =
      iree_make_string_view(storage, (iree_host_size_t)length);
  ++arguments->count;
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_profile_llc_arguments(
    const loom_llvmir_target_profile_t* profile,
    loom_llvmir_target_profile_llc_arguments_t* out_arguments) {
  if (out_arguments == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM llc argument output is required");
  }
  *out_arguments = (loom_llvmir_target_profile_llc_arguments_t){0};
  if (profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is required");
  }
  if (profile->target_env == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target environment is required");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mtriple="), profile->target_env->target_triple, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mcpu="), profile->target_cpu, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mattr="), profile->target_features, out_arguments));
  return iree_ok_status();
}
