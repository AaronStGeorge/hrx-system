// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/target_env.h"

const loom_llvmir_target_env_t* loom_llvmir_target_env_x86_64_unknown_linux_gnu(
    void) {
  static const loom_llvmir_target_env_t target_env = {
      .name = IREE_SVL("x86_64-unknown-linux-gnu"),
      .target_triple = IREE_SVL("x86_64-unknown-linux-gnu"),
      .object_format = LOOM_LLVMIR_OBJECT_FORMAT_ELF,
      .default_pointer_bitwidth = 64,
      .index_bitwidth = 64,
      .offset_bitwidth = 64,
      .address_spaces =
          {
              .generic = 0,
              .global = 0,
              .local = 0,
              .constant = 0,
              .private = 0,
              .buffer_resource = UINT32_MAX,
          },
  };
  return &target_env;
}

const loom_llvmir_target_env_t* loom_llvmir_target_env_amdgcn_amd_amdhsa(void) {
  static const loom_llvmir_target_env_t target_env = {
      .name = IREE_SVL("amdgcn-amd-amdhsa"),
      .target_triple = IREE_SVL("amdgcn-amd-amdhsa"),
      .object_format = LOOM_LLVMIR_OBJECT_FORMAT_ELF,
      .default_pointer_bitwidth = 64,
      .index_bitwidth = 32,
      .offset_bitwidth = 64,
      .address_spaces =
          {
              .generic = 0,
              .global = 1,
              .local = 3,
              .constant = 4,
              .private = 5,
              .buffer_resource = 7,
          },
  };
  return &target_env;
}

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
