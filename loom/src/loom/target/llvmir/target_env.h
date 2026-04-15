// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM target environment descriptions.
//
// Target environments collect the pieces of target state that should be chosen
// by target/ABI selection, not by the generic LLVM IR module builder. The
// module stores a copied target_config snapshot; these descriptions are the
// reusable source data used while constructing that snapshot and
// target-specific pointer/address-space types.

#ifndef LOOM_TARGET_LLVMIR_TARGET_ENV_H_
#define LOOM_TARGET_LLVMIR_TARGET_ENV_H_

#include "loom/target/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_llvmir_object_format_e {
  LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN = 0,
  LOOM_LLVMIR_OBJECT_FORMAT_ELF = 1,
  LOOM_LLVMIR_OBJECT_FORMAT_COFF = 2,
  LOOM_LLVMIR_OBJECT_FORMAT_MACHO = 3,
} loom_llvmir_object_format_t;

typedef struct loom_llvmir_target_address_spaces_t {
  // Generic/default pointer address space.
  uint32_t generic;
  // Device or process global memory address space.
  uint32_t global;
  // Workgroup/shared memory address space.
  uint32_t local;
  // Constant memory address space.
  uint32_t constant;
  // Per-invocation/private memory address space.
  uint32_t private_memory;
  // Target-specific buffer-resource pointer address space, or UINT32_MAX.
  uint32_t buffer_resource;
} loom_llvmir_target_address_spaces_t;

typedef struct loom_llvmir_target_env_t {
  // Stable target environment name for diagnostics and tests.
  iree_string_view_t name;
  // LLVM target triple.
  iree_string_view_t target_triple;
  // LLVM data layout, or empty when supplied by a downstream toolchain.
  iree_string_view_t data_layout;
  // Linkable object format produced by this target.
  loom_llvmir_object_format_t object_format;
  // Default pointer bit width for target-independent layout decisions.
  uint32_t default_pointer_bitwidth;
  // Index bit width chosen for lowered index values.
  uint32_t index_bitwidth;
  // Offset bit width chosen for lowered byte offsets.
  uint32_t offset_bitwidth;
  // Address space assignments used by target-specific ABI lowering.
  loom_llvmir_target_address_spaces_t address_spaces;
} loom_llvmir_target_env_t;

const loom_llvmir_target_env_t* loom_llvmir_target_env_x86_64_unknown_linux_gnu(
    void);

const loom_llvmir_target_env_t* loom_llvmir_target_env_amdgcn_amd_amdhsa(void);

iree_status_t loom_llvmir_target_env_module_config(
    const loom_llvmir_target_env_t* target_env, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TARGET_ENV_H_
