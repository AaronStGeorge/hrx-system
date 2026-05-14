// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_SPIRV_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_storage_class_e {
  // UniformConstant storage class.
  LOOM_SPIRV_STORAGE_CLASS_UNIFORM_CONSTANT = 0,
  // Uniform storage class.
  LOOM_SPIRV_STORAGE_CLASS_UNIFORM = 2,
  // Workgroup storage class.
  LOOM_SPIRV_STORAGE_CLASS_WORKGROUP = 4,
  // CrossWorkgroup storage class.
  LOOM_SPIRV_STORAGE_CLASS_CROSS_WORKGROUP = 5,
  // Function storage class.
  LOOM_SPIRV_STORAGE_CLASS_FUNCTION = 7,
  // Generic storage class.
  LOOM_SPIRV_STORAGE_CLASS_GENERIC = 8,
  // StorageBuffer storage class.
  LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER = 12,
  // PhysicalStorageBuffer storage class used by buffer device addresses.
  LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER = 5349,
} loom_spirv_storage_class_t;

// Target bundle table consumed by the generated spirv.target op.
extern const loom_target_bundle_table_t loom_spirv_target_bundles;

// Vulkan 1.3 SPIR-V target bundle.
extern const loom_target_bundle_t loom_spirv_low_target_bundle_vulkan1_3;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_TARGET_RECORDS_H_
