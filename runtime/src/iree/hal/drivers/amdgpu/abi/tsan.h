// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-visible TSAN report ABI shared by AMDGPU host code and instrumented
// device code.

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_TSAN_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_TSAN_H_

#include "iree/hal/drivers/amdgpu/abi/common.h"

// Name of the executable global containing |iree_hal_amdgpu_tsan_config_t|.
#define IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME "iree_tsan_config"

// ABI version for |iree_hal_amdgpu_tsan_config_t|.
#define IREE_HAL_AMDGPU_TSAN_CONFIG_ABI_VERSION_0 0u

// ABI version for |iree_hal_amdgpu_tsan_report_t|.
#define IREE_HAL_AMDGPU_TSAN_REPORT_ABI_VERSION_0 0u

// Bitfield specifying properties of the TSAN configuration.
typedef uint32_t iree_hal_amdgpu_tsan_config_flags_t;
enum iree_hal_amdgpu_tsan_config_flag_bits_t {
  IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_NONE = 0u,
  // TSAN race checking is enabled for the owning logical device.
  IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_ENABLED = 1u << 0,
};

// TSAN check kind that triggered a report.
typedef uint32_t iree_hal_amdgpu_tsan_check_kind_t;
enum iree_hal_amdgpu_tsan_check_kind_bits_t {
  // Check kind was not provided by the instrumentation site.
  IREE_HAL_AMDGPU_TSAN_CHECK_KIND_UNKNOWN = 0u,
  // Conflicting accesses to the same memory location were observed without
  // intervening synchronization.
  IREE_HAL_AMDGPU_TSAN_CHECK_KIND_DATA_RACE = 1u,
};

// TSAN memory space containing the observed address.
typedef uint32_t iree_hal_amdgpu_tsan_memory_space_t;
enum iree_hal_amdgpu_tsan_memory_space_bits_t {
  // Memory space was not provided by the instrumentation site.
  IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_UNKNOWN = 0u,
  // Global device memory.
  IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL = 1u,
  // Workgroup-shared LDS memory.
  IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP = 2u,
  // Per-workitem private memory.
  IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_PRIVATE = 3u,
};

// TSAN access kind participating in a race report.
typedef uint32_t iree_hal_amdgpu_tsan_access_kind_t;
enum iree_hal_amdgpu_tsan_access_kind_bits_t {
  // Access kind was not provided by the instrumentation site.
  IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN = 0u,
  // Instrumented read access.
  IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ = 1u,
  // Instrumented write access.
  IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE = 2u,
  // Instrumented read-modify-write access.
  IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE = 3u,
  // Instrumented atomic access.
  IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_ATOMIC = 4u,
};

// Bitfield specifying properties of a TSAN report.
typedef uint32_t iree_hal_amdgpu_tsan_report_flags_t;
enum iree_hal_amdgpu_tsan_report_flag_bits_t {
  IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_NONE = 0u,
  // Current access was an atomic memory operation.
  IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_CURRENT_ATOMIC = 1u << 0,
  // Prior access was an atomic memory operation.
  IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_PRIOR_ATOMIC = 1u << 1,
};

// Runtime-published TSAN configuration read by instrumented device code.
//
// The first implementation uses one shadow slot per physical device. The ABI is
// intentionally shaped so a queue-sharded executable can later publish
// queue-specific AQL ring coordinates without changing the device-side field
// contract: a dispatch pointer maps to a dispatch slot when |queue_aql_base| is
// non-zero, otherwise slot zero is used.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_tsan_config_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Flags describing enabled TSAN runtime facilities.
  iree_hal_amdgpu_tsan_config_flags_t flags;
  // Log2 application memory bytes represented by one shadow entry.
  uint32_t memory_granule_shift;
  // Device-visible base of the queue or device shadow allocation.
  uint64_t shadow_base;
  // Total bytes in the shadow allocation.
  uint64_t shadow_size;
  // Bytes reserved for one dispatch slot.
  uint64_t dispatch_shadow_stride;
  // Bytes reserved for one workgroup inside a dispatch slot.
  uint64_t workgroup_shadow_stride;
  // Maximum workgroup ordinals represented by one dispatch slot.
  uint32_t workgroup_capacity;
  // Bytes in each shadow entry.
  uint32_t shadow_entry_size;
  // Device-visible base of the owning queue AQL ring, or 0 for slot zero.
  uint64_t queue_aql_base;
  // Power-of-two AQL packet slot mask used with |queue_aql_base|.
  uint64_t queue_aql_slot_mask;
  // Reserved for future TSAN runtime state. Must be zero.
  uint64_t reserved[3];
} iree_hal_amdgpu_tsan_config_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_tsan_config_t) == 96,
                          "TSAN config size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_config_t, shadow_base) == 16,
    "TSAN config shadow fields must follow the fixed header");

// TSAN diagnostic payload carried by feedback packets of kind TSAN.
//
// The generic feedback packet header carries the current dispatch pointer,
// workgroup id, and workitem id. This payload carries TSAN-specific state and
// the prior access attribution needed to explain the conflict.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_tsan_report_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Check kind that triggered the report.
  iree_hal_amdgpu_tsan_check_kind_t check_kind;
  // Flags describing optional report fields.
  iree_hal_amdgpu_tsan_report_flags_t flags;
  // Memory space containing |memory_address|.
  iree_hal_amdgpu_tsan_memory_space_t memory_space;
  // Access kind performed by the reporting workitem.
  iree_hal_amdgpu_tsan_access_kind_t current_access_kind;
  // Access kind previously observed for the same memory location.
  iree_hal_amdgpu_tsan_access_kind_t prior_access_kind;
  // Access size in bytes.
  uint32_t access_size;
  // Compiler-assigned instrumentation site identifier for the current access.
  uint64_t current_site_id;
  // Compiler-assigned instrumentation site identifier for the prior access.
  uint64_t prior_site_id;
  // Address or memory-space-relative byte offset that raced.
  uint64_t memory_address;
  // Shadow address consulted by the check, or 0 when unavailable.
  uint64_t shadow_address;
  // Shadow value observed by the check, or 0 when unavailable.
  uint64_t shadow_value;
  // Workgroup id that produced the prior access.
  uint32_t prior_workgroup_id[3];
  // Workitem id that produced the prior access.
  uint32_t prior_workitem_id[3];
} iree_hal_amdgpu_tsan_report_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_tsan_report_t) == 96,
                          "TSAN report payload size is part of the ABI");

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_TSAN_H_
