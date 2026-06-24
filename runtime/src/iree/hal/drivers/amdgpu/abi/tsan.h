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

// ABI version for |iree_hal_amdgpu_tsan_assignment_plan_t|.
#define IREE_HAL_AMDGPU_TSAN_ASSIGNMENT_PLAN_ABI_VERSION_0 0u

// ABI version for |iree_hal_amdgpu_tsan_queue_state_t|.
#define IREE_HAL_AMDGPU_TSAN_QUEUE_STATE_ABI_VERSION_0 0u

// ABI version for |iree_hal_amdgpu_tsan_dispatch_state_t|.
#define IREE_HAL_AMDGPU_TSAN_DISPATCH_STATE_ABI_VERSION_0 0u

// Bitfield specifying properties of the TSAN configuration.
typedef uint32_t iree_hal_amdgpu_tsan_config_flags_t;
enum iree_hal_amdgpu_tsan_config_flag_bits_t {
  IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_NONE = 0u,
  // TSAN race checking is enabled for the owning logical device.
  IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_ENABLED = 1u << 0,
  // Queue-scoped TSAN state is available through queue-local config fields.
  IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_STATE = 1u << 1,
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
  // Prior workitem id is stored as a linear local id in prior_workitem_id_x.
  IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR = 1u << 2,
  // Current workitem id is stored as a linear local id in
  // current_workitem_id_x.
  IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR = 1u << 3,
};

// Bitfield specifying properties of a TSAN assignment plan.
typedef uint16_t iree_hal_amdgpu_tsan_assignment_plan_flags_t;
enum iree_hal_amdgpu_tsan_assignment_plan_flag_bits_t {
  IREE_HAL_AMDGPU_TSAN_ASSIGNMENT_PLAN_FLAG_NONE = 0u,
};

// Bitfield specifying properties of a TSAN assignment record.
typedef uint16_t iree_hal_amdgpu_tsan_assignment_record_flags_t;
enum iree_hal_amdgpu_tsan_assignment_record_flag_bits_t {
  IREE_HAL_AMDGPU_TSAN_ASSIGNMENT_RECORD_FLAG_NONE = 0u,
};

// Bitfield specifying properties of queue-owned TSAN state.
typedef uint32_t iree_hal_amdgpu_tsan_queue_state_flags_t;
enum iree_hal_amdgpu_tsan_queue_state_flag_bits_t {
  IREE_HAL_AMDGPU_TSAN_QUEUE_STATE_FLAG_NONE = 0u,
};

// Bitfield specifying properties of a queue-local dispatch state slot.
typedef uint32_t iree_hal_amdgpu_tsan_dispatch_state_flags_t;
enum iree_hal_amdgpu_tsan_dispatch_state_flag_bits_t {
  IREE_HAL_AMDGPU_TSAN_DISPATCH_STATE_FLAG_NONE = 0u,
  // The slot was assigned by TSAN queue setup.
  IREE_HAL_AMDGPU_TSAN_DISPATCH_STATE_FLAG_ASSIGNED = 1u << 0,
};

// Runtime-published TSAN configuration read by instrumented device code.
//
// Queue-scoped executable variants publish queue-specific packet geometry so
// device code can derive a queue-local dispatch slot from its HSA dispatch ID
// without host per-dispatch mutation.
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
  // Host-observed base of the owning queue AQL ring, or 0 when unavailable.
  uint64_t queue_aql_base;
  // Power-of-two AQL packet slot mask used with the HSA dispatch ID.
  uint64_t queue_aql_slot_mask;
  // Device-visible iree_hal_amdgpu_tsan_queue_state_t pointer, or zero.
  uint64_t queue_state_base;
  // Number of queue-local dispatch shadow slots available.
  uint32_t shadow_slot_count;
  // Reserved for future TSAN runtime state. Must be zero.
  uint32_t reserved0;
  // Fast-path device-visible base of dispatch-state entries, or zero.
  uint64_t dispatch_state_base;
} iree_hal_amdgpu_tsan_config_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_tsan_config_t) == 96,
                          "TSAN config size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_config_t, shadow_base) == 16,
    "TSAN config shadow fields must follow the fixed header");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_config_t, queue_state_base) == 72,
    "TSAN config queue state fields must follow the queue AQL fields");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_config_t, dispatch_state_base) ==
        88,
    "TSAN config dispatch state base must be in the tail slot");

// Queue-owned TSAN state read by assignment dispatches and instrumented
// kernels.
//
// One state record exists for each logical AMDGPU host queue when queue-scoped
// TSAN support is enabled. The record owns the stable queue geometry and points
// at the mutable dispatch-state table indexed by AQL packet slot. Assignment
// dispatches publish per-payload dispatch state into that table before the
// payload packets become visible to the hardware scheduler.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_tsan_queue_state_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Flags from iree_hal_amdgpu_tsan_queue_state_flag_bits_t.
  iree_hal_amdgpu_tsan_queue_state_flags_t flags;
  // Flattened logical queue ordinal in the owning HAL device.
  uint32_t queue_ordinal;
  // Physical GPU device ordinal owning this queue.
  uint32_t physical_device_ordinal;
  // Queue ordinal relative to |physical_device_ordinal|.
  uint32_t physical_queue_ordinal;
  // Number of entries in the dispatch-state table.
  uint32_t dispatch_state_count;
  // Number of queue-local dispatch shadow slots available.
  uint32_t shadow_slot_count;
  // Host-observed base address of the HSA AQL packet ring.
  uint64_t aql_ring_base;
  // Power-of-two packet-ring slot mask.
  uint64_t aql_ring_mask;
  // Device-visible base of iree_hal_amdgpu_tsan_dispatch_state_t entries.
  uint64_t dispatch_state_base;
  // Device-visible base of this queue's shadow allocation.
  uint64_t shadow_base;
  // Total bytes in this queue's shadow allocation.
  uint64_t shadow_size;
  // Bytes reserved for one dispatch shadow slot.
  uint64_t dispatch_shadow_stride;
  // Bytes reserved for one workgroup inside a dispatch shadow slot.
  uint64_t workgroup_shadow_stride;
  // Maximum workgroup ordinals represented by one dispatch shadow slot.
  uint32_t workgroup_capacity;
  // Bytes in each shadow entry.
  uint32_t shadow_entry_size;
  // Log2 application memory bytes represented by one shadow entry.
  uint32_t memory_granule_shift;
  // Reserved for future queue state. Must be zero.
  uint32_t reserved0;
  // Reserved for future queue state. Must be zero.
  uint64_t reserved1;
} iree_hal_amdgpu_tsan_queue_state_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_tsan_queue_state_t) == 112,
                          "TSAN queue state size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_queue_state_t, aql_ring_base) ==
        32,
    "TSAN queue state AQL fields must follow the fixed header");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_queue_state_t, shadow_base) == 56,
    "TSAN queue state shadow fields must follow dispatch-state fields");

// Queue-local dispatch TSAN state indexed by an AQL packet slot.
//
// Assignment dispatches update the state for each following TSAN-instrumented
// payload dispatch in the same command-buffer block. Instrumented kernels read
// their own slot to select the queue-local shadow slot and replay generation.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_tsan_dispatch_state_t {
  // Replay generation assigned to the dispatch using this AQL slot.
  uint64_t generation;
  // Queue-local shadow slot assigned to the dispatch using this AQL slot.
  uint32_t shadow_slot;
  // Flags from iree_hal_amdgpu_tsan_dispatch_state_flag_bits_t.
  iree_hal_amdgpu_tsan_dispatch_state_flags_t flags;
} iree_hal_amdgpu_tsan_dispatch_state_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_tsan_dispatch_state_t) == 16,
                          "TSAN dispatch state size is part of the device ABI");

// TSAN shadow entry bit layout used for workgroup-local memory race detection.
enum iree_hal_amdgpu_tsan_shadow_entry_layout_t {
  // Byte length of each TSAN local-memory shadow entry.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_BYTE_LENGTH = 8u,
  // Per-workgroup shadow header byte length.
  IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_BYTE_LENGTH = 8u,
  // Offset of the per-workgroup barrier epoch inside the shadow header.
  IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_EPOCH_OFFSET = 0u,
  // Bit offset of the encoded prior access kind in a shadow entry.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_SHIFT = 0u,
  // Bit count of the encoded prior access kind in a shadow entry.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT = 3u,
  // Bit offset of the encoded prior linear local workitem id.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT = 3u,
  // Bit count of the encoded prior linear local workitem id.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT = 10u,
  // Bit offset of the encoded workgroup barrier epoch.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT = 13u,
  // Bit count of the encoded workgroup barrier epoch.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT = 10u,
  // Bit offset of the encoded dispatch generation low bits.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT = 23u,
  // Bit count of the encoded dispatch generation low bits.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT = 20u,
  // Bit offset of the encoded instrumentation site low bits.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT = 43u,
  // Bit count of the encoded instrumentation site low bits.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT = 21u,
};

// Encoded access kind stored in each TSAN shadow entry.
typedef uint32_t iree_hal_amdgpu_tsan_shadow_access_kind_t;
enum iree_hal_amdgpu_tsan_shadow_access_kind_bits_t {
  // No prior access has been recorded for the shadow entry.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY = 0u,
  // Prior access was a non-atomic read.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ = 1u,
  // Prior access was a non-atomic write.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE = 2u,
  // Prior access was a non-atomic read-modify-write observation.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE = 3u,
  // Prior access was an atomic memory operation.
  IREE_HAL_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC = 4u,
};

// Device-readable assignment plan for one TSAN-instrumented command-buffer
// block.
//
// The host materializes this once while finalizing a command buffer. A block
// prefix assignment dispatch reads the plan at replay time and publishes
// queue-local TSAN state for the payload dispatch packets described by the
// following records. The plan is immutable after command-buffer finalization;
// replay supplies only queue-specific state and a dynamic generation epoch.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_tsan_assignment_plan_t {
  // Size of this header in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this plan header layout.
  uint32_t abi_version;
  // Number of assignment records following this header.
  uint32_t record_count;
  // Maximum simultaneously live local-memory shadow slots needed by the block.
  uint32_t max_live_shadow_slots;
  // Static addend folded into the dynamic replay generation epoch.
  uint32_t generation_base;
  // Flags from iree_hal_amdgpu_tsan_assignment_plan_flag_bits_t.
  iree_hal_amdgpu_tsan_assignment_plan_flags_t flags;
  // Reserved for future plan metadata. Must be zero.
  uint16_t reserved0;
} iree_hal_amdgpu_tsan_assignment_plan_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_tsan_assignment_plan_t) == 24,
    "TSAN assignment plan size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_assignment_plan_t,
                         generation_base) == 16,
    "TSAN assignment plan generation fields must follow the fixed header");

// Static assignment data for one TSAN-instrumented payload dispatch.
//
// The assignment dispatch derives the target queue slot by adding
// |packet_delta| to its own HSA dispatch ID and wrapping through the queue AQL
// ring mask. This keeps immutable plans independent of host-observed AQL packet
// addresses.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_tsan_assignment_record_t {
  // Target dispatch packet delta from the assignment packet, in AQL packets.
  uint32_t packet_delta;
  // Static addend folded into the effective replay generation.
  uint32_t generation_delta;
  // Logical shadow slot assigned to the target dispatch within the block span.
  uint32_t shadow_slot;
  // Flags from iree_hal_amdgpu_tsan_assignment_record_flag_bits_t.
  iree_hal_amdgpu_tsan_assignment_record_flags_t flags;
  // Reserved for future record metadata. Must be zero.
  uint16_t reserved0;
} iree_hal_amdgpu_tsan_assignment_record_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_tsan_assignment_record_t) == 16,
    "TSAN assignment record size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_assignment_record_t,
                         shadow_slot) == 8,
    "TSAN assignment record slot fields must follow generation fields");

// Kernargs for the block-prefix TSAN assignment dispatch.
//
// These are the only per-replay values required by the memoized command-buffer
// assignment path. |queue_state_ptr| is intentionally opaque to this header so
// queue state can evolve independently from the immutable plan layout.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_tsan_assignment_args_t {
  // Device pointer to iree_hal_amdgpu_tsan_assignment_plan_t.
  uint64_t plan_ptr;
  // Device pointer to queue-owned TSAN state.
  uint64_t queue_state_ptr;
  // Replay-varying generation epoch.
  uint64_t generation_epoch;
} iree_hal_amdgpu_tsan_assignment_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_tsan_assignment_args_t) == 24,
    "TSAN assignment args size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_assignment_args_t,
                         generation_epoch) == 16,
    "TSAN assignment args epoch must follow device pointers");

// Kernargs for the queue-local TSAN state initialization dispatch.
//
// Queue creation submits one initializer dispatch before user work can enter
// the AQL ring. The dispatch clears mutable per-packet and shadow storage, then
// publishes the immutable queue header from |queue_state_template|. Host code
// keeps an identical mirror for cold executable-global publication and never
// reads this device allocation back.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_tsan_queue_initialize_args_t {
  // Device pointer to the queue-owned TSAN state header.
  iree_hal_amdgpu_tsan_queue_state_t* queue_state;
  // Device pointer to queue-local dispatch-state entries.
  iree_hal_amdgpu_tsan_dispatch_state_t* dispatch_states;
  // Device pointer to queue-local shadow storage.
  void* shadow_base;
  // Byte length of |dispatch_states|.
  uint64_t dispatch_state_length;
  // Byte length of |shadow_base|.
  uint64_t shadow_size;
  // Device pointer to the queue header template in this kernarg record.
  const iree_hal_amdgpu_tsan_queue_state_t* queue_state_template;
  // Header value written to |queue_state| after mutable storage is cleared.
  iree_hal_amdgpu_tsan_queue_state_t queue_state_template_value;
} iree_hal_amdgpu_tsan_queue_initialize_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_tsan_queue_initialize_args_t) == 160,
    "TSAN queue initialize args size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_tsan_queue_initialize_args_t,
                         queue_state_template_value) == 48,
    "TSAN queue initialize args template follows pointer/size fields");

// TSAN diagnostic payload carried by feedback packets of kind TSAN.
//
// The generic feedback packet header carries common source attribution used by
// all feedback packet kinds. This payload carries the TSAN-specific current and
// prior access attribution needed to explain the conflict.
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
  // Workgroup id that produced the current access report.
  uint32_t current_workgroup_id[3];
  // Workitem id that produced the current access report.
  uint32_t current_workitem_id[3];
  // Workgroup id that produced the prior access.
  uint32_t prior_workgroup_id[3];
  // Workitem id that produced the prior access.
  uint32_t prior_workitem_id[3];
} iree_hal_amdgpu_tsan_report_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_tsan_report_t) == 120,
                          "TSAN report payload size is part of the ABI");

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_TSAN_H_
