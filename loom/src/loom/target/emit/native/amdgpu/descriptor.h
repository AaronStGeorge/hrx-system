// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA kernel descriptor byte emission.
//
// The kernel descriptor is the loader-visible resource contract for one kernel.
// Loom keeps it as structured facts until final object layout gives us the
// descriptor-to-entry offset, then writes the fixed 64-byte AMDHSA descriptor
// directly into a native contribution.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_DESCRIPTOR_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_DESCRIPTOR_H_

#include "iree/base/api.h"
#include "loom/target/emit/native/amdgpu/metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Byte size of an AMDHSA code object v5 kernel descriptor.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH = 64,
  // Required descriptor symbol alignment in bytes.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ALIGNMENT = 64,
};

typedef struct loom_amdgpu_kernel_descriptor_t {
  // Target CPU string such as `gfx1100`.
  iree_string_view_t target_cpu;
  // Fixed LDS/group segment size in bytes.
  uint32_t group_segment_fixed_size;
  // Fixed scratch/private segment size in bytes.
  uint32_t private_segment_fixed_size;
  // Kernel kernarg segment size in bytes.
  uint32_t kernarg_size;
  // Byte offset from the descriptor symbol to the kernel entry instruction.
  int64_t kernel_code_entry_byte_offset;
  // Physical SGPR high-water count for the kernel body.
  uint32_t next_free_sgpr;
  // Physical VGPR high-water count for the kernel body.
  uint32_t next_free_vgpr;
  // Total user SGPR count encoded in COMPUTE_PGM_RSRC2.
  uint32_t user_sgpr_count;
  // True when private-segment-buffer user SGPRs are enabled.
  bool enable_sgpr_private_segment_buffer;
  // True when dispatch-ptr user SGPRs are enabled.
  bool enable_sgpr_dispatch_ptr;
  // True when queue-ptr user SGPRs are enabled.
  bool enable_sgpr_queue_ptr;
  // True when kernarg-segment-ptr user SGPRs are enabled.
  bool enable_sgpr_kernarg_segment_ptr;
  // True when dispatch-id user SGPRs are enabled.
  bool enable_sgpr_dispatch_id;
  // True when flat-scratch-init user SGPRs are enabled.
  bool enable_sgpr_flat_scratch_init;
  // True when private-segment-size user SGPRs are enabled.
  bool enable_sgpr_private_segment_size;
  // True when architected private segment setup is enabled.
  bool enable_private_segment;
  // True when workgroup-id-x system SGPR setup is enabled.
  bool enable_sgpr_workgroup_id_x;
  // True when workgroup-id-y system SGPR setup is enabled.
  bool enable_sgpr_workgroup_id_y;
  // True when workgroup-id-z system SGPR setup is enabled.
  bool enable_sgpr_workgroup_id_z;
  // True when workgroup-info system SGPR setup is enabled.
  bool enable_sgpr_workgroup_info;
  // System VGPR workitem ID mode: 0=x, 1=xy, 2=xyz, 3=undefined.
  uint32_t system_vgpr_workitem_id;
  // True when the kernel executes in wavefront-size-32 mode.
  bool enable_wavefront_size32;
  // True when the kernel may use a dynamically sized stack.
  bool uses_dynamic_stack;
} loom_amdgpu_kernel_descriptor_t;

// Initializes descriptor facts from the metadata kernel record used for the
// same kernel.
//
// Descriptor-only ABI toggles are initialized disabled. Callers that require
// user SGPRs, system SGPRs, or private-segment state must enable them
// explicitly before calling loom_amdgpu_kernel_descriptor_write.
iree_status_t loom_amdgpu_kernel_descriptor_initialize_from_metadata(
    iree_string_view_t target_cpu,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel,
    int64_t kernel_code_entry_byte_offset,
    loom_amdgpu_kernel_descriptor_t* out_descriptor);

// Verifies that descriptor facts still agree with the metadata kernel record
// for fields that both payloads expose.
iree_status_t loom_amdgpu_kernel_descriptor_validate_metadata(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel);

// Writes one AMDHSA kernel descriptor into |target_bytes|.
iree_status_t loom_amdgpu_kernel_descriptor_write(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    iree_byte_span_t target_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_DESCRIPTOR_H_
