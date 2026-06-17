// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low sanitizer race feedback report builders.
//
// Loom models race diagnostics through sanitizer.race operations. The current
// AMDGPU HAL feedback channel exposes those diagnostics through the TSAN packet
// schema, so this file is the narrow boundary that maps Loom race reports onto
// that target runtime ABI.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_REPORT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_REPORT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;
typedef struct loom_amdgpu_feedback_packet_address_t
    loom_amdgpu_feedback_packet_address_t;

typedef struct loom_amdgpu_sanitizer_race_report_t {
  // TSAN check kind that triggered the report.
  loom_value_id_t check_kind;
  // Report flags describing optional atomic-access properties.
  loom_value_id_t flags;
  // Memory space containing |memory_address|.
  loom_value_id_t memory_space;
  // Access kind performed by the reporting workitem.
  loom_value_id_t current_access_kind;
  // Access kind previously observed for the same memory location.
  loom_value_id_t prior_access_kind;
  // Access size in bytes.
  loom_value_id_t access_size;
  // Compiler-assigned instrumentation site identifier for the current access.
  loom_value_id_t current_site_id;
  // Compiler-assigned instrumentation site identifier for the prior access.
  loom_value_id_t prior_site_id;
  // Address or memory-space-relative byte offset that raced.
  loom_value_id_t memory_address;
  // Shadow address consulted by the check, or zero when unavailable.
  loom_value_id_t shadow_address;
  // Shadow value observed by the check, or zero when unavailable.
  loom_value_id_t shadow_value;
  // X dimension workgroup id that produced the prior access.
  loom_value_id_t prior_workgroup_id_x;
  // Y dimension workgroup id that produced the prior access.
  loom_value_id_t prior_workgroup_id_y;
  // Z dimension workgroup id that produced the prior access.
  loom_value_id_t prior_workgroup_id_z;
  // X dimension workitem id that produced the prior access.
  loom_value_id_t prior_workitem_id_x;
  // Y dimension workitem id that produced the prior access.
  loom_value_id_t prior_workitem_id_y;
  // Z dimension workitem id that produced the prior access.
  loom_value_id_t prior_workitem_id_z;
} loom_amdgpu_sanitizer_race_report_t;

// Emits the AMDGPU sanitizer race report payload into a reserved feedback
// packet.
//
// The generic feedback packet header must be emitted separately with kind
// LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN and a payload length of
// LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH. This helper writes only the payload
// bytes beginning at LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH.
iree_status_t loom_amdgpu_build_sanitizer_race_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_REPORT_H_
