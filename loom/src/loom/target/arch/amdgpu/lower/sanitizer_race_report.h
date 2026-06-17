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
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_feedback.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;
typedef struct loom_block_t loom_block_t;
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
  // X dimension workgroup id that produced the current access report.
  loom_value_id_t current_workgroup_id_x;
  // Y dimension workgroup id that produced the current access report.
  loom_value_id_t current_workgroup_id_y;
  // Z dimension workgroup id that produced the current access report.
  loom_value_id_t current_workgroup_id_z;
  // X dimension workitem id that produced the current access report.
  loom_value_id_t current_workitem_id_x;
  // Y dimension workitem id that produced the current access report.
  loom_value_id_t current_workitem_id_y;
  // Z dimension workitem id that produced the current access report.
  loom_value_id_t current_workitem_id_z;
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

typedef struct loom_amdgpu_sanitizer_race_report_trap_island_t {
  // Entry block accepting the source/report tuple for one failed race site.
  loom_block_t* entry_block;
  // Final trap block reached after optional reporting or when reporting is not
  // available.
  loom_block_t* trap_block;
  // Block arguments carrying current source coordinates in |entry_block|.
  loom_amdgpu_sanitizer_report_source_t source_args;
  // Block arguments carrying race report values in |entry_block|.
  loom_amdgpu_sanitizer_race_report_t report_args;
} loom_amdgpu_sanitizer_race_report_trap_island_t;

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

// Emits a fatal AMDGPU sanitizer race report producer.
//
// The builder must be positioned at the end of a target-low block. This helper
// emits a cold CFG that loads the runtime feedback config, branches around
// feedback channel dereferences when feedback is disabled, reserves packet
// storage when possible, writes and publishes one TSAN-kind feedback packet on
// successful reservation, and emits S_TRAP on all paths. The builder is left in
// the trap block after the S_TRAP packet; callers remain responsible for adding
// the target-low terminator required by the surrounding function.
iree_status_t loom_amdgpu_build_sanitizer_race_report_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location);

// Builds a shared cold report/trap island for failed race observations.
//
// The island accepts one failed race site's report tuple as block arguments,
// attempts to enqueue a feedback packet, and traps whether reporting succeeds,
// is disabled, or cannot reserve packet storage. Call
// loom_amdgpu_build_sanitizer_race_report_trap_branch from each per-site cold
// block to enter the island. Leaves the builder positioned at the island's trap
// block after the trap packet has been emitted.
iree_status_t loom_amdgpu_build_sanitizer_race_report_trap_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* after_block, loom_symbol_ref_t feedback_config_symbol,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_trap_island_t* out_island);

// Terminates the current cold block with a branch into |island|.
//
// Values are converted to the island's canonical block-argument register
// classes in the current block, so this should only be used on the
// already-failing path.
iree_status_t loom_amdgpu_build_sanitizer_race_report_trap_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_trap_island_t* island,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_REPORT_H_
