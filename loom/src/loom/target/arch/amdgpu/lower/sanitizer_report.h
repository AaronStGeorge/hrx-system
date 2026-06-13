// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low sanitizer feedback report builders.
//
// The compiler-side sanitizer model is intentionally generic. The current
// AMDGPU HAL feedback channel exposes sanitizer access failures through the
// ASAN feedback packet schema, so this file is the narrow boundary that maps
// Loom sanitizer reports onto that target runtime ABI.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_REPORT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_REPORT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;
typedef struct loom_block_t loom_block_t;
typedef struct loom_amdgpu_feedback_packet_address_t
    loom_amdgpu_feedback_packet_address_t;

// ABI version for the AMDGPU sanitizer access report payload.
#define LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ABI_VERSION 0u

// Byte length of the AMDGPU sanitizer access report payload.
#define LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_BYTE_LENGTH 64u

// Trap ID used after emitting a sanitizer report.
#define LOOM_AMDGPU_SANITIZER_TRAP_ID 0x100u

// Access kind values carried by AMDGPU sanitizer access reports.
typedef uint32_t loom_amdgpu_sanitizer_access_kind_t;

enum loom_amdgpu_sanitizer_access_kind_e {
  // Access kind was not provided by the instrumentation site.
  LOOM_AMDGPU_SANITIZER_ACCESS_KIND_UNKNOWN = 0u,
  // Instrumented read access.
  LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ = 1u,
  // Instrumented write access.
  LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE = 2u,
  // Instrumented atomic read-modify-write access.
  LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC = 3u,
};

// Bitfield specifying properties of an AMDGPU sanitizer access report.
typedef uint32_t loom_amdgpu_sanitizer_report_flags_t;

enum loom_amdgpu_sanitizer_report_flag_bits_e {
  // No report-level flags are set.
  LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE = 0u,
};

enum loom_amdgpu_sanitizer_access_report_layout_e {
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_RECORD_LENGTH_OFFSET = 0u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ABI_VERSION_OFFSET = 4u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ACCESS_KIND_OFFSET = 8u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_FLAGS_OFFSET = 12u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_FAULT_ADDRESS_OFFSET = 16u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ACCESS_SIZE_OFFSET = 24u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_SITE_ID_OFFSET = 32u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_SHADOW_ADDRESS_OFFSET = 40u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_SHADOW_VALUE_OFFSET = 48u,
  LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_RESERVED0_OFFSET = 56u,
};

typedef struct loom_amdgpu_sanitizer_access_report_t {
  // Instrumented access kind that triggered the report.
  loom_amdgpu_sanitizer_access_kind_t access_kind;
  // Report flags.
  loom_amdgpu_sanitizer_report_flags_t flags;
  // Application address that failed the sanitizer check.
  loom_value_id_t fault_address;
  // Access size in bytes.
  loom_value_id_t access_size;
  // Compiler-assigned site token or dense site ID.
  loom_value_id_t site_id;
  // Shadow address consulted by the check, or zero when unavailable.
  loom_value_id_t shadow_address;
  // Shadow value observed by the check, or zero when unavailable.
  loom_value_id_t shadow_value;
} loom_amdgpu_sanitizer_access_report_t;

typedef struct loom_amdgpu_sanitizer_report_source_t {
  // Device-visible dispatch packet pointer captured for host diagnostics.
  loom_value_id_t dispatch_ptr;
  // X dimension workgroup id captured for host diagnostics.
  loom_value_id_t workgroup_id_x;
  // X dimension workitem id captured for host diagnostics.
  loom_value_id_t workitem_id_x;
} loom_amdgpu_sanitizer_report_source_t;

typedef struct loom_amdgpu_sanitizer_access_report_trap_island_t {
  // Entry block accepting the source/report tuple for one failed access site.
  loom_block_t* entry_block;
  // Final trap block reached after optional reporting or when reporting is not
  // available.
  loom_block_t* trap_block;
  // Access kind handled by this island.
  loom_amdgpu_sanitizer_access_kind_t access_kind;
  // Report flags handled by this island.
  loom_amdgpu_sanitizer_report_flags_t flags;
  // Block arguments carrying source coordinates in |entry_block|.
  loom_amdgpu_sanitizer_report_source_t source_args;
  // Block arguments carrying access report values in |entry_block|.
  loom_amdgpu_sanitizer_access_report_t report_args;
} loom_amdgpu_sanitizer_access_report_trap_island_t;

// Emits the AMDGPU sanitizer access report payload into a reserved feedback
// packet.
//
// The generic feedback packet header must be emitted separately with kind
// LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN and a payload length of
// LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_BYTE_LENGTH. This helper writes only the
// payload bytes beginning at LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH.
iree_status_t loom_amdgpu_build_sanitizer_access_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location);

// Emits a fatal AMDGPU sanitizer access report producer.
//
// The builder must be positioned at the end of a target-low block. This helper
// emits a cold CFG that loads the runtime feedback config, branches around
// feedback channel dereferences when feedback is disabled, reserves packet
// storage when possible, writes and publishes one ASAN-kind feedback packet on
// successful reservation, and emits S_TRAP on all paths. The builder is left in
// the trap block after the S_TRAP packet; callers remain responsible for adding
// the target-low terminator required by the surrounding function.
iree_status_t loom_amdgpu_build_sanitizer_access_report_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location);

// Builds a shared cold report/trap island for failed access assertions.
//
// The island accepts one failed access site's report tuple as block arguments,
// attempts to enqueue a feedback packet, and traps whether reporting succeeds,
// is disabled, or cannot reserve packet storage. Call
// loom_amdgpu_build_sanitizer_access_report_trap_branch from each per-site cold
// block to enter the island. Leaves the builder positioned at the island's trap
// block after the trap packet has been emitted.
iree_status_t loom_amdgpu_build_sanitizer_access_report_trap_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* after_block, loom_symbol_ref_t feedback_config_symbol,
    loom_amdgpu_sanitizer_access_kind_t access_kind,
    loom_amdgpu_sanitizer_report_flags_t flags, loom_location_id_t location,
    loom_amdgpu_sanitizer_access_report_trap_island_t* out_island);

// Terminates the current cold block with a branch into |island|.
//
// Values are converted to the island's canonical block-argument register
// classes in the current block, so this should only be used on the
// already-failing path.
iree_status_t loom_amdgpu_build_sanitizer_access_report_trap_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_report_trap_island_t* island,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_REPORT_H_
