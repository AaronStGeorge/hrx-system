// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low helpers for AMDGPU device feedback producers.
//
// The feedback channel is a generic runtime contract used by diagnostics and
// future device-to-host facilities. Sanitizers are one packet schema carried by
// the channel; channel addressing, configuration loads, reservation, and
// publication belong here instead of in sanitizer-specific lowering code.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_FEEDBACK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_FEEDBACK_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/feedback_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

typedef struct loom_amdgpu_feedback_config_values_t {
  // Address of the runtime-published feedback config global.
  loom_value_id_t address;
  // Feedback config flags loaded from loom_amdgpu_feedback_config_layout_e.
  loom_value_id_t flags;
  // Device-visible feedback channel pointer loaded from the config.
  loom_value_id_t channel_base;
  // Runtime-published host notification signal loaded from the config.
  loom_value_id_t notify_signal;
} loom_amdgpu_feedback_config_values_t;

typedef struct loom_amdgpu_feedback_channel_header_values_t {
  // Device-visible feedback channel header address loaded from the config.
  loom_value_id_t address;
  // Channel record byte length loaded from
  // loom_amdgpu_feedback_channel_layout_e.
  loom_value_id_t record_length;
  // Channel ABI version loaded from loom_amdgpu_feedback_channel_layout_e.
  loom_value_id_t abi_version;
  // Channel flags loaded from loom_amdgpu_feedback_channel_layout_e.
  loom_value_id_t flags;
  // Device-visible feedback packet ring pointer loaded from the channel.
  loom_value_id_t ring_base;
  // Feedback packet ring capacity in bytes loaded from the channel.
  loom_value_id_t ring_capacity;
} loom_amdgpu_feedback_channel_header_values_t;

typedef struct loom_amdgpu_feedback_packet_header_t {
  // Total packet byte length including the fixed header and padded payload.
  uint32_t record_length;
  // Packet schema kind owned by the producer facility.
  loom_amdgpu_feedback_packet_kind_t kind;
  // Packet-level behavior flags.
  loom_amdgpu_feedback_packet_flags_t flags;
  // Absolute ring byte position assigned by reservation.
  loom_value_id_t sequence;
  // Device-visible dispatch packet pointer captured by the producer.
  loom_value_id_t source_dispatch_ptr;
  // X dimension workgroup id captured by the producer.
  loom_value_id_t source_workgroup_id_x;
  // X dimension workitem id captured by the producer.
  loom_value_id_t source_workitem_id_x;
} loom_amdgpu_feedback_packet_header_t;

// Emits target-low IR that materializes and scalar-loads common feedback config
// fields.
//
// The loaded fields are uniform across the dispatch and are emitted as SGPR
// values. This helper does not emit control flow, reserve packet storage,
// validate the channel header, or publish packets; those steps are shared by
// concrete producers and build on top of the values returned here.
iree_status_t loom_amdgpu_build_feedback_config_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t config_symbol, loom_location_id_t location,
    loom_amdgpu_feedback_config_values_t* out_values);

// Emits target-low IR that scalar-loads the stable feedback channel header
// fields needed by packet producers.
//
// The volatile producer/consumer cursors intentionally are not loaded here:
// reservation must use scoped atomic operations with the ordering required by
// the runtime feedback ABI.
iree_status_t loom_amdgpu_build_feedback_channel_header_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_amdgpu_feedback_channel_header_values_t* out_values);

// Emits target-low IR that initializes a reserved feedback packet header.
//
// |packet_base| must be the SGPRx2 device-visible packet address returned by
// reservation. Dynamic header values may be SGPR or VGPR low registers and are
// copied into VGPRs as required by global-store packet operands.
iree_status_t loom_amdgpu_build_feedback_packet_header(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t packet_base,
    const loom_amdgpu_feedback_packet_header_t* header,
    loom_location_id_t location);

// Emits target-low IR that release-publishes a reserved feedback packet state.
//
// |packet_base| must be the SGPRx2 device-visible packet address returned by
// reservation. This only publishes the packet state; producer-specific code is
// responsible for emitting any later host notification or failure policy.
iree_status_t loom_amdgpu_build_feedback_publish_packet_state(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t packet_base, loom_location_id_t location);

// Emits an AMDGPU message-send packet with the given target-defined message
// immediate.
iree_status_t loom_amdgpu_build_feedback_send_message(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_location_id_t location);

// Emits an AMDGPU message-send packet that returns a 32-bit scalar result.
iree_status_t loom_amdgpu_build_feedback_send_message_rtn_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_location_id_t location, loom_value_id_t* out_value);

// Emits an AMDGPU halt packet with the given target-defined reason immediate.
iree_status_t loom_amdgpu_build_feedback_halt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t reason, loom_location_id_t location);

// Emits an AMDGPU trap packet with the given target-defined trap id.
iree_status_t loom_amdgpu_build_feedback_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t trap_id, loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_FEEDBACK_H_
