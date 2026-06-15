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
  // Runtime-published opaque host source context loaded from the config.
  loom_value_id_t source_context;
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

typedef struct loom_amdgpu_feedback_packet_address_t {
  // Uniform packet base address consumed by GLOBAL_*_SADDR packets.
  loom_value_id_t base;
  // Per-lane byte offset from |base| consumed by GLOBAL_*_SADDR packets.
  loom_value_id_t byte_offset;
} loom_amdgpu_feedback_packet_address_t;

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
  // Runtime-published opaque host source context copied from the config.
  loom_value_id_t source_context;
} loom_amdgpu_feedback_packet_header_t;

typedef struct loom_amdgpu_feedback_reservation_attempt_t {
  // Monotonic reservation head value this attempt tried to claim.
  loom_value_id_t expected_head;
  // Monotonic reservation head value published when the CAS succeeds.
  loom_value_id_t next_head;
  // Monotonic reservation head value observed by the CAS.
  loom_value_id_t observed_head;
  // Native execution mask identifying lanes whose CAS succeeded.
  loom_value_id_t cas_succeeded;
} loom_amdgpu_feedback_reservation_attempt_t;

typedef struct loom_amdgpu_feedback_reservation_t {
  // Packet storage address for the reserved ring slot.
  loom_amdgpu_feedback_packet_address_t packet_address;
  // Absolute ring byte position assigned by reservation.
  loom_value_id_t sequence;
  // Native execution mask identifying lanes with reserved packet storage.
  loom_value_id_t reserved_mask;
} loom_amdgpu_feedback_reservation_t;

// Emits target-low IR that materializes and loads common feedback config
// fields from host-visible system memory.
//
// The loaded fields are uniform across the dispatch and are emitted as SGPR
// values. This helper does not emit control flow, reserve packet storage,
// validate the channel header, or publish packets; those steps are shared by
// concrete producers and build on top of the values returned here.
iree_status_t loom_amdgpu_build_feedback_config_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t config_symbol, loom_location_id_t location,
    loom_amdgpu_feedback_config_values_t* out_values);

// Emits target-low IR that tests whether feedback is enabled in config flags.
//
// |config_flags| must be an SGPR value loaded from
// loom_amdgpu_feedback_config_layout_e. The returned SCC value is suitable for
// low.cond_br. Producers should branch on this before dereferencing
// |channel_base| or |notify_signal| from the runtime-published config.
iree_status_t loom_amdgpu_build_feedback_config_enabled_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t config_flags, loom_location_id_t location,
    loom_value_id_t* out_scc);

// Emits target-low IR that loads the stable feedback channel header fields
// needed by packet producers from host-visible system memory.
//
// The volatile producer/consumer cursors intentionally are not loaded here:
// reservation must use scoped atomic operations with the ordering required by
// the runtime feedback ABI.
iree_status_t loom_amdgpu_build_feedback_channel_header_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_amdgpu_feedback_channel_header_values_t* out_values);

// Emits target-low IR that represents a uniform packet address.
//
// |packet_base| must be an SGPRx2 device-visible packet address. The returned
// address uses |packet_base| as the GLOBAL_*_SADDR base and materializes a zero
// VGPR byte offset. Reservation helpers that compute per-lane ring slots should
// instead populate loom_amdgpu_feedback_packet_address_t with the ring base and
// dynamic ring offset directly.
iree_status_t loom_amdgpu_build_feedback_uniform_packet_address(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t packet_base, loom_location_id_t location,
    loom_amdgpu_feedback_packet_address_t* out_address);

// Emits target-low IR that increments channel->dropped_packet_count.
//
// |channel_base| must be an SGPRx2 pointer to the device-visible feedback
// channel header. The update uses relaxed system-scope atomic semantics because
// the counter is diagnostic telemetry and does not publish packet contents or
// participate in reservation ownership.
iree_status_t loom_amdgpu_build_feedback_dropped_packet_count_increment(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location);

// Emits target-low IR that loads channel->reservation_head.
//
// |channel_base| must be an SGPRx2 pointer to the device-visible feedback
// channel header. The load is a relaxed system-scope cursor read and returns a
// VGPRx2 value because GLOBAL_LOAD_*_SADDR is a vector-memory packet even when
// every active lane reads the same address.
iree_status_t loom_amdgpu_build_feedback_reservation_head_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_value_id_t* out_value);

// Emits target-low IR that loads channel->read_tail with acquire semantics.
//
// |channel_base| must be an SGPRx2 pointer to the device-visible feedback
// channel header. The host drain release-stores this cursor after consuming
// ready packets; device producers use the acquire load when checking capacity
// so subsequent decisions observe the host-published tail.
iree_status_t loom_amdgpu_build_feedback_read_tail_acquire_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_value_id_t* out_value);

// Emits target-low IR that acq-rel compare-exchanges channel->reservation_head.
//
// |channel_base| must be an SGPRx2 pointer to the device-visible feedback
// channel header. |expected_head| and |desired_head| must be 64-bit low
// register values. The returned |out_old_head| is the value observed by the
// atomic; a caller reserves storage only when it equals |expected_head|.
iree_status_t
loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t expected_head,
    loom_value_id_t desired_head, loom_location_id_t location,
    loom_value_id_t* out_old_head);

// Emits one target-low reservation CAS attempt.
//
// |channel_base| must be an SGPRx2 pointer to the device-visible feedback
// channel header. |reservation_head| must be a VGPRx2 value loaded from
// channel->reservation_head. |packet_length| is the statically known total
// packet byte length including header and padded payload.
//
// This helper does not check ring capacity, retry failed CAS operations,
// compute a packet ring offset, initialize packet storage, publish packet
// state, notify the host, or decide producer failure policy. Those steps are
// shared by higher-level feedback producers and are intentionally kept outside
// of this single-attempt primitive.
iree_status_t loom_amdgpu_build_feedback_reservation_attempt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t reservation_head,
    uint32_t packet_length, loom_location_id_t location,
    loom_amdgpu_feedback_reservation_attempt_t* out_attempt);

// Emits target-low CFG that reserves packet storage in the feedback ring.
//
// |channel_base| must be an SGPRx2 pointer to the device-visible feedback
// channel header. |ring_base| and |ring_capacity| must be SGPRx2 fields loaded
// from that channel. |packet_length| is the statically known total packet byte
// length including header and padded payload.
//
// This helper emits from the current low block and leaves |builder| positioned
// at a newly inserted continuation block. The returned values are block
// arguments in that continuation. |reserved_mask| is non-zero on the reserved
// path and zero on the dropped path; producers that must abort can branch on it
// to choose between report+trap and trap-only paths.
//
// The HAL-created channel owns structural validation: capacity is a non-zero
// power of two, fits in 32 bits, and is at least |packet_length|. The generated
// device hot path only checks current availability with low 32-bit cursor
// arithmetic so instrumentation minimally perturbs the original kernel.
iree_status_t loom_amdgpu_build_feedback_reservation(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t ring_base,
    loom_value_id_t ring_capacity, uint32_t packet_length,
    loom_location_id_t location,
    loom_amdgpu_feedback_reservation_t* out_reservation);

// Emits target-low IR that tests whether a reservation produced packet storage.
//
// |reservation_mask| must be the SGPRx2 mask returned by
// loom_amdgpu_build_feedback_reservation. The returned SCC value is suitable
// for low.cond_br.
iree_status_t loom_amdgpu_build_feedback_reservation_succeeded_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t reservation_mask, loom_location_id_t location,
    loom_value_id_t* out_scc);

// Materializes |source| as a VGPR register range for packet stores or cold
// feedback block arguments.
//
// The source may already be a VGPR value, or it may be an SGPR value copied
// lane-wise into VGPRs. Values with any other register class or unit count fail
// loudly because feedback producers cannot encode them in per-lane stores.
iree_status_t loom_amdgpu_build_feedback_vgpr_registers(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, uint32_t expected_unit_count,
    loom_location_id_t location, loom_value_id_t* out_value);

// Emits target-low IR that initializes a reserved feedback packet header.
//
// |packet_address| must reference packet storage returned by reservation.
// Dynamic header values may be SGPR or VGPR low registers and are copied into
// VGPRs as required by global-store packet operands.
iree_status_t loom_amdgpu_build_feedback_packet_header(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_feedback_packet_header_t* header,
    loom_location_id_t location);

// Emits target-low IR that stores one 32-bit value into a feedback packet.
//
// |packet_address| must reference reserved packet storage. |value| may be an
// SGPR or VGPR low register and is copied to a VGPR if required by the selected
// GLOBAL_STORE_* packet descriptor.
iree_status_t loom_amdgpu_build_feedback_packet_store_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location);

// Emits target-low IR that stores one 64-bit value into a feedback packet.
//
// |packet_address| must reference reserved packet storage. |value| may be an
// SGPRx2 or VGPRx2 low register and is copied to VGPRs if required by the
// selected GLOBAL_STORE_* packet descriptor.
iree_status_t loom_amdgpu_build_feedback_packet_store_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location);

// Emits target-low IR that stores one immediate 32-bit value into a feedback
// packet.
iree_status_t loom_amdgpu_build_feedback_packet_store_u32_constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, uint32_t value, loom_location_id_t location);

// Emits target-low IR that stores one immediate 64-bit value into a feedback
// packet.
iree_status_t loom_amdgpu_build_feedback_packet_store_u64_constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, uint64_t value, loom_location_id_t location);

// Emits target-low IR that release-publishes a reserved feedback packet state.
//
// |packet_address| must reference packet storage returned by reservation. This
// only publishes the packet state; producer-specific code is responsible for
// emitting any later host notification or failure policy.
iree_status_t loom_amdgpu_build_feedback_publish_packet_state(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_location_id_t location);

// Emits target-low IR that wakes the runtime feedback service.
//
// |notify_signal| must be the SGPRx2 host-interrupt signal loaded from an
// enabled feedback config. This helper does not emit null checks for the signal
// or its mailbox pointer; callers must only use it after branching onto a path
// where the runtime feedback ABI guarantees an interrupt-capable signal.
iree_status_t loom_amdgpu_build_feedback_notify_host(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t notify_signal, loom_location_id_t location);

// Emits target-low IR that release-publishes a reserved packet and wakes the
// runtime feedback service.
//
// |packet_address| must reference packet storage returned by reservation and
// |notify_signal| must satisfy
// loom_amdgpu_build_feedback_notify_host's preconditions.
iree_status_t loom_amdgpu_build_feedback_publish_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_value_id_t notify_signal, loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_FEEDBACK_H_
