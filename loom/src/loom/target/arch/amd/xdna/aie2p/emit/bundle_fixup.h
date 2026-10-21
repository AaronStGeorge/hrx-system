// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_FIXUP_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_FIXUP_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Detached address-fixup tables awaiting final physical bundle placement.
typedef struct loom_aie2p_bundle_address_fixups_t {
  // Number of scheduled packets addressable by the packet-index tables.
  iree_host_size_t scheduled_packet_count;
  // Storage fixups retained while their instructions were encoded.
  loom_aie2p_planned_storage_fixup_t* storage_fixups;
  // Number of records in |storage_fixups|.
  iree_host_size_t storage_fixup_count;
  // Storage-fixup index for each scheduled packet, or UINT32_MAX.
  const uint32_t* storage_fixup_indices_by_packet;
  // Read-only data fixups retained while their instructions were encoded.
  loom_aie2p_planned_read_only_data_fixup_t* read_only_data_fixups;
  // Number of records in |read_only_data_fixups|.
  iree_host_size_t read_only_data_fixup_count;
  // Read-only data fixup index for each scheduled packet, or UINT32_MAX.
  const uint32_t* read_only_data_fixup_indices_by_packet;
} loom_aie2p_bundle_address_fixups_t;

// Resolves each address fixup to the unique final physical bundle containing
// its scheduled instruction.
void loom_aie2p_bundle_fixup_resolve_addresses(
    const loom_aie2p_planned_bundle_t* bundles, iree_host_size_t bundle_count,
    const loom_aie2p_planned_slot_t* slots,
    loom_aie2p_bundle_address_fixups_t* fixups);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_FIXUP_H_
