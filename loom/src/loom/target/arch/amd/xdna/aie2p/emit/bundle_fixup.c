// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_fixup.h"

void loom_aie2p_bundle_fixup_resolve_addresses(
    const loom_aie2p_planned_bundle_t* bundles, iree_host_size_t bundle_count,
    const loom_aie2p_planned_slot_t* slots,
    loom_aie2p_bundle_address_fixups_t* fixups) {
  for (iree_host_size_t bundle_index = 0; bundle_index < bundle_count;
       ++bundle_index) {
    const loom_aie2p_planned_bundle_t* bundle = &bundles[bundle_index];
    for (uint8_t slot_ordinal = 0; slot_ordinal < bundle->slot_count;
         ++slot_ordinal) {
      const loom_aie2p_planned_slot_t* slot =
          &slots[bundle->slot_start + slot_ordinal];
      if (!iree_any_bit_set(
              slot->flags,
              LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_STORAGE_ADDRESS)) {
        if (!iree_any_bit_set(
                slot->flags,
                LOOM_AIE2P_PLANNED_SLOT_FLAG_READ_ONLY_DATA_ADDRESS)) {
          continue;
        }
        IREE_ASSERT_LT(slot->scheduled_packet_index,
                       fixups->scheduled_packet_count);
        const uint32_t matched_fixup_index =
            fixups->read_only_data_fixup_indices_by_packet
                [slot->scheduled_packet_index];
        IREE_ASSERT_NE(matched_fixup_index, UINT32_MAX);
        IREE_ASSERT_LT(matched_fixup_index, fixups->read_only_data_fixup_count);
        IREE_ASSERT_EQ(
            fixups->read_only_data_fixups[matched_fixup_index].bundle_index,
            LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE);
        fixups->read_only_data_fixups[matched_fixup_index].bundle_index =
            (uint32_t)bundle_index;
        continue;
      }
      IREE_ASSERT_LT(slot->scheduled_packet_index,
                     fixups->scheduled_packet_count);
      const uint32_t matched_fixup_index =
          fixups->storage_fixup_indices_by_packet[slot->scheduled_packet_index];
      IREE_ASSERT_NE(matched_fixup_index, UINT32_MAX);
      IREE_ASSERT_LT(matched_fixup_index, fixups->storage_fixup_count);
      IREE_ASSERT_EQ(fixups->storage_fixups[matched_fixup_index].bundle_index,
                     LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE);
      fixups->storage_fixups[matched_fixup_index].bundle_index =
          (uint32_t)bundle_index;
    }
  }
  for (iree_host_size_t i = 0; i < fixups->storage_fixup_count; ++i) {
    IREE_ASSERT_NE(fixups->storage_fixups[i].bundle_index,
                   LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE);
  }
  for (iree_host_size_t i = 0; i < fixups->read_only_data_fixup_count; ++i) {
    IREE_ASSERT_NE(fixups->read_only_data_fixups[i].bundle_index,
                   LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE);
  }
}
