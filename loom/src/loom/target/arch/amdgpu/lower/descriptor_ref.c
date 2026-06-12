// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/ops/op_defs.h"

iree_status_t loom_amdgpu_lookup_descriptor_ref(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id) {
  *out_descriptor = NULL;
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  const uint32_t descriptor_ordinal =
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "generated AMDGPU lowering references missing descriptor ref %" PRIu16,
        descriptor_ref);
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, key, out_opcode_id));
  *out_descriptor = descriptor;
  return iree_ok_status();
}
