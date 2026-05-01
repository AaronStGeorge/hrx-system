// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptor_traits.h"

loom_trait_flags_t loom_low_descriptor_effective_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  loom_trait_flags_t traits = 0;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const uint32_t effect_index = descriptor->effect_start + i;
    IREE_ASSERT(effect_index < descriptor_set->effect_count);
    const loom_low_effect_t* effect = &descriptor_set->effects[effect_index];
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ:
        traits |= LOOM_TRAIT_READS_MEMORY;
        break;
      case LOOM_LOW_EFFECT_KIND_WRITE:
        traits |= LOOM_TRAIT_WRITES_MEMORY;
        break;
      case LOOM_LOW_EFFECT_KIND_CONTROL:
        traits |= LOOM_TRAIT_TERMINATOR;
        break;
      case LOOM_LOW_EFFECT_KIND_CALL:
      case LOOM_LOW_EFFECT_KIND_BARRIER:
      case LOOM_LOW_EFFECT_KIND_COUNTER:
        traits |= LOOM_TRAIT_UNKNOWN_EFFECTS;
        break;
      case LOOM_LOW_EFFECT_KIND_CONVERGENT:
        traits |= LOOM_TRAIT_CONVERGENT;
        break;
      default:
        traits |= LOOM_TRAIT_UNKNOWN_EFFECTS;
        break;
    }
  }

  if (iree_any_bit_set(descriptor->flags,
                       LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR)) {
    traits |= LOOM_TRAIT_TERMINATOR;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS)) return traits;

  if (!iree_any_bit_set(traits,
                        LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY)) {
    if (descriptor->effect_count == 0 &&
        iree_any_bit_set(descriptor->flags,
                         LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING)) {
      traits |= LOOM_TRAIT_UNKNOWN_EFFECTS;
    } else {
      traits |= LOOM_TRAIT_PURE;
    }
  }
  return traits;
}
