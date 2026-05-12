// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/descriptor_semantics.h"

static bool loom_amdgpu_descriptor_has_semantic_tag(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t expected) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->semantic_tag_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return false;
  }
  const iree_string_view_t semantic_tag = loom_low_descriptor_set_string(
      descriptor_set, descriptor->semantic_tag_string_offset);
  return iree_string_view_equal(semantic_tag, expected);
}

bool loom_amdgpu_descriptor_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_resource_kind_t kind) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return false;
  }
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  const uint32_t end = (uint32_t)schedule_class->issue_use_start +
                       schedule_class->issue_use_count;
  if (end > descriptor_set->issue_use_count) {
    return false;
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &descriptor_set->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >= descriptor_set->resource_count) {
      continue;
    }
    if (descriptor_set->resources[issue_use->resource_id].kind == kind) {
      return true;
    }
  }
  return false;
}

bool loom_amdgpu_descriptor_uses_vector_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return loom_amdgpu_descriptor_uses_resource_kind(
      descriptor_set, descriptor, LOOM_LOW_RESOURCE_KIND_VECTOR_ALU);
}

bool loom_amdgpu_descriptor_is_transcendental(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return loom_amdgpu_descriptor_has_semantic_tag(descriptor_set, descriptor,
                                                 IREE_SV("float.exp2.f32")) ||
         loom_amdgpu_descriptor_has_semantic_tag(descriptor_set, descriptor,
                                                 IREE_SV("float.log2.f32")) ||
         loom_amdgpu_descriptor_has_semantic_tag(
             descriptor_set, descriptor, IREE_SV("float.sin_turns.f32")) ||
         loom_amdgpu_descriptor_has_semantic_tag(
             descriptor_set, descriptor, IREE_SV("float.cos_turns.f32")) ||
         loom_amdgpu_descriptor_has_semantic_tag(descriptor_set, descriptor,
                                                 IREE_SV("float.sqrt.f32")) ||
         loom_amdgpu_descriptor_has_semantic_tag(descriptor_set, descriptor,
                                                 IREE_SV("float.rsqrt.f32")) ||
         loom_amdgpu_descriptor_has_semantic_tag(
             descriptor_set, descriptor, IREE_SV("float.reciprocal.f32"));
}
