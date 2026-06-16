// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builder-level lookup for generated AMDGPU target-low descriptor references.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Resolves |descriptor_ref| to a descriptor row and interns its opcode spelling
// in the builder module.
iree_status_t loom_amdgpu_lookup_descriptor_ref(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id);

// Builds the register type for the implicit resource operand carried by
// |descriptor|. Descriptor rows that use M0 publish it as an implicit resource
// operand.
iree_status_t loom_amdgpu_make_descriptor_implicit_resource_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_type_t* out_type);

// Returns whether |descriptor| declares an immediate named |name|.
bool loom_amdgpu_descriptor_has_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t name);

// Removes optional attrs not declared by |descriptor| while preserving
// the leading |required_count| attrs.
void loom_amdgpu_filter_descriptor_optional_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_host_size_t required_count,
    loom_named_attr_t* attrs, iree_host_size_t* inout_attr_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_
