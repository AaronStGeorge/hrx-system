// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low helpers for AMDGPU host-visible system memory operations.
//
// Runtime producer paths such as feedback packets, host signals, host calls,
// and future device-side queue producers all need the same system-scope memory
// policy. This file owns the cache attrs and explicit ordering packets that
// make those producer/consumer contracts visible outside the dispatch.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SYSTEM_MEMORY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SYSTEM_MEMORY_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/target/arch/amdgpu/target_info.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Returns the vector-memory cache policy encoding for |descriptor_set|.
loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_system_memory_cache_policy_encoding(
    const loom_low_descriptor_set_t* descriptor_set);

// Builds a 32-bit integer descriptor attr.
iree_status_t loom_amdgpu_system_memory_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr);

// Builds a byte offset descriptor attr.
iree_status_t loom_amdgpu_system_memory_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset, loom_named_attr_t* out_attr);

// Appends attrs for a system-scope load cursor read.
iree_status_t loom_amdgpu_system_memory_append_load_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Appends attrs for a system-release store that publishes host-visible data.
iree_status_t loom_amdgpu_system_memory_append_release_store_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Appends attrs for a system-scope no-return atomic update.
iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Appends attrs for a system-scope returning atomic update.
iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Emits explicit target-low packets that release prior global writes before a
// following system-scope publication operation.
iree_status_t loom_amdgpu_system_memory_build_release_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location);

// Emits explicit target-low packets that make later global reads observe data
// published by a preceding system-scope acquire load or returning atomic.
iree_status_t loom_amdgpu_system_memory_build_acquire_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SYSTEM_MEMORY_H_
