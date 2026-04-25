// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables
// clang-format off

#ifndef LOOM_OPS_KERNEL_OPS_H_
#define LOOM_OPS_KERNEL_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_KERNEL_BARRIER = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 0),
  LOOM_OP_KERNEL_ASYNC_COPY = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 1),
  LOOM_OP_KERNEL_ASYNC_COPY_MASK = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 2),
  LOOM_OP_KERNEL_ASYNC_GATHER = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 3),
  LOOM_OP_KERNEL_ASYNC_GATHER_MASK = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 4),
  LOOM_OP_KERNEL_ASYNC_GROUP = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 5),
  LOOM_OP_KERNEL_ASYNC_WAIT = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 6),
  LOOM_OP_KERNEL_TENSOR_LDS_DESCRIPTOR = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 7),
  LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 8),
  LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 9),
  LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 10),
  LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 11),
  LOOM_OP_KERNEL_WORKITEM_ID = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 12),
  LOOM_OP_KERNEL_WORKGROUP_ID = LOOM_OP_KIND(LOOM_DIALECT_KERNEL, 13),
  LOOM_OP_KERNEL_COUNT_ = 14,
};

// Target-independent cache scope for memory operations.
typedef enum loom_kernel_cache_scope_e {
  LOOM_KERNEL_CACHE_SCOPE_CU = 0,
  LOOM_KERNEL_CACHE_SCOPE_SE = 1,
  LOOM_KERNEL_CACHE_SCOPE_DEVICE = 2,
  LOOM_KERNEL_CACHE_SCOPE_SYSTEM = 3,
  LOOM_KERNEL_CACHE_SCOPE_COUNT_ = 4,
} loom_kernel_cache_scope_t;

// Target-independent temporal cache policy for memory operations.
typedef enum loom_kernel_cache_temporal_e {
  LOOM_KERNEL_CACHE_TEMPORAL_REGULAR = 0,
  LOOM_KERNEL_CACHE_TEMPORAL_NON_TEMPORAL = 1,
  LOOM_KERNEL_CACHE_TEMPORAL_HIGH_TEMPORAL = 2,
  LOOM_KERNEL_CACHE_TEMPORAL_LAST_USE = 3,
  LOOM_KERNEL_CACHE_TEMPORAL_WRITEBACK = 4,
  LOOM_KERNEL_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR = 5,
  LOOM_KERNEL_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL = 6,
  LOOM_KERNEL_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL = 7,
  LOOM_KERNEL_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK = 8,
  LOOM_KERNEL_CACHE_TEMPORAL_BYPASS = 9,
  LOOM_KERNEL_CACHE_TEMPORAL_COUNT_ = 10,
} loom_kernel_cache_temporal_t;

// Required async copy direction.
typedef enum loom_kernel_direction_e {
  LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP = 0,
  LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL = 1,
  LOOM_KERNEL_DIRECTION_COUNT_ = 2,
} loom_kernel_direction_t;

// Three-dimensional kernel coordinate axis.
typedef enum loom_kernel_dimension_e {
  LOOM_KERNEL_DIMENSION_X = 0,
  LOOM_KERNEL_DIMENSION_Y = 1,
  LOOM_KERNEL_DIMENSION_Z = 2,
  LOOM_KERNEL_DIMENSION_COUNT_ = 3,
} loom_kernel_dimension_t;

// Target-independent memory space fenced by a kernel synchronization op.
typedef enum loom_kernel_barrier_memory_space_e {
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_UNKNOWN = 0,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_GLOBAL = 1,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_WORKGROUP = 2,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_PRIVATE = 3,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_CONSTANT = 4,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_HOST = 5,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_DESCRIPTOR = 6,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_GENERIC = 7,
  LOOM_KERNEL_BARRIER_MEMORY_SPACE_COUNT_ = 8,
} loom_kernel_barrier_memory_space_t;

// Target-independent memory ordering for kernel synchronization ops.
typedef enum loom_kernel_barrier_ordering_e {
  LOOM_KERNEL_BARRIER_ORDERING_RELAXED = 0,
  LOOM_KERNEL_BARRIER_ORDERING_ACQUIRE = 1,
  LOOM_KERNEL_BARRIER_ORDERING_RELEASE = 2,
  LOOM_KERNEL_BARRIER_ORDERING_ACQ_REL = 3,
  LOOM_KERNEL_BARRIER_ORDERING_SEQ_CST = 4,
  LOOM_KERNEL_BARRIER_ORDERING_COUNT_ = 5,
} loom_kernel_barrier_ordering_t;

// Target-independent synchronization scope.
typedef enum loom_kernel_barrier_scope_e {
  LOOM_KERNEL_BARRIER_SCOPE_THREAD = 0,
  LOOM_KERNEL_BARRIER_SCOPE_SUBGROUP = 1,
  LOOM_KERNEL_BARRIER_SCOPE_WORKGROUP = 2,
  LOOM_KERNEL_BARRIER_SCOPE_DEVICE = 3,
  LOOM_KERNEL_BARRIER_SCOPE_SYSTEM = 4,
  LOOM_KERNEL_BARRIER_SCOPE_COUNT_ = 5,
} loom_kernel_barrier_scope_t;

// LOOM_OP_KERNEL_BARRIER: Synchronize invocations in an explicit execution scope and fence a named memory space with a required ordering. The supported kernel barrier is a workgroup execution barrier over workgroup memory with acquire-release ordering. Async-copy completion is modeled by kernel.async.wait; use kernel.barrier only when invocations must rendezvous before consuming shared memory.
// kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}
LOOM_DEFINE_ISA(loom_kernel_barrier_isa, LOOM_OP_KERNEL_BARRIER)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_barrier_memory_space, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_barrier_ordering, 1)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_barrier_scope, 2)
iree_status_t loom_kernel_barrier_build(
    loom_builder_t* builder,
    uint8_t memory_space,
    uint8_t ordering,
    uint8_t scope,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_barrier_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_COPY: Initiate an asynchronous byte-for-byte transfer between two already originated views. The source and destination view types may use different logical element types or shapes, but they must describe the same static byte footprint. The direction attribute makes the required memory-space flow explicit. The returned token must be committed to exactly one kernel.async.group before the copied bytes are waited or consumed.
// %copy = kernel.async.copy %src to %dst {cache_scope = cu, cache_temporal = regular, direction = global_to_workgroup} : view<16xi8> to view<16xi8> -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_copy_isa, LOOM_OP_KERNEL_ASYNC_COPY)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_dest, 1)
LOOM_DEFINE_RESULT(loom_kernel_async_copy_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_copy_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_copy_cache_temporal, 1)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_copy_direction, 2)
iree_status_t loom_kernel_async_copy_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    uint8_t direction,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_copy_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_COPY_MASK: Predicated form of kernel.async.copy. When predicate is true, the op initiates the same transfer as kernel.async.copy. When predicate is false, the op performs no memory access and produces an already complete token so grouping and waiting remain structurally uniform.
// %copy = kernel.async.copy.mask %src to %dst, %in_bounds {cache_scope = cu, cache_temporal = non_temporal, direction = global_to_workgroup} : view<16xi8> to view<16xi8>, i1 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_copy_mask_isa, LOOM_OP_KERNEL_ASYNC_COPY_MASK)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_mask_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_mask_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_copy_mask_predicate, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_copy_mask_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_copy_mask_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_copy_mask_cache_temporal, 1)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_copy_mask_direction, 2)
iree_status_t loom_kernel_async_copy_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t predicate,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    uint8_t direction,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_copy_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_GATHER: Initiate a subgroup-collective asynchronous gather from each invocation's source view into a lane-contiguous workgroup destination view. The destination view has one leading subgroup-lane axis and a trailing lane slot with enough static bytes to hold one source payload. If the lane slot is larger than the source footprint, the extra destination bytes are padding bytes with unspecified contents. The destination denotes the subgroup-uniform base tile; the current subgroup lane is applied by the op semantics and must not be pre-applied by forming a lane subview. This directly represents AMDGPU global_load_lds-style staging, including padded narrow loads, without requiring a later pass to rediscover that a set of per-lane copies was really one subgroup LDS DMA operation.
// %copy = kernel.async.gather %src_lane to %lds_tile {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8> -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_gather_isa, LOOM_OP_KERNEL_ASYNC_GATHER)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_dest, 1)
LOOM_DEFINE_RESULT(loom_kernel_async_gather_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_gather_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_gather_cache_temporal, 1)
iree_status_t loom_kernel_async_gather_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_GATHER_MASK: Predicated form of kernel.async.gather. False predicates perform no source or destination access for the current invocation but still produce a completed token, preserving a uniform async group shape for tails and guarded tiles.
// %copy = kernel.async.gather.mask %src_lane to %lds_tile, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8>, i1 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_gather_mask_isa, LOOM_OP_KERNEL_ASYNC_GATHER_MASK)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_mask_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_mask_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_gather_mask_predicate, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_gather_mask_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_gather_mask_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_gather_mask_cache_temporal, 1)
iree_status_t loom_kernel_async_gather_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t predicate,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_GROUP: Commit zero or more async copy/gather/cluster/tensor tokens into the ordered async stream. Empty groups are valid pipeline markers. The resulting group completes after all committed transfers complete. Groups are ordered by program order; waiting a group also completes older groups in the same stream.
// %empty = kernel.async.group -> kernel.async.group
LOOM_DEFINE_ISA(loom_kernel_async_group_isa, LOOM_OP_KERNEL_ASYNC_GROUP)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_kernel_async_group_tokens, 0)
LOOM_DEFINE_RESULT(loom_kernel_async_group_group, 0)
iree_status_t loom_kernel_async_group_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* tokens,
    iree_host_size_t tokens_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_group_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_WAIT: Wait until a committed async-copy group has completed. This completes the named group and all older groups in the same ordered async stream. The required newer_groups value states how many younger groups are allowed to remain outstanding after the wait, matching AMDGPU wait.asyncmark and NVVM wait_group count semantics without making lowering rediscover the software-pipeline distance from scratch. It is not a workgroup barrier; use kernel.barrier separately when other invocations must observe the copied destination data.
// kernel.async.wait %group {newer_groups = 0} : kernel.async.group
LOOM_DEFINE_ISA(loom_kernel_async_wait_isa, LOOM_OP_KERNEL_ASYNC_WAIT)
LOOM_DEFINE_OPERAND(loom_kernel_async_wait_group, 0)
LOOM_DEFINE_ATTR_I64(loom_kernel_async_wait_newer_groups, 0)
iree_status_t loom_kernel_async_wait_build(
    loom_builder_t* builder,
    loom_value_id_t group,
    int64_t newer_groups,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_wait_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_TENSOR_LDS_DESCRIPTOR: Bundle AMDGPU tensor-memory descriptor groups into one typed SSA value. The dgroups are the exact operands lowered to llvm.amdgcn.tensor.load.to.lds or llvm.amdgcn.tensor.store.from.lds: D0 is vector<4xi32>, D1 is vector<8xi32>, and optional D2/D3 are vector<4xi32>. Gfx1250 uses two-group and four-group descriptor forms; the LLVM intrinsic's fifth D4 operand is lowered as zero because gfx1250 ignores it. The op is pure and contains no memory endpoints; endpoint views are operands of the async tensor ops so fact propagation and alias analysis do not need to decode hardware bitfields.
// %desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1) : vector<4xi32>, vector<8xi32> -> kernel.tensor.lds.descriptor
LOOM_DEFINE_ISA(loom_kernel_tensor_lds_descriptor_isa, LOOM_OP_KERNEL_TENSOR_LDS_DESCRIPTOR)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_kernel_tensor_lds_descriptor_dgroups, 0)
LOOM_DEFINE_RESULT(loom_kernel_tensor_lds_descriptor_descriptor, 0)
iree_status_t loom_kernel_tensor_lds_descriptor_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* dgroups,
    iree_host_size_t dgroups_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_tensor_lds_descriptor_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS: Initiate an AMDGPU gfx1250+ tensor-memory load from a global-like source view into a workgroup/LDS destination view using an explicit kernel.tensor.lds.descriptor. The descriptor supplies the exact hardware dgroups, while the source and destination views keep the logical rank, element type, layout, and memory-space facts visible. The endpoints must have the same rank in [1, 5], the same 1/2/4/8 byte element type, and memory spaces global/constant/descriptor to workgroup. The returned token must be committed to exactly one kernel.async.group.
// %copy = kernel.async.tensor.load.to.lds %global_tile to %lds_tile using %desc {cache_scope = cu, cache_temporal = regular} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_tensor_load_to_lds_isa, LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_load_to_lds_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_load_to_lds_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_load_to_lds_descriptor, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_tensor_load_to_lds_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_tensor_load_to_lds_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_tensor_load_to_lds_cache_temporal, 1)
iree_status_t loom_kernel_async_tensor_load_to_lds_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t descriptor,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_tensor_load_to_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS: Initiate an AMDGPU gfx1250+ tensor-memory store from a workgroup/LDS source view into a global-like destination view using an explicit kernel.tensor.lds.descriptor. The descriptor supplies the exact hardware dgroups, while the source and destination views keep the logical rank, element type, layout, and memory-space facts visible. The endpoints must have the same rank in [1, 5], the same 1/2/4/8 byte element type, and memory spaces workgroup to global/descriptor. The returned token must be committed to exactly one kernel.async.group.
// %copy = kernel.async.tensor.store.from.lds %lds_tile to %global_tile using %desc {cache_scope = device, cache_temporal = non_temporal_writeback} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_tensor_store_from_lds_isa, LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_store_from_lds_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_store_from_lds_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_tensor_store_from_lds_descriptor, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_tensor_store_from_lds_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_tensor_store_from_lds_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_tensor_store_from_lds_cache_temporal, 1)
iree_status_t loom_kernel_async_tensor_store_from_lds_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t descriptor,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_tensor_store_from_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER: Initiate an AMDGPU gfx1250+ cluster asynchronous load from a global-like source view into a workgroup/LDS destination view. The required i32 cluster_mask is the hardware workgroup broadcast mask loaded through M0. Source and destination must have the same static byte footprint, and that footprint must be exactly 1, 4, 8, or 16 bytes; target lowering maps those widths to llvm.amdgcn.cluster.load.async.to.lds.b8/b32/b64/b128. The LLVM offset and cache-policy immediate operands are lowering choices derived from the view address and cache attributes, not separate Loom semantics. The returned token must be committed to exactly one kernel.async.group.
// %copy = kernel.async.cluster.gather %src to %lds using %mask {cache_scope = se, cache_temporal = high_temporal} : view<16xi8> to view<16xi8>, i32 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_cluster_gather_isa, LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_cluster_mask, 2)
LOOM_DEFINE_RESULT(loom_kernel_async_cluster_gather_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_cluster_gather_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_cluster_gather_cache_temporal, 1)
iree_status_t loom_kernel_async_cluster_gather_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t cluster_mask,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_cluster_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK: Predicated form of kernel.async.cluster.gather. False predicates perform no source or destination access for the current invocation but still produce a completed token, preserving a uniform async group shape for tails and guarded tiles. The cluster_mask remains the target workgroup broadcast mask and is distinct from the scalar i1 predicate.
// %copy = kernel.async.cluster.gather.mask %src to %lds using %mask, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<4xi8>, i32, i1 -> kernel.async.token
LOOM_DEFINE_ISA(loom_kernel_async_cluster_gather_mask_isa, LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_source, 0)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_dest, 1)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_cluster_mask, 2)
LOOM_DEFINE_OPERAND(loom_kernel_async_cluster_gather_mask_predicate, 3)
LOOM_DEFINE_RESULT(loom_kernel_async_cluster_gather_mask_token, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_cluster_gather_mask_cache_scope, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_async_cluster_gather_mask_cache_temporal, 1)
iree_status_t loom_kernel_async_cluster_gather_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t dest,
    loom_may_consume loom_value_id_t cluster_mask,
    loom_may_consume loom_value_id_t predicate,
    uint8_t cache_scope,
    uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_kernel_async_cluster_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_KERNEL_WORKITEM_ID: Read one coordinate of the current invocation within its workgroup. The result is a logical index value, not a byte offset; target lowering decides whether the coordinate is carried in scalar, vector, or dedicated target registers.
// %tid = kernel.workitem.id<x> : index
LOOM_DEFINE_ISA(loom_kernel_workitem_id_isa, LOOM_OP_KERNEL_WORKITEM_ID)
LOOM_DEFINE_RESULT(loom_kernel_workitem_id_result, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_workitem_id_dimension, 0)
iree_status_t loom_kernel_workitem_id_build(
    loom_builder_t* builder,
    uint8_t dimension,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_KERNEL_WORKGROUP_ID: Read one coordinate of the current workgroup within the dispatch grid. The result is a logical index value; target lowering decides whether the coordinate is carried in scalar registers, ABI state, or target-specific builtin values.
// %bid = kernel.workgroup.id<x> : index
LOOM_DEFINE_ISA(loom_kernel_workgroup_id_isa, LOOM_OP_KERNEL_WORKGROUP_ID)
LOOM_DEFINE_RESULT(loom_kernel_workgroup_id_result, 0)
LOOM_DEFINE_ATTR_ENUM(loom_kernel_workgroup_id_dimension, 0)
iree_status_t loom_kernel_workgroup_id_build(
    loom_builder_t* builder,
    uint8_t dimension,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the kernel dialect.
const loom_op_vtable_t* const* loom_kernel_dialect_vtables(
    iree_host_size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_KERNEL_OPS_H_
