// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal implementation contracts for vector-to-scalar lowering.
//
// This is not a public API. It defines the shared contracts between the pass
// driver, lane core, and specialized lane families.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_INTERNAL_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_INTERNAL_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/pass.h"
#include "loom/transforms/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_vector_to_scalar_stat_e {
  // Number of vector ops lowered by this pass.
  LOOM_VECTOR_TO_SCALAR_STAT_OPS_LOWERED = 0,
  // Number of scf.for loops created by this pass.
  LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED = 1,
  // Number of scalar lane programs materialized by this pass.
  LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED = 2,
} loom_vector_to_scalar_stat_t;

typedef enum loom_vector_to_scalar_lane_kind_e {
  LOOM_VECTOR_TO_SCALAR_LANE_GENERIC = 0,
  LOOM_VECTOR_TO_SCALAR_LANE_IOTA = 1,
  LOOM_VECTOR_TO_SCALAR_LANE_MASK_RANGE = 2,
  LOOM_VECTOR_TO_SCALAR_LANE_BROADCAST = 3,
  LOOM_VECTOR_TO_SCALAR_LANE_EXTRACT = 4,
  LOOM_VECTOR_TO_SCALAR_LANE_INSERT = 5,
  LOOM_VECTOR_TO_SCALAR_LANE_SLICE = 6,
  LOOM_VECTOR_TO_SCALAR_LANE_CONCAT = 7,
  LOOM_VECTOR_TO_SCALAR_LANE_TRANSPOSE = 8,
  LOOM_VECTOR_TO_SCALAR_LANE_SHUFFLE = 9,
  LOOM_VECTOR_TO_SCALAR_LANE_INTERLEAVE = 10,
  LOOM_VECTOR_TO_SCALAR_LANE_DEINTERLEAVE = 11,
  LOOM_VECTOR_TO_SCALAR_LANE_BITCAST = 12,
  LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTU = 13,
  LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTS = 14,
  LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_INSERT = 15,
  LOOM_VECTOR_TO_SCALAR_LANE_DOT4I = 16,
  LOOM_VECTOR_TO_SCALAR_LANE_BITPACK = 17,
  LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKU = 18,
  LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKS = 19,
  LOOM_VECTOR_TO_SCALAR_LANE_TABLE_LOOKUP = 20,
  LOOM_VECTOR_TO_SCALAR_LANE_TABLE_QUANTIZE = 21,
  LOOM_VECTOR_TO_SCALAR_LANE_TRANSFORM = 22,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD = 23,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD_MASK = 24,
  LOOM_VECTOR_TO_SCALAR_LANE_GATHER = 25,
  LOOM_VECTOR_TO_SCALAR_LANE_GATHER_MASK = 26,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD_EXPAND = 27,
} loom_vector_to_scalar_lane_kind_t;

typedef struct loom_vector_to_scalar_descriptor_t {
  // Vector op kind matched by this descriptor.
  loom_op_kind_t vector_kind;
  // Op kind emitted per lane for generic mechanical lowering.
  loom_op_kind_t lane_op_kind;
  // Lane program family.
  loom_vector_to_scalar_lane_kind_t lane_kind;
  // Number of vector operands consumed as lane inputs.
  uint8_t lane_operand_count;
  // Number of leading attrs copied from the vector op to the scalar op.
  uint8_t copied_attr_count;
  // Whether op->instance_flags may be forwarded to the scalar op.
  bool forward_instance_flags;
  // Whether non-zero op->instance_flags must be rejected.
  bool reject_instance_flags;
  // Whether the scalar result type is i1 instead of the vector element type.
  bool result_is_i1;
  // Operand that can seed a dynamic aggregate loop, or UINT8_MAX.
  uint8_t seed_operand_index;
} loom_vector_to_scalar_descriptor_t;

typedef struct loom_vector_to_scalar_state_t {
  // Current pass instance owning statistics and transient arena state.
  loom_pass_t* pass;
  // Rewriter used to insert replacement lane IR and maintain use-def state.
  loom_rewriter_t* rewriter;
  // Vector op currently being scalarized.
  loom_op_t* op;
  // Descriptor that selects the lane-program family for |op|.
  const loom_vector_to_scalar_descriptor_t* descriptor;
  // Rewriter value checkpoint used to preserve result names on new values.
  loom_value_id_t value_checkpoint;
  // Result ordinal being scalarized for multi-result vector ops.
  uint16_t result_ordinal;
  // Vector result type currently being expanded into scalar lanes.
  loom_type_t vector_type;
  // Scalar type produced by each lane program.
  loom_type_t result_scalar_type;
  // Source location assigned to replacement ops.
  loom_location_id_t location;
} loom_vector_to_scalar_state_t;

typedef struct loom_vector_to_scalar_index_list_t {
  // Dynamic index SSA values, or NULL when all lane indices are static.
  const loom_value_id_t* dynamic_indices;
  // Static index values, or NULL when any lane index is dynamic.
  const int64_t* static_indices;
  // Number of logical vector axes addressed by this index list.
  uint8_t rank;
} loom_vector_to_scalar_index_list_t;

typedef struct loom_vector_to_scalar_index_term_t {
  // SSA index value when |is_dynamic| is true.
  loom_value_id_t dynamic_value;
  // Static index value when |is_dynamic| is false.
  int64_t static_value;
  // Whether this term is represented by |dynamic_value| instead of a constant.
  bool is_dynamic;
} loom_vector_to_scalar_index_term_t;

const loom_vector_to_scalar_descriptor_t* loom_vector_to_scalar_find_descriptor(
    loom_op_kind_t kind);

bool loom_vector_to_scalar_indices_are_dynamic(
    loom_vector_to_scalar_index_list_t indices);

// Copies static index attribute storage into the builder arena so constructed
// ops do not retain references to pass scratch memory.
iree_status_t loom_vector_to_scalar_copy_static_indices(
    loom_builder_t* builder, const int64_t* indices,
    iree_host_size_t index_count, int64_t** out_indices);

loom_type_t loom_vector_to_scalar_lane_type(loom_type_t vector_type);

iree_status_t loom_vector_to_scalar_build_scalar_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, int64_t integer_value,
    loom_value_id_t* out_value_id);

// Builds a scalar or index constant using |value| exactly as the constant
// attribute, preserving non-integer floating-point constants such as transform
// normalization scales.
iree_status_t loom_vector_to_scalar_build_scalar_attr_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, loom_attribute_t value,
    loom_value_id_t* out_value_id);

iree_status_t loom_vector_to_scalar_build_vector_zero(
    loom_vector_to_scalar_state_t* state, loom_type_t result_type,
    loom_value_id_t* out_value_id);

iree_status_t loom_vector_to_scalar_build_generic_lane_op(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    uint8_t instance_flags, const loom_value_id_t* operands,
    uint16_t operand_count, const loom_attribute_t* attrs, uint8_t attr_count,
    loom_type_t result_type, loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_scalar_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_value_id_t* out_result);

int64_t loom_vector_to_scalar_integer_mask_value(int32_t bit_width,
                                                 int64_t used_bits);

int64_t loom_vector_to_scalar_shifted_integer_mask_value(int32_t bit_width,
                                                         int64_t offset,
                                                         int64_t used_bits);

iree_status_t loom_vector_to_scalar_build_integer_mask(
    loom_vector_to_scalar_state_t* state, loom_type_t type, int64_t used_bits,
    loom_value_id_t* out_mask);

iree_status_t loom_vector_to_scalar_cast_integer_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type, bool signed_extend,
    loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_scalar_shift(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type, int64_t amount,
    loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_index_term_as_scalar(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_type_t result_type,
    loom_value_id_t* out_value);

iree_status_t loom_vector_to_scalar_build_scalar_shift_term(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type,
    loom_vector_to_scalar_index_term_t amount, loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_bitstream_base_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t lane_ordinal, int64_t lane_bit_width,
    loom_vector_to_scalar_index_term_t* out_position);

iree_status_t loom_vector_to_scalar_build_single_bit_extract(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_type_t lane_type, loom_vector_to_scalar_index_term_t bit_shift,
    loom_value_id_t one_mask, loom_value_id_t* out_bit);

iree_status_t loom_vector_to_scalar_checked_static_bit_position(
    int64_t ordinal, int64_t bit_width, int64_t* out_position);

iree_status_t loom_vector_to_scalar_checked_static_bit_end(int64_t start,
                                                           int64_t bit_width,
                                                           int64_t* out_end);

iree_status_t loom_vector_to_scalar_build_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_materialize_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_materialize_linear_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t vector_value,
    loom_type_t vector_type, loom_vector_to_scalar_index_term_t ordinal,
    loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_insert_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_value_id_t aggregate, loom_type_t aggregate_type,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_aggregate);

iree_status_t loom_vector_to_scalar_dim_bound(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type, uint8_t axis,
    loom_value_id_t* out_bound);

iree_status_t loom_vector_to_scalar_build_index_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_result);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_static_term(
    int64_t value);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dynamic_term(
    loom_value_id_t value);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_lane_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, uint8_t axis);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_value_term(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value);

iree_status_t loom_vector_to_scalar_build_term_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs,
    loom_vector_to_scalar_index_term_t* out_term);

bool loom_vector_to_scalar_terms_equal_static(
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, bool* out_equal);

iree_status_t loom_vector_to_scalar_build_index_term_cmp(
    loom_vector_to_scalar_state_t* state, uint32_t predicate,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, loom_value_id_t* out_condition);

iree_status_t loom_vector_to_scalar_build_i1_and(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_terms_to_index_list(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_index_term_t* terms, uint8_t rank,
    loom_vector_to_scalar_index_list_t* out_indices);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dim_bound_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    uint8_t axis);

iree_status_t loom_vector_to_scalar_linear_ordinal_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    loom_vector_to_scalar_index_list_t indices,
    loom_vector_to_scalar_index_term_t* out_ordinal);

iree_status_t loom_vector_to_scalar_term_value(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_value_id_t* out_value);

iree_status_t loom_vector_to_scalar_ordinal_for_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_type_t result_type,
    loom_value_id_t* out_ordinal);

int64_t loom_vector_to_scalar_linear_ordinal_static(loom_type_t vector_type,
                                                    const int64_t* indices);

void loom_vector_to_scalar_indices_from_ordinal(loom_type_t vector_type,
                                                iree_host_size_t ordinal,
                                                int64_t* indices);

iree_status_t loom_vector_to_scalar_build_select_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t condition,
    loom_value_id_t true_lane, loom_value_id_t false_lane,
    loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_broadcast_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_extract_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_insert_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_slice_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_concat_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_transpose_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_shuffle_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_interleave_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_deinterleave_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitcast_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitfield_extract_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_extract,
    loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitfield_insert_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_dot4i_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitunpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_unpack,
    loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_table_lookup_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_table_quantize_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_transform_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_masked_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_gather_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_masked_gather_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_load_expand_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_lower_memory_store(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_lower_memory_store_compress(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_lower_memory_atomic_reduce(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_lower_memory_atomic_rmw(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_terms_from_explicit_indices(
    loom_vector_to_scalar_state_t* state, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices,
    loom_vector_to_scalar_index_term_t** out_terms, uint8_t* out_count);

iree_status_t loom_vector_to_scalar_try_materialize_def_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_type_t vector_type, loom_vector_to_scalar_index_list_t indices,
    bool* out_materialized, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_constant_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_poison_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_lower_splat(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_reduce(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_dotf(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_aggregate(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_INTERNAL_H_
