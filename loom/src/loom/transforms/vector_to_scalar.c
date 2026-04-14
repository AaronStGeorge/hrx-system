// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector_to_scalar.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/rewriter.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_VECTOR_TO_SCALAR_STAT_OPS_LOWERED = 0,
  LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED = 1,
  LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED = 2,
};

static const loom_pass_statistic_def_t kVectorToScalarStatistics[] = {
    {IREE_SVL("ops-lowered"), IREE_SVL("Number of vector ops lowered.")},
    {IREE_SVL("loops-created"), IREE_SVL("Number of scf.for loops created.")},
    {IREE_SVL("lanes-materialized"),
     IREE_SVL("Number of scalar lane programs materialized.")},
};

static const loom_pass_info_t loom_vector_to_scalar_pass_info_storage = {
    .name = IREE_SVL("vector-to-scalar"),
    .description =
        IREE_SVL("Expose vector lane semantics as scalar ops and scf loops."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kVectorToScalarStatistics,
    .statistic_count = IREE_ARRAYSIZE(kVectorToScalarStatistics),
};

const loom_pass_info_t* loom_vector_to_scalar_pass_info(void) {
  return &loom_vector_to_scalar_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Lane programs
//===----------------------------------------------------------------------===//

typedef enum loom_vector_to_scalar_lane_kind_e {
  LOOM_VECTOR_TO_SCALAR_LANE_GENERIC = 0,
  LOOM_VECTOR_TO_SCALAR_LANE_IOTA = 1,
  LOOM_VECTOR_TO_SCALAR_LANE_MASK_RANGE = 2,
} loom_vector_to_scalar_lane_kind_t;

typedef struct loom_vector_to_scalar_descriptor_t {
  // Vector op kind matched by this descriptor.
  loom_op_kind_t vector_kind;
  // Scalar op kind emitted per lane for generic mechanical lowering.
  loom_op_kind_t scalar_kind;
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

static const loom_vector_to_scalar_descriptor_t kVectorToScalarDescriptors[] = {
    {LOOM_OP_VECTOR_SELECT, LOOM_OP_SCALAR_SELECT,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 3, 0, false, false, false, 1},
    {LOOM_OP_VECTOR_CMPI, LOOM_OP_SCALAR_CMPI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 1, false, false, true, UINT8_MAX},
    {LOOM_OP_VECTOR_CMPF, LOOM_OP_SCALAR_CMPF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 1, false, false, true, UINT8_MAX},
    {LOOM_OP_VECTOR_ADDF, LOOM_OP_SCALAR_ADDF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_SUBF, LOOM_OP_SCALAR_SUBF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_MULF, LOOM_OP_SCALAR_MULF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_DIVF, LOOM_OP_SCALAR_DIVF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_REMF, LOOM_OP_SCALAR_REMF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_NEGF, LOOM_OP_SCALAR_NEGF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ABSF, LOOM_OP_SCALAR_ABSF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_MINIMUMF, LOOM_OP_SCALAR_MINIMUMF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_MAXIMUMF, LOOM_OP_SCALAR_MAXIMUMF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_MINNUMF, LOOM_OP_SCALAR_MINNUMF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_MAXNUMF, LOOM_OP_SCALAR_MAXNUMF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_COPYSIGNF, LOOM_OP_SCALAR_COPYSIGNF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_FMAF, LOOM_OP_SCALAR_FMAF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 3, 0, false, true, false, 2},
    {LOOM_OP_VECTOR_ADDI, LOOM_OP_SCALAR_ADDI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, true, false, false, 0},
    {LOOM_OP_VECTOR_MULI, LOOM_OP_SCALAR_MULI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, true, false, false, 0},
    {LOOM_OP_VECTOR_ANDI, LOOM_OP_SCALAR_ANDI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, false, false, 0},
    {LOOM_OP_VECTOR_ORI, LOOM_OP_SCALAR_ORI, LOOM_VECTOR_TO_SCALAR_LANE_GENERIC,
     2, 0, false, false, false, 0},
    {LOOM_OP_VECTOR_XORI, LOOM_OP_SCALAR_XORI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, false, false, 0},
    {LOOM_OP_VECTOR_CTPOPI, LOOM_OP_SCALAR_CTPOPI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_EXPF, LOOM_OP_SCALAR_EXPF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_EXP2F, LOOM_OP_SCALAR_EXP2F,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_EXPM1F, LOOM_OP_SCALAR_EXPM1F,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_LOGF, LOOM_OP_SCALAR_LOGF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_LOG2F, LOOM_OP_SCALAR_LOG2F,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_LOG10F, LOOM_OP_SCALAR_LOG10F,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_LOG1PF, LOOM_OP_SCALAR_LOG1PF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_POWF, LOOM_OP_SCALAR_POWF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_SQRTF, LOOM_OP_SCALAR_SQRTF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_RSQRTF, LOOM_OP_SCALAR_RSQRTF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_CBRTF, LOOM_OP_SCALAR_CBRTF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_SINF, LOOM_OP_SCALAR_SINF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_COSF, LOOM_OP_SCALAR_COSF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_TANF, LOOM_OP_SCALAR_TANF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ASINF, LOOM_OP_SCALAR_ASINF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ACOSF, LOOM_OP_SCALAR_ACOSF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ATANF, LOOM_OP_SCALAR_ATANF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ATAN2F, LOOM_OP_SCALAR_ATAN2F,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 2, 0, false, true, false, 0},
    {LOOM_OP_VECTOR_SINHF, LOOM_OP_SCALAR_SINHF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_COSHF, LOOM_OP_SCALAR_COSHF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_TANHF, LOOM_OP_SCALAR_TANHF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ASINHF, LOOM_OP_SCALAR_ASINHF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ACOSHF, LOOM_OP_SCALAR_ACOSHF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ATANHF, LOOM_OP_SCALAR_ATANHF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ERFF, LOOM_OP_SCALAR_ERFF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ERFCF, LOOM_OP_SCALAR_ERFCF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_CEILF, LOOM_OP_SCALAR_CEILF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_FLOORF, LOOM_OP_SCALAR_FLOORF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ROUNDF, LOOM_OP_SCALAR_ROUNDF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_ROUNDEVENF, LOOM_OP_SCALAR_ROUNDEVENF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_TRUNCF, LOOM_OP_SCALAR_TRUNCF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, true, false, UINT8_MAX},
    {LOOM_OP_VECTOR_EXTF, LOOM_OP_SCALAR_EXTF,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_FPTRUNC, LOOM_OP_SCALAR_FPTRUNC,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_EXTSI, LOOM_OP_SCALAR_EXTSI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_EXTUI, LOOM_OP_SCALAR_EXTUI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_TRUNCI, LOOM_OP_SCALAR_TRUNCI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_SITOFP, LOOM_OP_SCALAR_SITOFP,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_UITOFP, LOOM_OP_SCALAR_UITOFP,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_FPTOSI, LOOM_OP_SCALAR_FPTOSI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_FPTOUI, LOOM_OP_SCALAR_FPTOUI,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_IOTA, LOOM_OP_KIND_UNKNOWN, LOOM_VECTOR_TO_SCALAR_LANE_IOTA,
     0, 0, false, false, false, UINT8_MAX},
    {LOOM_OP_VECTOR_MASK_RANGE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_MASK_RANGE, 0, 0, false, false, true,
     UINT8_MAX},
};

typedef struct loom_vector_to_scalar_state_t {
  loom_pass_t* pass;
  loom_rewriter_t* rewriter;
  loom_op_t* op;
  const loom_vector_to_scalar_descriptor_t* descriptor;
  loom_value_id_t value_checkpoint;
  loom_type_t vector_type;
  loom_type_t result_scalar_type;
  loom_location_id_t location;
} loom_vector_to_scalar_state_t;

typedef struct loom_vector_to_scalar_index_list_t {
  const loom_value_id_t* dynamic_indices;
  const int64_t* static_indices;
  uint8_t rank;
} loom_vector_to_scalar_index_list_t;

static const loom_vector_to_scalar_descriptor_t*
loom_vector_to_scalar_find_descriptor(loom_op_kind_t kind) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVectorToScalarDescriptors);
       ++i) {
    if (kVectorToScalarDescriptors[i].vector_kind == kind) {
      return &kVectorToScalarDescriptors[i];
    }
  }
  return NULL;
}

static bool loom_vector_to_scalar_indices_are_dynamic(
    loom_vector_to_scalar_index_list_t indices) {
  return indices.dynamic_indices != NULL;
}

static iree_status_t loom_vector_to_scalar_build_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

static iree_status_t loom_vector_to_scalar_copy_static_indices(
    loom_builder_t* builder, const int64_t* indices, uint8_t rank,
    int64_t** out_indices) {
  *out_indices = NULL;
  if (rank == 0) return iree_ok_status();
  int64_t* copied = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, rank, sizeof(int64_t), (void**)&copied));
  memcpy(copied, indices, (iree_host_size_t)rank * sizeof(int64_t));
  *out_indices = copied;
  return iree_ok_status();
}

static loom_type_t loom_vector_to_scalar_lane_type(loom_type_t vector_type) {
  return loom_type_scalar(loom_type_element_type(vector_type));
}

static loom_attribute_t loom_vector_to_scalar_zero_attr(
    loom_scalar_type_t scalar_type) {
  if (scalar_type == LOOM_SCALAR_TYPE_I1) return loom_attr_bool(false);
  if (loom_scalar_type_is_float(scalar_type)) return loom_attr_f64(0.0);
  return loom_attr_i64(0);
}

static iree_status_t loom_vector_to_scalar_build_scalar_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, int64_t integer_value,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_attribute_t attr = loom_scalar_type_is_float(scalar_type)
                              ? loom_attr_f64((double)integer_value)
                              : loom_attr_i64(integer_value);
  loom_op_t* constant_op = NULL;
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, attr, result_type,
                                                   location, &constant_op));
    *out_value_id = loom_index_constant_result(constant_op);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, attr, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_vector_zero(
    loom_vector_to_scalar_state_t* state, loom_type_t result_type,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &state->rewriter->builder, loom_vector_to_scalar_zero_attr(scalar_type),
      result_type, state->location, &constant_op));
  *out_value_id = loom_vector_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_generic_scalar_op(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    uint8_t instance_flags, const loom_value_id_t* operands,
    uint16_t operand_count, const loom_attribute_t* attrs, uint8_t attr_count,
    loom_type_t result_type, loom_value_id_t* out_result) {
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(state->rewriter->module->context, kind);
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no vtable registered for scalar op kind %u",
                            (unsigned)kind);
  }
  if (operand_count != vtable->fixed_operand_count ||
      attr_count != vtable->attribute_count ||
      vtable->fixed_result_count != 1 || vtable->region_count != 0 ||
      (vtable->vtable_flags & (LOOM_OP_VTABLE_VARIADIC_OPERANDS |
                               LOOM_OP_VTABLE_VARIADIC_RESULTS)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scalar op %.*s is not a fixed one-result lane op",
                            (int)loom_op_vtable_name(vtable).size,
                            loom_op_vtable_name(vtable).data);
  }

  loom_builder_t* builder = &state->rewriter->builder;
  loom_op_t* scalar_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, kind, operand_count, 1,
                                                0, 0, attr_count,
                                                state->location, &scalar_op));
  scalar_op->instance_flags = instance_flags;
  loom_value_id_t* op_operands = loom_op_operands(scalar_op);
  for (uint16_t i = 0; i < operand_count; ++i) {
    op_operands[i] = operands[i];
  }
  IREE_RETURN_IF_ERROR(loom_builder_define_value(
      builder, result_type, &loom_op_results(scalar_op)[0]));
  loom_attribute_t* op_attrs = loom_op_attrs(scalar_op);
  for (uint8_t i = 0; i < attr_count; ++i) {
    op_attrs[i] = attrs[i];
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, scalar_op));
  *out_result = loom_op_results(scalar_op)[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_extract_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t vector_value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t vector_type =
      loom_module_value_type(state->rewriter->module, vector_value);
  if (!loom_type_is_vector(vector_type)) {
    *out_lane = vector_value;
    return iree_ok_status();
  }
  loom_type_t lane_type = loom_vector_to_scalar_lane_type(vector_type);
  int64_t* static_indices = NULL;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->builder.arena, indices.rank,
                                  sizeof(int64_t), (void**)&static_indices));
    for (uint8_t i = 0; i < indices.rank; ++i) {
      static_indices[i] = INT64_MIN;
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_copy_static_indices(
        &state->rewriter->builder, indices.static_indices, indices.rank,
        &static_indices));
  }
  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &state->rewriter->builder, vector_value, indices.dynamic_indices,
      loom_vector_to_scalar_indices_are_dynamic(indices) ? indices.rank : 0,
      static_indices, indices.rank, lane_type, state->location, &extract_op));
  *out_lane = loom_vector_extract_result(extract_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_insert_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_value_id_t aggregate, loom_type_t aggregate_type,
    loom_vector_to_scalar_index_list_t indices,
    loom_value_id_t* out_aggregate) {
  int64_t* static_indices = NULL;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->builder.arena, indices.rank,
                                  sizeof(int64_t), (void**)&static_indices));
    for (uint8_t i = 0; i < indices.rank; ++i) {
      static_indices[i] = INT64_MIN;
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_copy_static_indices(
        &state->rewriter->builder, indices.static_indices, indices.rank,
        &static_indices));
  }
  loom_op_t* insert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_insert_build(
      &state->rewriter->builder, lane, aggregate, indices.dynamic_indices,
      loom_vector_to_scalar_indices_are_dynamic(indices) ? indices.rank : 0,
      static_indices, indices.rank, aggregate_type, state->location,
      &insert_op));
  *out_aggregate = loom_vector_insert_result(insert_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_dim_bound(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type, uint8_t axis,
    loom_value_id_t* out_bound) {
  if (loom_type_dim_is_dynamic_at(vector_type, axis)) {
    *out_bound = loom_type_dim_value_id_at(vector_type, axis);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, (int64_t)loom_type_dim_static_size_at(vector_type, axis),
      out_bound);
}

static iree_status_t loom_vector_to_scalar_build_index_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_result) {
  switch (kind) {
    case LOOM_OP_INDEX_ADD: {
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          &state->rewriter->builder, lhs, rhs,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &op));
      *out_result = loom_index_add_result(op);
      return iree_ok_status();
    }
    case LOOM_OP_INDEX_MUL: {
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_mul_build(
          &state->rewriter->builder, lhs, rhs,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &op));
      *out_result = loom_index_mul_result(op);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported index binary op kind %u",
                              (unsigned)kind);
  }
}

static iree_status_t loom_vector_to_scalar_linear_ordinal_dynamic(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_ordinal) {
  if (indices.rank == 1) {
    *out_ordinal = indices.dynamic_indices[0];
    return iree_ok_status();
  }

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &ordinal));
  for (uint8_t axis = 0; axis < indices.rank; ++axis) {
    if (axis > 0) {
      loom_value_id_t dim = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
          state, state->vector_type, axis, &dim));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
          state, LOOM_OP_INDEX_MUL, ordinal, dim, &ordinal));
    }
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
        state, LOOM_OP_INDEX_ADD, ordinal, indices.dynamic_indices[axis],
        &ordinal));
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static int64_t loom_vector_to_scalar_linear_ordinal_static(
    loom_type_t vector_type, const int64_t* indices) {
  int64_t ordinal = 0;
  uint8_t rank = loom_type_rank(vector_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (axis > 0) {
      ordinal *= (int64_t)loom_type_dim_static_size_at(vector_type, axis);
    }
    ordinal += indices[axis];
  }
  return ordinal;
}

static iree_status_t loom_vector_to_scalar_materialize_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

static iree_status_t loom_vector_to_scalar_cast_index_to_scalar(
    loom_vector_to_scalar_state_t* state, loom_value_id_t index_value,
    loom_type_t result_type, loom_value_id_t* out_value) {
  if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_INDEX) {
    *out_value = index_value;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(&state->rewriter->builder, index_value,
                            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                            result_type, state->location, &cast_op));
  *out_value = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_ordinal_for_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_type_t result_type,
    loom_value_id_t* out_ordinal) {
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    loom_value_id_t index_ordinal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_dynamic(
        state, indices, &index_ordinal));
    return loom_vector_to_scalar_cast_index_to_scalar(state, index_ordinal,
                                                      result_type, out_ordinal);
  }
  int64_t ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, indices.static_indices);
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, result_type, state->location, ordinal,
      out_ordinal);
}

static iree_status_t loom_vector_to_scalar_build_coordinate_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t integer_kind,
    loom_op_kind_t index_kind, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t type, loom_value_id_t* out_result) {
  if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX) {
    return loom_vector_to_scalar_build_index_binary(state, index_kind, lhs, rhs,
                                                    out_result);
  }
  return loom_vector_to_scalar_build_generic_scalar_op(
      state, integer_kind, 0, (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0, type,
      out_result);
}

static iree_status_t loom_vector_to_scalar_build_iota_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t base = loom_vector_iota_base(state->op);
  loom_value_id_t step = loom_vector_iota_step(state->op);
  loom_type_t lane_type = state->result_scalar_type;
  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_ordinal_for_lane(
      state, indices, lane_type, &ordinal));
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_MULI, LOOM_OP_INDEX_MUL, ordinal, step, lane_type,
      &scaled));
  return loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_ADDI, LOOM_OP_INDEX_ADD, base, scaled, lane_type,
      out_lane);
}

static iree_status_t loom_vector_to_scalar_build_mask_range_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t lower_bound = loom_vector_mask_range_lower_bound(state->op);
  loom_value_id_t upper_bound = loom_vector_mask_range_upper_bound(state->op);
  loom_value_id_t step = loom_vector_mask_range_step(state->op);
  loom_type_t coordinate_type =
      loom_module_value_type(state->rewriter->module, lower_bound);

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_ordinal_for_lane(
      state, indices, coordinate_type, &ordinal));
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_MULI, LOOM_OP_INDEX_MUL, ordinal, step,
      coordinate_type, &scaled));
  loom_value_id_t coordinate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_ADDI, LOOM_OP_INDEX_ADD, lower_bound, scaled,
      coordinate_type, &coordinate));

  if (loom_type_element_type(coordinate_type) == LOOM_SCALAR_TYPE_INDEX) {
    loom_op_t* cmp_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        &state->rewriter->builder, LOOM_INDEX_CMP_PREDICATE_SLT, coordinate,
        upper_bound, coordinate_type, loom_type_scalar(LOOM_SCALAR_TYPE_I1),
        state->location, &cmp_op));
    *out_lane = loom_index_cmp_result(cmp_op);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_generic_scalar_op(
      state, LOOM_OP_SCALAR_CMPI, 0,
      (loom_value_id_t[]){coordinate, upper_bound}, 2,
      (loom_attribute_t[]){loom_attr_enum(LOOM_SCALAR_CMPI_PREDICATE_SLT)}, 1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), out_lane);
}

static iree_status_t loom_vector_to_scalar_build_generic_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  const loom_vector_to_scalar_descriptor_t* descriptor = state->descriptor;
  if (descriptor->reject_instance_flags && state->op->instance_flags != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vector-to-scalar cannot lower %.*s with instance flags because vector "
        "value-domain assumptions are not scalar fast-math flags",
        (int)loom_op_name(state->rewriter->module, state->op).size,
        loom_op_name(state->rewriter->module, state->op).data);
  }

  loom_value_id_t lane_operands[4] = {0};
  const loom_value_id_t* operands = loom_op_const_operands(state->op);
  for (uint8_t i = 0; i < descriptor->lane_operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, operands[i], indices, &lane_operands[i]));
  }

  uint8_t instance_flags =
      descriptor->forward_instance_flags ? state->op->instance_flags : 0;
  const loom_attribute_t* attrs = loom_op_attrs(state->op);
  return loom_vector_to_scalar_build_generic_scalar_op(
      state, descriptor->scalar_kind, instance_flags, lane_operands,
      descriptor->lane_operand_count, attrs, descriptor->copied_attr_count,
      state->result_scalar_type, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
  }
  switch (state->descriptor->lane_kind) {
    case LOOM_VECTOR_TO_SCALAR_LANE_GENERIC:
      return loom_vector_to_scalar_build_generic_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_IOTA:
      return loom_vector_to_scalar_build_iota_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_MASK_RANGE:
      return loom_vector_to_scalar_build_mask_range_lane(state, indices,
                                                         out_lane);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown vector-to-scalar lane kind %u",
                              (unsigned)state->descriptor->lane_kind);
  }
}

static bool loom_vector_to_scalar_try_from_elements_lane(
    loom_vector_to_scalar_state_t* state, loom_op_t* def_op,
    loom_type_t vector_type, loom_vector_to_scalar_index_list_t indices,
    loom_value_id_t* out_lane) {
  if (!loom_vector_from_elements_isa(def_op)) return false;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) return false;
  if (!loom_type_is_all_static(vector_type)) return false;
  int64_t ordinal = loom_vector_to_scalar_linear_ordinal_static(
      vector_type, indices.static_indices);
  if (ordinal < 0) return false;
  loom_value_slice_t elements = loom_vector_from_elements_elements(def_op);
  if ((uint64_t)ordinal >= elements.count) return false;
  (void)state;
  *out_lane = loom_value_slice_get(elements, (uint16_t)ordinal);
  return true;
}

static iree_status_t loom_vector_to_scalar_try_materialize_def_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_type_t vector_type, loom_vector_to_scalar_index_list_t indices,
    bool* out_materialized, loom_value_id_t* out_lane) {
  *out_materialized = false;
  loom_op_t* def_op =
      loom_value_def_op(loom_module_value(state->rewriter->module, value));
  if (!def_op) return iree_ok_status();
  if (loom_vector_to_scalar_try_from_elements_lane(state, def_op, vector_type,
                                                   indices, out_lane)) {
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_splat_isa(def_op)) {
    *out_lane = loom_vector_splat_scalar(def_op);
    *out_materialized = true;
    return iree_ok_status();
  }

  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(def_op->kind);
  if (!descriptor) return iree_ok_status();
  loom_type_t result_type = loom_module_value_type(state->rewriter->module,
                                                   loom_op_results(def_op)[0]);
  loom_vector_to_scalar_state_t def_state = {
      .pass = state->pass,
      .rewriter = state->rewriter,
      .op = def_op,
      .descriptor = descriptor,
      .value_checkpoint = state->value_checkpoint,
      .vector_type = result_type,
      .result_scalar_type = descriptor->result_is_i1
                                ? loom_type_scalar(LOOM_SCALAR_TYPE_I1)
                                : loom_vector_to_scalar_lane_type(result_type),
      .location = def_op->location,
  };
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_lane(&def_state, indices, out_lane));
  *out_materialized = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_materialize_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t type = loom_module_value_type(state->rewriter->module, value);
  if (!loom_type_is_vector(type)) {
    *out_lane = value;
    return iree_ok_status();
  }
  bool materialized = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_try_materialize_def_lane(
      state, value, type, indices, &materialized, out_lane));
  if (materialized) return iree_ok_status();
  return loom_vector_to_scalar_extract_lane(state, value, indices, out_lane);
}

//===----------------------------------------------------------------------===//
// Static aggregate lowering
//===----------------------------------------------------------------------===//

static void loom_vector_to_scalar_indices_from_ordinal(loom_type_t vector_type,
                                                       iree_host_size_t ordinal,
                                                       int64_t* indices) {
  uint8_t rank = loom_type_rank(vector_type);
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    uint8_t axis = (uint8_t)(rank - reverse_axis - 1);
    uint64_t dim = loom_type_dim_static_size_at(vector_type, axis);
    indices[axis] = dim == 0 ? 0 : (int64_t)(ordinal % dim);
    if (dim != 0) ordinal /= dim;
  }
}

static iree_status_t loom_vector_to_scalar_lower_static_aggregate(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  iree_host_size_t element_count = 0;
  if (!loom_type_static_element_count(state->vector_type, &element_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all-static vector result type");
  }

  loom_builder_t* builder = &state->rewriter->builder;
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  uint8_t rank = loom_type_rank(state->vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, rank, sizeof(int64_t), (void**)&indices));
  }
  for (iree_host_size_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->vector_type, ordinal,
                                               indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_lane(state, index_list,
                                                          &elements[ordinal]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      builder, elements, element_count, state->vector_type, state->location,
      &from_elements_op));
  *out_replacement = loom_vector_from_elements_result(from_elements_op);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dynamic aggregate lowering
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_aggregate_loop_axis(
    loom_vector_to_scalar_state_t* state, uint8_t axis,
    loom_value_id_t current_aggregate, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      state, state->vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_for_build(&state->rewriter->builder, lower_bound, upper_bound,
                         step, &current_aggregate, 1, &state->vector_type, 1,
                         NULL, 0, state->location, &loop));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->vector_type),
    };
    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_lane(state, index_list, &lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_insert_lane(
        state, lane, aggregate_arg, state->vector_type, index_list,
        &yielded_aggregate));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_aggregate_loop_axis(
        state, (uint8_t)(axis + 1), aggregate_arg, dynamic_indices,
        &yielded_aggregate));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_dynamic_seed(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_seed) {
  if (state->descriptor->seed_operand_index != UINT8_MAX &&
      state->descriptor->seed_operand_index < state->op->operand_count) {
    loom_value_id_t seed = loom_op_const_operands(
        state->op)[state->descriptor->seed_operand_index];
    loom_type_t seed_type =
        loom_module_value_type(state->rewriter->module, seed);
    if (loom_type_equal(seed_type, state->vector_type)) {
      *out_seed = seed;
      return iree_ok_status();
    }
  }
  return loom_vector_to_scalar_build_vector_zero(state, state->vector_type,
                                                 out_seed);
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_aggregate(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_value_id_t seed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dynamic_seed(state, &seed));

  uint8_t rank = loom_type_rank(state->vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&dynamic_indices));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_aggregate_loop_axis(
      state, 0, seed, dynamic_indices, out_replacement));
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_aggregate(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  if (loom_type_is_all_static(state->vector_type)) {
    return loom_vector_to_scalar_lower_static_aggregate(state, out_replacement);
  }
  return loom_vector_to_scalar_lower_dynamic_aggregate(state, out_replacement);
}

//===----------------------------------------------------------------------===//
// Splat lowering
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_lower_static_splat(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  iree_host_size_t element_count = 0;
  if (!loom_type_static_element_count(state->vector_type, &element_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all-static vector result type");
  }
  loom_value_id_t scalar = loom_vector_splat_scalar(state->op);
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    elements[i] = scalar;
  }
  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, element_count, state->vector_type,
      state->location, &from_elements_op));
  *out_replacement = loom_vector_from_elements_result(from_elements_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_splat_loop_axis(
    loom_vector_to_scalar_state_t* state, uint8_t axis,
    loom_value_id_t current_aggregate, loom_value_id_t scalar,
    loom_value_id_t* dynamic_indices, loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      state, state->vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_for_build(&state->rewriter->builder, lower_bound, upper_bound,
                         step, &current_aggregate, 1, &state->vector_type, 1,
                         NULL, 0, state->location, &loop));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  if (axis + 1 == loom_type_rank(state->vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->vector_type),
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_insert_lane(
        state, scalar, aggregate_arg, state->vector_type, index_list,
        &yielded_aggregate));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_splat_loop_axis(
        state, (uint8_t)(axis + 1), aggregate_arg, scalar, dynamic_indices,
        &yielded_aggregate));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_splat(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_value_id_t seed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_vector_zero(
      state, state->vector_type, &seed));

  uint8_t rank = loom_type_rank(state->vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&dynamic_indices));
  return loom_vector_to_scalar_splat_loop_axis(
      state, 0, seed, loom_vector_splat_scalar(state->op), dynamic_indices,
      out_replacement);
}

static iree_status_t loom_vector_to_scalar_lower_splat(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  if (loom_type_is_all_static(state->vector_type)) {
    return loom_vector_to_scalar_lower_static_splat(state, out_replacement);
  }
  return loom_vector_to_scalar_lower_dynamic_splat(state, out_replacement);
}

//===----------------------------------------------------------------------===//
// Reduction lowering
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_reduce_scalar_kind(
    uint8_t reduce_kind, loom_op_kind_t* out_scalar_kind) {
  switch ((loom_vector_reduce_kind_t)reduce_kind) {
    case LOOM_VECTOR_REDUCE_KIND_ADDI:
      *out_scalar_kind = LOOM_OP_SCALAR_ADDI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ADDF:
      *out_scalar_kind = LOOM_OP_SCALAR_ADDF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MULI:
      *out_scalar_kind = LOOM_OP_SCALAR_MULI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MULF:
      *out_scalar_kind = LOOM_OP_SCALAR_MULF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINSI:
      *out_scalar_kind = LOOM_OP_SCALAR_MINSI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXSI:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXSI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINUI:
      *out_scalar_kind = LOOM_OP_SCALAR_MINUI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXUI:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXUI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ANDI:
      *out_scalar_kind = LOOM_OP_SCALAR_ANDI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ORI:
      *out_scalar_kind = LOOM_OP_SCALAR_ORI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_XORI:
      *out_scalar_kind = LOOM_OP_SCALAR_XORI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINIMUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MINIMUMF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXIMUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXIMUMF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINNUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MINNUMF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXNUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXNUMF;
      return true;
    default:
      return false;
  }
}

typedef struct loom_vector_to_scalar_accumulator_state_t {
  loom_vector_to_scalar_state_t lane_state;
  loom_value_id_t input;
  loom_value_id_t rhs;
  loom_value_id_t init;
  loom_op_kind_t scalar_kind;
  bool use_fmaf;
} loom_vector_to_scalar_accumulator_state_t;

static iree_status_t loom_vector_to_scalar_build_accumulator_lane(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t accumulator,
    loom_value_id_t* out_next) {
  loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      &state->lane_state, state->input, indices, &lhs_lane));
  if (state->use_fmaf) {
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        &state->lane_state, state->rhs, indices, &rhs_lane));
    return loom_vector_to_scalar_build_generic_scalar_op(
        &state->lane_state, LOOM_OP_SCALAR_FMAF, 0,
        (loom_value_id_t[]){lhs_lane, rhs_lane, accumulator}, 3, NULL, 0,
        state->lane_state.result_scalar_type, out_next);
  }
  return loom_vector_to_scalar_build_generic_scalar_op(
      &state->lane_state, state->scalar_kind, 0,
      (loom_value_id_t[]){accumulator, lhs_lane}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_next);
}

static iree_status_t loom_vector_to_scalar_lower_static_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  iree_host_size_t element_count = 0;
  if (!loom_type_static_element_count(state->lane_state.vector_type,
                                      &element_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all-static vector input type");
  }
  loom_value_id_t accumulator = state->init;
  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->lane_state.rewriter->arena, rank,
                                  sizeof(int64_t), (void**)&indices));
  }
  for (iree_host_size_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->lane_state.vector_type,
                                               ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_accumulator_lane(
        state, index_list, accumulator, &accumulator));
    if (state->lane_state.pass->statistics) {
      loom_pass_statistic_add(state->lane_state.pass,
                              LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
    }
  }
  *out_replacement = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_accumulator_loop_axis(
    loom_vector_to_scalar_accumulator_state_t* state, uint8_t axis,
    loom_value_id_t current_accumulator, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_accumulator) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 0,
      &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 1,
      &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      &state->lane_state, state->lane_state.vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->lane_state.rewriter->builder, lower_bound, upper_bound, step,
      &current_accumulator, 1, &state->lane_state.result_scalar_type, 1, NULL,
      0, state->lane_state.location, &loop));
  if (state->lane_state.pass->statistics) {
    loom_pass_statistic_add(state->lane_state.pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->lane_state.rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_accumulator = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->lane_state.vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->lane_state.vector_type),
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_accumulator_lane(
        state, index_list, accumulator_arg, &yielded_accumulator));
    if (state->lane_state.pass->statistics) {
      loom_pass_statistic_add(state->lane_state.pass,
                              LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_accumulator_loop_axis(
        state, (uint8_t)(axis + 1), accumulator_arg, dynamic_indices,
        &yielded_accumulator));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->lane_state.rewriter->builder, &yielded_accumulator, 1,
      state->lane_state.location, &yield_op));
  loom_builder_restore(&state->lane_state.rewriter->builder, saved);

  *out_accumulator = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lane_state.rewriter->arena, rank, sizeof(loom_value_id_t),
      (void**)&dynamic_indices));
  return loom_vector_to_scalar_accumulator_loop_axis(
      state, 0, state->init, dynamic_indices, out_replacement);
}

static iree_status_t loom_vector_to_scalar_lower_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  if (loom_type_is_all_static(state->lane_state.vector_type)) {
    return loom_vector_to_scalar_lower_static_accumulator(state,
                                                          out_replacement);
  }
  return loom_vector_to_scalar_lower_dynamic_accumulator(state,
                                                         out_replacement);
}

static iree_status_t loom_vector_to_scalar_lower_reduce(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_op_kind_t scalar_kind = LOOM_OP_KIND_UNKNOWN;
  if (!loom_vector_to_scalar_reduce_scalar_kind(
          loom_vector_reduce_kind(state->op), &scalar_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported vector.reduce kind %u",
                            (unsigned)loom_vector_reduce_kind(state->op));
  }
  loom_vector_to_scalar_accumulator_state_t accumulator_state = {
      .lane_state = *state,
      .input = loom_vector_reduce_input(state->op),
      .init = loom_vector_reduce_init(state->op),
      .scalar_kind = scalar_kind,
  };
  return loom_vector_to_scalar_lower_accumulator(&accumulator_state,
                                                 out_replacement);
}

static iree_status_t loom_vector_to_scalar_lower_dotf(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_vector_to_scalar_accumulator_state_t accumulator_state = {
      .lane_state = *state,
      .input = loom_vector_dotf_lhs(state->op),
      .rhs = loom_vector_dotf_rhs(state->op),
      .init = loom_vector_dotf_init(state->op),
      .use_fmaf = true,
  };
  return loom_vector_to_scalar_lower_accumulator(&accumulator_state,
                                                 out_replacement);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_prepare_state(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_vector_to_scalar_descriptor_t* descriptor,
    loom_vector_to_scalar_state_t* out_state) {
  loom_value_id_t result = loom_op_results(op)[0];
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  if (!loom_type_is_vector(result_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected vector result for %.*s",
                            (int)loom_op_name(rewriter->module, op).size,
                            loom_op_name(rewriter->module, op).data);
  }
  loom_type_t scalar_type = descriptor && descriptor->result_is_i1
                                ? loom_type_scalar(LOOM_SCALAR_TYPE_I1)
                                : loom_vector_to_scalar_lane_type(result_type);
  *out_state = (loom_vector_to_scalar_state_t){
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .descriptor = descriptor,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = result_type,
      .result_scalar_type = scalar_type,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_replace_one_result(
    loom_vector_to_scalar_state_t* state, loom_value_id_t replacement) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, state->op, &replacement, 1, state->value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, state->op, &replacement, 1));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, LOOM_VECTOR_TO_SCALAR_STAT_OPS_LOWERED,
                            1);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_op(loom_pass_t* pass,
                                                    loom_rewriter_t* rewriter,
                                                    loom_op_t* op) {
  if (op->result_count != 1) return iree_ok_status();

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;

  if (loom_vector_splat_isa(op)) {
    loom_vector_to_scalar_state_t state = {0};
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_prepare_state(pass, rewriter, op, NULL, &state));
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_splat(&state, &replacement));
    return loom_vector_to_scalar_replace_one_result(&state, replacement);
  }

  if (loom_vector_reduce_isa(op)) {
    loom_vector_to_scalar_state_t state = {
        .pass = pass,
        .rewriter = rewriter,
        .op = op,
        .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
        .vector_type = loom_module_value_type(rewriter->module,
                                              loom_vector_reduce_input(op)),
        .result_scalar_type = loom_module_value_type(
            rewriter->module, loom_vector_reduce_init(op)),
        .location = op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_reduce(&state, &replacement));
    return loom_vector_to_scalar_replace_one_result(&state, replacement);
  }

  if (loom_vector_dotf_isa(op)) {
    loom_vector_to_scalar_state_t state = {
        .pass = pass,
        .rewriter = rewriter,
        .op = op,
        .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
        .vector_type =
            loom_module_value_type(rewriter->module, loom_vector_dotf_lhs(op)),
        .result_scalar_type =
            loom_module_value_type(rewriter->module, loom_vector_dotf_init(op)),
        .location = op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_dotf(&state, &replacement));
    return loom_vector_to_scalar_replace_one_result(&state, replacement);
  }

  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(op->kind);
  if (!descriptor) return iree_ok_status();

  loom_vector_to_scalar_state_t state = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_prepare_state(pass, rewriter, op,
                                                           descriptor, &state));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_aggregate(&state, &replacement));
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

iree_status_t loom_vector_to_scalar_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  iree_status_t status = loom_rewriter_seed_function(&rewriter, function);
  while (iree_status_is_ok(status)) {
    loom_op_t* op = loom_rewriter_pop(&rewriter);
    if (!op) break;
    bool erased = false;
    status = loom_rewriter_erase_if_dead(&rewriter, op, &erased);
    if (!iree_status_is_ok(status) || erased) continue;
    status = loom_vector_to_scalar_lower_op(pass, &rewriter, op);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
