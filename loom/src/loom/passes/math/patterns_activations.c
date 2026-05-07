// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/passes/math/patterns.h"
#include "loom/rewrite/rewriter.h"

typedef iree_status_t (*loom_math_legalize_constant_build_fn_t)(
    loom_builder_t* builder, loom_attribute_t value, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_unary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_binary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

typedef struct loom_math_legalize_lane_builders_t {
  // Builds a uniform scalar or vector constant in the source lane domain.
  loom_math_legalize_constant_build_fn_t constant;
  // Builds a lane-wise addition in the source lane domain.
  loom_math_legalize_binary_build_fn_t addf;
  // Builds a lane-wise multiplication in the source lane domain.
  loom_math_legalize_binary_build_fn_t mulf;
  // Builds a lane-wise division in the source lane domain.
  loom_math_legalize_binary_build_fn_t divf;
  // Builds a lane-wise base-2 exponential in the source lane domain.
  loom_math_legalize_unary_build_fn_t exp2f;
  // Builds a lane-wise base-2 logarithm in the source lane domain.
  loom_math_legalize_unary_build_fn_t log2f;
  // Builds a lane-wise hyperbolic tangent in the source lane domain.
  loom_math_legalize_unary_build_fn_t tanhf;
  // Builds a lane-wise logistic op in the source lane domain.
  loom_math_legalize_unary_build_fn_t logisticf;
} loom_math_legalize_lane_builders_t;

typedef struct loom_math_legalize_activation_source_t {
  // Source op operand that feeds the semantic activation.
  loom_value_id_t input;
  // Source result type preserved by the replacement expression.
  loom_type_t result_type;
  // Source fast-math flags forwarded to replacement floating-point ops.
  uint8_t fastmath_flags;
  // Source location copied to replacement ops.
  loom_location_id_t location;
  // Builder table for the source scalar/vector lane domain.
  const loom_math_legalize_lane_builders_t* lane_builders;
} loom_math_legalize_activation_source_t;

static const loom_math_legalize_lane_builders_t kScalarLaneBuilders = {
    .constant = loom_scalar_constant_build,
    .addf = loom_scalar_addf_build,
    .mulf = loom_scalar_mulf_build,
    .divf = loom_scalar_divf_build,
    .exp2f = loom_scalar_exp2f_build,
    .log2f = loom_scalar_log2f_build,
    .tanhf = loom_scalar_tanhf_build,
    .logisticf = loom_scalar_logisticf_build,
};

static const loom_math_legalize_lane_builders_t kVectorLaneBuilders = {
    .constant = loom_vector_constant_build,
    .addf = loom_vector_addf_build,
    .mulf = loom_vector_mulf_build,
    .divf = loom_vector_divf_build,
    .exp2f = loom_vector_exp2f_build,
    .log2f = loom_vector_log2f_build,
    .tanhf = loom_vector_tanhf_build,
    .logisticf = loom_vector_logisticf_build,
};

static iree_status_t loom_math_legalize_activation_source_initialize(
    const loom_math_legalize_recipe_context_t* context, const loom_op_t* op,
    loom_math_legalize_activation_source_t* out_source) {
  *out_source = (loom_math_legalize_activation_source_t){
      .result_type =
          loom_module_value_type(context->module, loom_op_results(op)[0]),
      .location = op->location,
  };
  switch (op->kind) {
    case LOOM_OP_SCALAR_EXPF:
      out_source->input = loom_scalar_expf_input(op);
      out_source->fastmath_flags = loom_scalar_expf_fastmath(op);
      out_source->lane_builders = &kScalarLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_SCALAR_LOGISTICF:
      out_source->input = loom_scalar_logisticf_input(op);
      out_source->fastmath_flags = loom_scalar_logisticf_fastmath(op);
      out_source->lane_builders = &kScalarLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SILUF:
      out_source->input = loom_scalar_siluf_input(op);
      out_source->fastmath_flags = loom_scalar_siluf_fastmath(op);
      out_source->lane_builders = &kScalarLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SOFTPLUSF:
      out_source->input = loom_scalar_softplusf_input(op);
      out_source->fastmath_flags = loom_scalar_softplusf_fastmath(op);
      out_source->lane_builders = &kScalarLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_SCALAR_GELUF:
      out_source->input = loom_scalar_geluf_input(op);
      out_source->fastmath_flags = loom_scalar_geluf_fastmath(op);
      out_source->lane_builders = &kScalarLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXPF:
      out_source->input = loom_vector_expf_input(op);
      out_source->fastmath_flags = loom_vector_expf_fastmath(op);
      out_source->lane_builders = &kVectorLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOGISTICF:
      out_source->input = loom_vector_logisticf_input(op);
      out_source->fastmath_flags = loom_vector_logisticf_fastmath(op);
      out_source->lane_builders = &kVectorLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SILUF:
      out_source->input = loom_vector_siluf_input(op);
      out_source->fastmath_flags = loom_vector_siluf_fastmath(op);
      out_source->lane_builders = &kVectorLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SOFTPLUSF:
      out_source->input = loom_vector_softplusf_input(op);
      out_source->fastmath_flags = loom_vector_softplusf_fastmath(op);
      out_source->lane_builders = &kVectorLaneBuilders;
      return iree_ok_status();
    case LOOM_OP_VECTOR_GELUF:
      out_source->input = loom_vector_geluf_input(op);
      out_source->fastmath_flags = loom_vector_geluf_fastmath(op);
      out_source->lane_builders = &kVectorLaneBuilders;
      return iree_ok_status();
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "math recipe row referenced unsupported op kind %u",
                          op->kind);
}

static double loom_math_legalize_gelu_logistic_scale(const loom_op_t* op) {
  return loom_attr_as_f64(loom_op_attrs(op)[1]);
}

static iree_status_t loom_math_legalize_build_constant(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source, double value,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(source->lane_builders->constant(
      builder, loom_attr_f64(value), source->result_type, source->location,
      &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_unary(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source,
    loom_math_legalize_unary_build_fn_t build, loom_value_id_t input,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(builder, source->fastmath_flags, input,
                             source->result_type, source->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_binary(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source,
    loom_math_legalize_binary_build_fn_t build, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(builder, source->fastmath_flags, lhs, rhs,
                             source->result_type, source->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_exp_exp2(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t log2_e = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 1.44269504088896340736, &log2_e));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input, log2_e,
      &scaled));
  return loom_math_legalize_build_unary(
      builder, source, source->lane_builders->exp2f, scaled, out_value);
}

static iree_status_t loom_math_legalize_build_logistic_exp2(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t negative_log2_e = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t exponent = LOOM_VALUE_ID_INVALID;
  loom_value_id_t denominator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, -1.44269504088896340736, &negative_log2_e));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input,
      negative_log2_e, &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->exp2f, scaled, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, one, exponent,
      &denominator));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->divf, one,
                                         denominator, out_value);
}

static iree_status_t loom_math_legalize_build_silu_logistic(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t logistic = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->logisticf, source->input,
      &logistic));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->mulf,
                                         source->input, logistic, out_value);
}

static iree_status_t loom_math_legalize_build_softplus_exp2(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t log2_e = LOOM_VALUE_ID_INVALID;
  loom_value_id_t ln2 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t exponent = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sum = LOOM_VALUE_ID_INVALID;
  loom_value_id_t log2_sum = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 1.44269504088896340736, &log2_e));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.69314718055994530942, &ln2));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input, log2_e,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->exp2f, scaled, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, one, exponent, &sum));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->log2f, sum, &log2_sum));
  return loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, log2_sum, ln2, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_tanh(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t half = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cubic_coefficient = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sqrt_2_over_pi = LOOM_VALUE_ID_INVALID;
  loom_value_id_t input_squared = LOOM_VALUE_ID_INVALID;
  loom_value_id_t input_cubed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cubic_term = LOOM_VALUE_ID_INVALID;
  loom_value_id_t inner_sum = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t tanh = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one_plus_tanh = LOOM_VALUE_ID_INVALID;
  loom_value_id_t half_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.5, &half));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.044715, &cubic_coefficient));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.79788456080286535588, &sqrt_2_over_pi));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input,
      source->input, &input_squared));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, input_squared,
      source->input, &input_cubed));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, cubic_coefficient,
      input_cubed, &cubic_term));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, source->input, cubic_term,
      &inner_sum));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, sqrt_2_over_pi, inner_sum,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->tanhf, scaled, &tanh));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, one, tanh, &one_plus_tanh));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, half, source->input,
      &half_input));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->mulf,
                                         half_input, one_plus_tanh, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_logistic(
    loom_builder_t* builder,
    const loom_math_legalize_activation_source_t* source, const loom_op_t* op,
    loom_value_id_t* out_value) {
  loom_value_id_t scale = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t logistic = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, loom_math_legalize_gelu_logistic_scale(op), &scale));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, scale, source->input,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->logisticf, scaled, &logistic));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->mulf,
                                         source->input, logistic, out_value);
}

static iree_status_t loom_math_legalize_build_recipe(
    const loom_math_legalize_recipe_t* recipe,
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, loom_value_id_t* out_value) {
  loom_math_legalize_activation_source_t source;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_activation_source_initialize(context, op, &source));
  switch (recipe->recipe) {
    case LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32:
      return loom_math_legalize_build_exp_exp2(&rewriter->builder, &source,
                                               out_value);
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32:
      return loom_math_legalize_build_logistic_exp2(&rewriter->builder, &source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32:
      return loom_math_legalize_build_silu_logistic(&rewriter->builder, &source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32:
      return loom_math_legalize_build_softplus_exp2(&rewriter->builder, &source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32:
      return loom_math_legalize_build_gelu_tanh(&rewriter->builder, &source,
                                                out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32:
      return loom_math_legalize_build_gelu_logistic(&rewriter->builder, &source,
                                                    op, out_value);
    case LOOM_TARGET_MATH_RECIPE_UNKNOWN:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unknown math activation recipe %u", recipe->recipe);
}

static iree_status_t loom_math_legalize_rewrite_activation(
    const loom_math_legalize_recipe_t* recipe,
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter) {
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_recipe(recipe, context, op,
                                                       rewriter, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static const loom_math_legalize_recipe_t kActivationRecipes[] = {
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32,
        .root_kind = LOOM_OP_SCALAR_EXPF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32,
        .root_kind = LOOM_OP_VECTOR_EXPF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32,
        .root_kind = LOOM_OP_SCALAR_LOGISTICF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32,
        .root_kind = LOOM_OP_VECTOR_LOGISTICF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32,
        .root_kind = LOOM_OP_SCALAR_SILUF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32,
        .root_kind = LOOM_OP_VECTOR_SILUF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32,
        .root_kind = LOOM_OP_SCALAR_SOFTPLUSF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32,
        .root_kind = LOOM_OP_VECTOR_SOFTPLUSF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32,
        .root_kind = LOOM_OP_SCALAR_GELUF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32,
        .root_kind = LOOM_OP_VECTOR_GELUF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32,
        .root_kind = LOOM_OP_SCALAR_GELUF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
    {
        .recipe = LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32,
        .root_kind = LOOM_OP_VECTOR_GELUF,
        .rewrite = loom_math_legalize_rewrite_activation,
    },
};

iree_status_t loom_math_legalize_collect_activation_recipes(
    iree_arena_allocator_t* arena,
    loom_math_legalize_recipe_table_t* out_table) {
  *out_table = (loom_math_legalize_recipe_table_t){
      .recipes = kActivationRecipes,
      .recipe_count = IREE_ARRAYSIZE(kActivationRecipes),
  };
  return iree_ok_status();
}
