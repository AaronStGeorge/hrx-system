// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/analysis/contract_vector.h"
#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/matrix_contract.h"
#include "loom/target/arch/amdgpu/matrix_contract_projection.h"
#include "loom/target/arch/amdgpu/target_info.h"

typedef struct loom_amdgpu_matrix_target_facts_t {
  // Generic vector.mma adapter options for this AMDGPU target.
  loom_contract_vector_mma_options_t options;
  // Matrix feature bits available on the selected processor.
  loom_amdgpu_matrix_feature_bits_t feature_bits;
  // Concrete wave size selected by the target processor.
  uint32_t wave_size;
} loom_amdgpu_matrix_target_facts_t;

static iree_status_t loom_amdgpu_matrix_target_facts_from_bundle(
    const loom_target_bundle_t* bundle,
    loom_amdgpu_matrix_target_facts_t* out_facts) {
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = (loom_amdgpu_matrix_target_facts_t){0};
  if (bundle == NULL || bundle->snapshot == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU matrix lowering requires a target "
                            "snapshot");
  }

  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_processor(
      bundle->snapshot->target_cpu, &processor));
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  (void)loom_amdgpu_matrix_feature_bits_for_profile(
      processor->matrix_feature_profile, &feature_bits);
  IREE_ASSERT_LE(processor->default_wavefront_size, UINT16_MAX);

  *out_facts = (loom_amdgpu_matrix_target_facts_t){
      .options =
          (loom_contract_vector_mma_options_t){
              .k_group_size = 1,
              .fragment =
                  (loom_contract_fragment_t){
                      .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
                      .subgroup_size =
                          (uint16_t)processor->default_wavefront_size,
                  },
              .capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX,
              .policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
          },
      .feature_bits = feature_bits,
      .wave_size = processor->default_wavefront_size,
  };
  return iree_ok_status();
}

static bool loom_amdgpu_matrix_select_contract(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* op, const loom_amdgpu_matrix_target_facts_t* target_facts,
    const loom_amdgpu_matrix_contract_descriptor_t** out_descriptor,
    loom_contract_diagnostic_t* out_contract_diagnostic,
    loom_amdgpu_matrix_contract_match_diagnostic_t* out_match_diagnostic) {
  IREE_ASSERT_ARGUMENT(out_descriptor);
  *out_descriptor = NULL;
  if (out_contract_diagnostic != NULL) {
    *out_contract_diagnostic = (loom_contract_diagnostic_t){0};
  }
  if (out_match_diagnostic != NULL) {
    *out_match_diagnostic = (loom_amdgpu_matrix_contract_match_diagnostic_t){0};
  }

  loom_contract_request_t contract_request = {0};
  if (!loom_contract_request_from_vector_mma_op(
          module, fact_table, op, &target_facts->options, &contract_request,
          out_contract_diagnostic)) {
    return false;
  }

  loom_amdgpu_matrix_contract_match_request_t match_request = {0};
  if (!loom_amdgpu_matrix_contract_match_request_from_contract(
          &contract_request, target_facts->feature_bits,
          target_facts->wave_size, &match_request, out_contract_diagnostic)) {
    return false;
  }

  *out_descriptor =
      loom_amdgpu_matrix_contract_select(&match_request, out_match_diagnostic);
  return *out_descriptor != NULL;
}

static iree_string_view_t loom_amdgpu_matrix_contract_rejection_reason(
    loom_contract_diagnostic_t contract_diagnostic,
    loom_amdgpu_matrix_contract_match_diagnostic_t match_diagnostic) {
  const loom_contract_rejection_bits_t contract_bits =
      contract_diagnostic.rejection_bits;
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_ROLE)) {
    return IREE_SV(
        "vector.mma operands must carry lhs, rhs, and init "
        "fragment facts");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
    return IREE_SV(
        "vector.mma fragment shapes must resolve to exact "
        "compatible M/N/K values");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_SCHEMA)) {
    return IREE_SV(
        "encoded vector.mma fragments need a resolved target "
        "fragment schema");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
    return IREE_SV(
        "vector.mma operands use numeric types unsupported by the "
        "AMDGPU matrix contract projection");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
    return IREE_SV(
        "AMDGPU matrix lowering requires subgroup-lane fragment "
        "ownership facts");
  }
  if (iree_any_bit_set(contract_bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
    return IREE_SV(
        "vector.mma requires matrix capabilities unavailable in "
        "the AMDGPU projection");
  }
  if (iree_any_bit_set(contract_bits,
                       LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
    return IREE_SV("vector.mma did not form a valid matrix contract request");
  }

  const loom_amdgpu_matrix_contract_rejection_bits_t match_bits =
      match_diagnostic.rejection_bits;
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FAMILY)) {
    return IREE_SV(
        "no AMDGPU matrix descriptor matches the requested "
        "instruction family");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE)) {
    return IREE_SV(
        "no AMDGPU matrix descriptor matches the exact M/N/K tile "
        "shape");
  }
  if (iree_any_bit_set(
          match_bits,
          LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RHS_PAYLOAD |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_ACCUMULATOR_PAYLOAD |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RESULT_PAYLOAD)) {
    return IREE_SV(
        "no AMDGPU matrix descriptor matches the physical operand "
        "payload shapes");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_KIND)) {
    return IREE_SV(
        "no AMDGPU matrix descriptor matches the scale operand "
        "kind");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES)) {
    return IREE_SV(
        "the selected AMDGPU processor lacks the required matrix "
        "instruction features");
  }
  if (iree_any_bit_set(match_bits,
                       LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE)) {
    return IREE_SV(
        "the selected AMDGPU wave size is incompatible with the "
        "matrix descriptor");
  }
  if (iree_any_bit_set(
          match_bits,
          LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SPARSE |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_CLAMP |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SIGN_SELECT |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_AB_MODIFIERS |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_C_MODIFIER |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_OPSEL |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS |
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS)) {
    return IREE_SV(
        "the selected AMDGPU matrix descriptor requires facts or "
        "operands not present on the vector.mma contract");
  }
  return IREE_SV("no AMDGPU matrix descriptor matches vector.mma");
}

iree_status_t loom_amdgpu_select_vector_mma_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_matrix_mma_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_plan = (loom_amdgpu_matrix_mma_plan_t){0};
  *out_selected = false;
  if (!loom_vector_mma_isa(source_op)) {
    return iree_ok_status();
  }

  loom_amdgpu_matrix_target_facts_t target_facts = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_target_facts_from_bundle(
      loom_low_lower_context_bundle(context), &target_facts));

  const loom_amdgpu_matrix_contract_descriptor_t* contract_descriptor = NULL;
  if (!loom_amdgpu_matrix_select_contract(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op, &target_facts,
          &contract_descriptor, NULL, NULL) ||
      contract_descriptor->low_descriptor_id ==
          LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_ID_NONE) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t low_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, contract_descriptor->low_descriptor_id, &low_descriptor));
  *out_plan = (loom_amdgpu_matrix_mma_plan_t){
      .descriptor = low_descriptor,
      .lhs = loom_vector_mma_lhs(source_op),
      .rhs = loom_vector_mma_rhs(source_op),
      .init = loom_vector_mma_init(source_op),
      .result = loom_vector_mma_result(source_op),
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_mma_plan_t* plan) {
  IREE_ASSERT(plan->descriptor.descriptor != NULL);

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->rhs, &low_rhs));
  loom_value_id_t low_init = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->init, &low_init));

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, plan->result, &result_low_type));
  const loom_value_id_t operands[] = {low_lhs, low_rhs, low_init};
  const loom_tied_result_t tied_results[] = {
      {
          .result_index = 0,
          .operand_index = 2,
          .has_type_change = false,
      },
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &result_low_type, 1, tied_results,
      IREE_ARRAYSIZE(tied_results), source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, plan->result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

iree_status_t loom_amdgpu_low_legality_verify_vector_mma(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle) ||
      !loom_vector_mma_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_matrix_target_facts_t target_facts = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_matrix_target_facts_from_bundle(bundle, &target_facts));
  loom_contract_diagnostic_t contract_diagnostic = {0};
  loom_amdgpu_matrix_contract_match_diagnostic_t match_diagnostic = {0};
  const loom_amdgpu_matrix_contract_descriptor_t* contract_descriptor = NULL;
  if (!loom_amdgpu_matrix_select_contract(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context), op, &target_facts,
          &contract_descriptor, &contract_diagnostic, &match_diagnostic)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("op"),
        loom_op_name(loom_target_low_legality_module(context), op),
        loom_amdgpu_matrix_contract_rejection_reason(contract_diagnostic,
                                                     match_diagnostic));
  }
  if (contract_descriptor->low_descriptor_id ==
      LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_ID_NONE) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"), contract_descriptor->name,
        IREE_SV("selected AMDGPU matrix contract has no target-low "
                "descriptor"));
  }
  return loom_target_low_legality_record_contract(
      context, provider, op, contract_descriptor->name, IREE_SV("selected"),
      IREE_SV("selected AMDGPU matrix contract descriptor"));
}
