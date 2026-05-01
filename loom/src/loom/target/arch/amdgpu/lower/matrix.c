// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/analysis/contract_vector.h"
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
  const uint16_t wavefront_size = (uint16_t)processor->default_wavefront_size;

  *out_facts = (loom_amdgpu_matrix_target_facts_t){
      .options =
          (loom_contract_vector_mma_options_t){
              .k_group_size = 1,
              .fragment =
                  (loom_contract_fragment_t){
                      .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
                      .subgroup_size = wavefront_size,
                  },
              .capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX,
              .policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
          },
      .feature_bits = feature_bits,
      .wave_size = wavefront_size,
  };
  return iree_ok_status();
}

static bool loom_amdgpu_matrix_select_contract(
    const loom_contract_request_t* contract_request,
    const loom_amdgpu_matrix_target_facts_t* target_facts,
    const loom_amdgpu_matrix_contract_descriptor_t** out_descriptor,
    loom_contract_diagnostic_t* out_contract_diagnostic,
    loom_amdgpu_matrix_contract_match_diagnostic_t* out_match_diagnostic) {
  *out_descriptor = NULL;
  if (out_contract_diagnostic != NULL) {
    *out_contract_diagnostic = (loom_contract_diagnostic_t){0};
  }
  if (out_match_diagnostic != NULL) {
    *out_match_diagnostic = (loom_amdgpu_matrix_contract_match_diagnostic_t){0};
  }

  loom_amdgpu_matrix_contract_match_request_t match_request = {0};
  if (!loom_amdgpu_matrix_contract_match_request_from_contract(
          contract_request, target_facts->feature_bits, target_facts->wave_size,
          &match_request, out_contract_diagnostic)) {
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

static iree_status_t loom_amdgpu_matrix_contract_query_reject(
    const loom_target_contract_query_environment_t* environment,
    loom_target_contract_query_outcome_t outcome,
    iree_string_view_t subject_kind, iree_string_view_t subject_name,
    iree_string_view_t reason,
    loom_target_contract_query_result_t* out_result) {
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  *rejection = (loom_target_contract_rejection_t){
      .subject_kind = subject_kind,
      .subject_name = subject_name,
      .reason = reason,
  };
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = outcome;
  out_result->rejection = rejection;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options) {
  (void)user_data;
  (void)rule;
  *out_options = (loom_contract_vector_mma_options_t){0};
  loom_amdgpu_matrix_target_facts_t target_facts = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_target_facts_from_bundle(
      environment->bundle, &target_facts));
  *out_options = target_facts.options;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_contract_request_t* contract_request,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  (void)rule;
  *out_result = loom_target_contract_query_result_empty();

  loom_amdgpu_matrix_target_facts_t target_facts = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_target_facts_from_bundle(
      environment->bundle, &target_facts));
  loom_contract_diagnostic_t contract_diagnostic = {0};
  loom_amdgpu_matrix_contract_match_diagnostic_t match_diagnostic = {0};
  const loom_amdgpu_matrix_contract_descriptor_t* contract_descriptor = NULL;
  if (!loom_amdgpu_matrix_select_contract(
          contract_request, &target_facts, &contract_descriptor,
          &contract_diagnostic, &match_diagnostic)) {
    return loom_amdgpu_matrix_contract_query_reject(
        environment, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED, IREE_SV("op"),
        IREE_SV("vector.mma"),
        loom_amdgpu_matrix_contract_rejection_reason(contract_diagnostic,
                                                     match_diagnostic),
        out_result);
  }
  if (contract_descriptor->low_descriptor_id ==
      LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_ID_NONE) {
    return loom_amdgpu_matrix_contract_query_reject(
        environment, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
        IREE_SV("contract"), contract_descriptor->name,
        IREE_SV("selected AMDGPU matrix contract has no target-low "
                "descriptor"),
        out_result);
  }
  if (loom_low_descriptor_set_lookup_descriptor_by_id(
          environment->descriptor_set,
          contract_descriptor->low_descriptor_id) ==
      LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_amdgpu_matrix_contract_query_reject(
        environment, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
        IREE_SV("descriptor"), contract_descriptor->name,
        IREE_SV("the selected AMDGPU descriptor set does not contain the "
                "matrix packet"),
        out_result);
  }

  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  out_result->selected_descriptor_id = contract_descriptor->low_descriptor_id;
  out_result->source_rejection_bits = contract_diagnostic.rejection_bits;
  out_result->target_rejection_bits = match_diagnostic.rejection_bits;
  return iree_ok_status();
}
