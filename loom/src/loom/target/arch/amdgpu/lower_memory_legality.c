// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/target/arch/amdgpu/lower_memory_internal.h"

iree_status_t loom_amdgpu_low_legality_verify_vector_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (op->kind != LOOM_OP_VECTOR_LOAD && op->kind != LOOM_OP_VECTOR_STORE) {
    *out_handled = false;
    return iree_ok_status();
  }

  loom_amdgpu_memory_access_plan_t access = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  const loom_func_like_t source_function =
      iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                       LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS)
          ? loom_target_low_legality_function(context)
          : (loom_func_like_t){0};
  if (!loom_amdgpu_memory_access_select(
          module, loom_target_low_legality_fact_table(context), descriptor_set,
          source_function, op, &access, &source_diagnostic, &diagnostic)) {
    const iree_string_view_t detail =
        source_diagnostic.rejection_bits != 0
            ? loom_low_source_memory_access_rejection_detail(
                  source_diagnostic.rejection_bits)
            : loom_amdgpu_memory_access_rejection_detail(
                  diagnostic.rejection_bits);
    return loom_target_low_legality_reject(context, provider, op,
                                           IREE_SV("memory"),
                                           loom_op_name(module, op), detail);
  }
  if (!loom_amdgpu_memory_cache_policy_can_lower(descriptor_set, &access)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("cache"), loom_op_name(module, op),
        IREE_SV("AMDGPU vector memory cache policy is not representable by "
                "the selected descriptor set"));
  }
  const loom_amdgpu_memory_operation_kind_t kind =
      loom_amdgpu_memory_operation_kind_from_source(&access.source);
  return loom_amdgpu_record_memory_access_diagnostic(
      provider, context, op, descriptor_set, &access, kind);
}
