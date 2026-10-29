// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/math_policy.h"

#include "iree/testing/gtest.h"
#include "loom/ir/types.h"
#include "loom/target/types.h"

namespace {

TEST(MathPolicyTest, OrdinaryElementaryFamily) {
  loom_target_math_policy_registry_t registry = {};
  loom_vm_math_policy_registry_initialize(&registry);
  const auto* policy = loom_target_math_policy_registry_lookup(
      &registry, UINT64_C(0x32476f14332ccb19));
  ASSERT_NE(policy, nullptr);
  const loom_target_config_t config = {};
  const loom_target_bundle_t bundle = {.config = &config};
  const loom_scalar_type_t elements[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  const loom_target_math_op_t operations[] = {
      LOOM_TARGET_MATH_OP_EXPF,   LOOM_TARGET_MATH_OP_EXP2F,
      LOOM_TARGET_MATH_OP_SQRTF,  LOOM_TARGET_MATH_OP_RSQRTF,
      LOOM_TARGET_MATH_OP_CBRTF,  LOOM_TARGET_MATH_OP_LOGF,
      LOOM_TARGET_MATH_OP_LOG2F,  LOOM_TARGET_MATH_OP_LOG10F,
      LOOM_TARGET_MATH_OP_LOG1PF, LOOM_TARGET_MATH_OP_EXPM1F,
      LOOM_TARGET_MATH_OP_POWF,   LOOM_TARGET_MATH_OP_SINHF,
      LOOM_TARGET_MATH_OP_COSHF,  LOOM_TARGET_MATH_OP_TANHF,
      LOOM_TARGET_MATH_OP_ASINHF, LOOM_TARGET_MATH_OP_ACOSHF,
      LOOM_TARGET_MATH_OP_ATANHF,
  };
  for (auto operation : operations) {
    SCOPED_TRACE(operation);
    for (auto element : elements) {
      SCOPED_TRACE(element);
      for (int lanes : {0, 1, 3, 8}) {
        SCOPED_TRACE(lanes);
        for (bool approximate : {false, true}) {
          const loom_target_math_query_t query = {
              .target_bundle = &bundle,
              .math_op = operation,
              .lane_domain = lanes ? LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR
                                   : LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
              .value_type =
                  lanes ? loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                                              loom_dim_pack_static(lanes), 0)
                        : loom_type_scalar(element),
              .element_type = element,
              .fastmath_flags = static_cast<loom_target_math_fastmath_flags_t>(
                  approximate ? LOOM_TARGET_MATH_FASTMATH_FLAG_AFN
                              : LOOM_TARGET_MATH_FASTMATH_FLAG_NONE),
          };
          loom_target_math_policy_decision_t decision = {};
          loom_target_math_policy_query(policy, &query, &decision);
          const bool narrow = loom_scalar_type_bitwidth(element) < 32;
          const bool inverse_recipe =
              approximate && element != LOOM_SCALAR_TYPE_F64 &&
              (operation == LOOM_TARGET_MATH_OP_ASINHF ||
               operation == LOOM_TARGET_MATH_OP_ACOSHF ||
               operation == LOOM_TARGET_MATH_OP_ATANHF);
          EXPECT_EQ(decision.action,
                    narrow || inverse_recipe
                        ? LOOM_TARGET_MATH_POLICY_ACTION_REWRITE
                        : LOOM_TARGET_MATH_POLICY_ACTION_KEEP);
          if (narrow || inverse_recipe) {
            EXPECT_EQ(decision.recipe,
                      inverse_recipe
                          ? LOOM_TARGET_MATH_RECIPE_INVERSE_HYPERBOLIC_F32
                          : LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND);
          }
          EXPECT_EQ(decision.recipe_fastmath_flags, 0);
        }
      }
    }
  }
}

}  // namespace
