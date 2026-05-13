// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/target/arch/ireevm/descriptors.h"
#include "loom/target/emit/ireevm/contracts/core.h"
#include "loom/target/emit/ireevm/contracts/core_lower_rules.h"
#include "loom/target/emit/ireevm/lower.h"

static bool loom_ireevm_type_is_supported_scalar(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  loom_scalar_type_t element_type = loom_type_element_type(type);
  switch (element_type) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_ireevm_map_type(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  (void)user_data;
  if (!loom_ireevm_type_is_supported_scalar(source_type)) {
    return loom_low_lower_emit_source_type_unsupported(
        context, source_op, IREE_SV("source"), source_type);
  }
  switch (loom_type_element_type(source_type)) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I32:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_I32, 1, out_low_type);
    case LOOM_SCALAR_TYPE_I64:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_I64, 2, out_low_type);
    case LOOM_SCALAR_TYPE_F32:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_F32, 1, out_low_type);
    case LOOM_SCALAR_TYPE_F64:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_F64, 2, out_low_type);
    default:
      IREE_ASSERT_UNREACHABLE(
          "checked by loom_ireevm_type_is_supported_scalar");
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL, "unreachable scalar type");
}

static const loom_low_lower_rule_set_t* const kIreeVmRuleSets[] = {
    &loom_ireevm_core_lower_rule_set,
};

static const loom_target_contract_binding_t kIreeVmContractBindings[] = {
    {&loom_ireevm_core_contract_fragment, 0},
};

static const loom_low_lower_policy_t kIreeVmLowLowerPolicy = {
    .name = IREE_SVL("iree-vm-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_ireevm_map_type, .user_data = NULL},
    .import_decl_kind = LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM,
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kIreeVmRuleSets),
            .values = kIreeVmRuleSets,
        },
    .contract_bindings = kIreeVmContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kIreeVmContractBindings),
};

const loom_low_lower_policy_t* loom_ireevm_low_lower_policy(void) {
  return &kIreeVmLowLowerPolicy;
}

void loom_ireevm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("ireevm.core"),
          .policy = &kIreeVmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
