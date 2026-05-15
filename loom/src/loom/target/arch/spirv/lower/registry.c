// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/target/arch/spirv/contracts/logical_core.h"
#include "loom/target/arch/spirv/contracts/logical_core_lower_rules.h"
#include "loom/target/arch/spirv/descriptors.h"
#include "loom/target/arch/spirv/lower.h"

static iree_status_t loom_spirv_make_register_type(
    loom_low_lower_context_t* context, uint16_t register_class_id,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, register_class_id, 1,
                                           out_type);
}

static bool loom_spirv_source_type_is_id(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_I32 ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX;
}

static bool loom_spirv_source_type_is_offset64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

static iree_status_t loom_spirv_map_type(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_spirv_source_type_is_id(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, out_low_type);
  }
  if (loom_spirv_source_type_is_offset64(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, out_low_type);
  }
  if (loom_type_is_buffer(source_type) || loom_type_is_view(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER,
        out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static const loom_low_lower_rule_set_t* const kSpirvRuleSets[] = {
    &loom_spirv_logical_core_lower_rule_set,
};

static const loom_target_contract_binding_t kSpirvContractBindings[] = {
    {&loom_spirv_logical_core_contract_fragment, 0},
};

static const loom_low_lower_policy_t kSpirvLowLowerPolicy = {
    .name = IREE_SVL("spirv-logical-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_spirv_map_type, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kSpirvRuleSets),
            .values = kSpirvRuleSets,
        },
    .contract_bindings = kSpirvContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kSpirvContractBindings),
};

const loom_low_lower_policy_t* loom_spirv_low_lower_policy(void) {
  return &kSpirvLowLowerPolicy;
}

void loom_spirv_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("spirv.logical.core"),
          .policy = &kSpirvLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
