// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_contract.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/target/facts.h"
#include "loom/target/launch.h"

static iree_status_t loom_target_function_contract_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "func %.*s string id %u is invalid",
        (int)field_name.size, field_name.data, (uint32_t)string_id);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_target_function_contract_apply_abi_attr(
    iree_string_view_t name, const loom_attribute_t* value,
    loom_target_export_plan_t* export_plan) {
  (void)value;
  (void)export_plan;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "func ABI override field '%.*s' is not supported",
                          (int)name.size, name.data);
}

static iree_status_t loom_target_function_contract_apply_abi_attrs(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_target_export_plan_t* export_plan) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_target_function_contract_string_from_id(
        module, entry->name_id, IREE_SV("ABI field"), &name));
    IREE_RETURN_IF_ERROR(loom_target_function_contract_apply_abi_attr(
        name, &entry->value, export_plan));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_function_contract_lookup_profile(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    const loom_target_profile_symbol_facts_t** out_target_profile) {
  *out_target_profile = NULL;
  if (!loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func @%.*s must declare a target profile",
                            (int)func_facts->name.size, func_facts->name.data);
  }
  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, func_facts->target_symbol, &target_base_facts));
  const loom_target_profile_symbol_facts_t* target_profile =
      loom_target_profile_symbol_facts_cast(target_base_facts);
  if (!target_profile) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "func target must resolve to target.profile symbol facts");
  }
  *out_target_profile = target_profile;
  return iree_ok_status();
}

iree_status_t loom_target_function_contract_resolve(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    loom_target_bundle_storage_t* out_bundle_storage) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(func_facts);
  IREE_ASSERT_ARGUMENT(out_bundle_storage);
  memset(out_bundle_storage, 0, sizeof(*out_bundle_storage));

  const loom_target_profile_symbol_facts_t* target_profile = NULL;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_lookup_profile(
      module, fact_table, func_facts, &target_profile));
  out_bundle_storage->snapshot = target_profile->snapshot;
  out_bundle_storage->export_plan = target_profile->export_plan;
  out_bundle_storage->config = target_profile->config;
  out_bundle_storage->bundle = (loom_target_bundle_t){
      .name = target_profile->bundle.name,
      .snapshot = &out_bundle_storage->snapshot,
      .export_plan = &out_bundle_storage->export_plan,
      .config = &out_bundle_storage->config,
  };

  out_bundle_storage->export_plan.name = func_facts->name;
  out_bundle_storage->export_plan.export_symbol = iree_string_view_empty();
  if (func_facts->has_abi) {
    if (func_facts->is_kernel_entry) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel entries derive their ABI from the "
                              "target profile");
    }
    out_bundle_storage->export_plan.abi_kind = func_facts->abi_kind;
  }
  if (out_bundle_storage->export_plan.abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func target contract must resolve a concrete ABI");
  }
  IREE_RETURN_IF_ERROR(loom_target_function_contract_apply_abi_attrs(
      module, func_facts->abi_attrs, &out_bundle_storage->export_plan));
  if (func_facts->is_kernel_entry &&
      out_bundle_storage->export_plan.abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel entry target contract must resolve a HAL "
                            "kernel ABI");
  }
  if (func_facts->has_required_workgroup_size) {
    out_bundle_storage->export_plan.hal_kernel.required_workgroup_size =
        func_facts->required_workgroup_size;
  }
  if (out_bundle_storage->export_plan.abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    if (out_bundle_storage->export_plan.hal_kernel.binding_alignment == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL kernel binding alignment must be non-zero");
    }
    IREE_RETURN_IF_ERROR(loom_target_validate_hal_kernel_launch(
        &out_bundle_storage->snapshot,
        &out_bundle_storage->export_plan.hal_kernel));
  }
  if (!iree_string_view_is_empty(func_facts->export_symbol)) {
    out_bundle_storage->export_plan.export_symbol = func_facts->export_symbol;
  }
  if (func_facts->has_export_linkage) {
    out_bundle_storage->export_plan.linkage = func_facts->export_linkage;
  }
  return iree_ok_status();
}
