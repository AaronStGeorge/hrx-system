// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_symbol_facts.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"

static iree_status_t loom_function_symbol_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "function %.*s string id %u is invalid",
        (int)field_name.size, field_name.data, (uint32_t)string_id);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static bool loom_function_symbol_attr_present(loom_func_like_t function,
                                              uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(function.op)[attr_index]);
}

static iree_status_t loom_function_symbol_string_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  if (attr->kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function contract field %.*s must be a string",
                            (int)field_name.size, field_name.data);
  }
  return loom_function_symbol_string_from_id(
      module, loom_attr_as_string_id(*attr), field_name, out_string);
}

static iree_status_t loom_function_symbol_u32_attr(
    const loom_attribute_t* attr, iree_string_view_t field_name,
    uint32_t* out_value) {
  if (attr->kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function contract field %.*s must be an integer",
                            (int)field_name.size, field_name.data);
  }
  int64_t value = loom_attr_as_i64(*attr);
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function contract field %.*s must fit in u32",
                            (int)field_name.size, field_name.data);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_function_symbol_linkage_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_target_linkage_t* out_linkage) {
  if (attr->kind == LOOM_ATTR_ENUM) {
    *out_linkage = (loom_target_linkage_t)loom_attr_as_enum(*attr);
    return iree_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_function_symbol_string_attr(
      module, attr, IREE_SV("linkage"), &value));
  if (iree_string_view_equal(value, IREE_SV("default"))) {
    *out_linkage = LOOM_TARGET_LINKAGE_DEFAULT;
  } else if (iree_string_view_equal(value, IREE_SV("dso_local"))) {
    *out_linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function contract linkage '%.*s' is unknown",
                            (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_function_symbol_apply_abi_attr(
    const loom_module_t* module, iree_string_view_t name,
    const loom_attribute_t* value, loom_target_export_plan_t* export_plan) {
  if (iree_string_view_equal(name, IREE_SV("hal_binding_alignment"))) {
    return loom_function_symbol_u32_attr(
        value, name, &export_plan->hal_kernel.binding_alignment);
  } else if (iree_string_view_equal(name, IREE_SV("hal_workgroup_size_x"))) {
    return loom_function_symbol_u32_attr(
        value, name, &export_plan->hal_kernel.required_workgroup_size.x);
  } else if (iree_string_view_equal(name, IREE_SV("hal_workgroup_size_y"))) {
    return loom_function_symbol_u32_attr(
        value, name, &export_plan->hal_kernel.required_workgroup_size.y);
  } else if (iree_string_view_equal(name, IREE_SV("hal_workgroup_size_z"))) {
    return loom_function_symbol_u32_attr(
        value, name, &export_plan->hal_kernel.required_workgroup_size.z);
  } else if (iree_string_view_equal(name,
                                    IREE_SV("hal_flat_workgroup_size_min"))) {
    return loom_function_symbol_u32_attr(
        value, name, &export_plan->hal_kernel.flat_workgroup_size_min);
  } else if (iree_string_view_equal(name,
                                    IREE_SV("hal_flat_workgroup_size_max"))) {
    return loom_function_symbol_u32_attr(
        value, name, &export_plan->hal_kernel.flat_workgroup_size_max);
  } else if (iree_string_view_equal(name,
                                    IREE_SV("hal_buffer_resource_flags"))) {
    return loom_function_symbol_u32_attr(
        value, name, &export_plan->hal_kernel.buffer_resource_flags);
  }
  (void)module;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "function ABI contract field '%.*s' is not supported",
                          (int)name.size, name.data);
}

static iree_status_t loom_function_symbol_apply_abi_attrs(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_target_export_plan_t* export_plan) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_function_symbol_string_from_id(
        module, entry->name_id, IREE_SV("ABI field"), &name));
    IREE_RETURN_IF_ERROR(loom_function_symbol_apply_abi_attr(
        module, name, &entry->value, export_plan));
  }
  return iree_ok_status();
}

static iree_status_t loom_function_symbol_apply_export_attr(
    const loom_module_t* module, iree_string_view_t name,
    const loom_attribute_t* value, loom_function_symbol_facts_t* facts) {
  if (iree_string_view_equal(name, IREE_SV("artifact"))) {
    if (value->kind != LOOM_ATTR_SYMBOL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function export field artifact must be a symbol");
    }
    facts->artifact_symbol = loom_attr_as_symbol(*value);
  } else if (iree_string_view_equal(name, IREE_SV("ordinal"))) {
    IREE_RETURN_IF_ERROR(
        loom_function_symbol_u32_attr(value, name, &facts->export_ordinal));
    facts->has_export_ordinal = true;
  } else if (iree_string_view_equal(name, IREE_SV("linkage"))) {
    return loom_function_symbol_linkage_attr(module, value,
                                             &facts->export_plan.linkage);
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function export contract field '%.*s' is not supported",
        (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_function_symbol_apply_export_attrs(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_function_symbol_facts_t* facts) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_function_symbol_string_from_id(
        module, entry->name_id, IREE_SV("export field"), &name));
    IREE_RETURN_IF_ERROR(loom_function_symbol_apply_export_attr(
        module, name, &entry->value, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_function_symbol_resolve_target_profile(
    loom_symbol_fact_context_t* context, loom_symbol_ref_t target_symbol,
    const loom_target_profile_symbol_facts_t** out_target_profile) {
  *out_target_profile = NULL;
  if (!loom_symbol_ref_is_valid(target_symbol)) {
    return iree_ok_status();
  }

  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_lookup_ref(
      context, target_symbol, &target_base_facts));
  const loom_target_profile_symbol_facts_t* target_profile =
      loom_target_profile_symbol_facts_cast(target_base_facts);
  if (!target_profile) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function target must resolve to target.profile symbol facts");
  }
  *out_target_profile = target_profile;
  return iree_ok_status();
}

static iree_status_t loom_function_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  IREE_ASSERT_ARGUMENT(domain);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;

  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "function symbol fact domain attached to a non-FuncLike op");
  }

  loom_function_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->function_op = function.op;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  IREE_RETURN_IF_ERROR(loom_function_symbol_string_from_id(
      module, symbol->name_id, IREE_SV("symbol"), &facts->name));
  facts->visibility = loom_func_like_visibility(function);
  facts->calling_convention = loom_func_like_cc(function);
  facts->purity = loom_func_like_purity(function);
  facts->has_body = loom_func_like_body(function) != NULL;
  facts->argument_ids =
      loom_func_like_arg_ids(function, &facts->argument_count);
  facts->result_ids = loom_op_const_results(function.op);
  facts->result_count = function.op->result_count;
  facts->target_symbol = loom_func_like_target(function);
  IREE_RETURN_IF_ERROR(loom_function_symbol_resolve_target_profile(
      context, facts->target_symbol, &facts->target_profile));

  bool has_abi_attr = loom_function_symbol_attr_present(
      function, function.vtable->abi_attr_index);
  loom_named_attr_slice_t abi_attrs = loom_func_like_abi_attrs(function);
  loom_string_id_t export_symbol_id = loom_func_like_export_symbol(function);
  bool has_export_symbol = export_symbol_id != LOOM_STRING_ID_INVALID;
  loom_named_attr_slice_t export_attrs = loom_func_like_export_attrs(function);
  bool has_function_contract = has_abi_attr || abi_attrs.count > 0 ||
                               has_export_symbol || export_attrs.count > 0;
  if (has_function_contract && !facts->target_profile) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function target ABI/export contracts require a target profile");
  }

  if (facts->target_profile) {
    facts->target_bundle = &facts->target_profile->bundle;
    facts->export_plan = facts->target_profile->export_plan;
    facts->export_plan.name = facts->name;
    facts->export_plan.source_symbol = iree_string_view_empty();
    facts->export_plan.export_symbol = iree_string_view_empty();
    if (has_abi_attr) {
      facts->export_plan.abi_kind =
          (loom_target_abi_kind_t)loom_func_like_abi(function);
    }
    if (facts->export_plan.abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function target contract must resolve a concrete ABI");
    }
    IREE_RETURN_IF_ERROR(loom_function_symbol_apply_abi_attrs(
        module, abi_attrs, &facts->export_plan));
    if (has_export_symbol) {
      IREE_RETURN_IF_ERROR(loom_function_symbol_string_from_id(
          module, export_symbol_id, IREE_SV("export_symbol"),
          &facts->export_plan.export_symbol));
      facts->exports = true;
    }
    IREE_RETURN_IF_ERROR(
        loom_function_symbol_apply_export_attrs(module, export_attrs, facts));
    if (export_attrs.count > 0) {
      facts->exports = true;
    }
  }

  *out_facts = &facts->base;
  return iree_ok_status();
}

const loom_symbol_fact_domain_t loom_function_symbol_fact_domain = {
    .compute = loom_function_symbol_fact_compute,
};

const loom_function_symbol_facts_t* loom_function_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts) {
  if (!facts || facts->domain != &loom_function_symbol_fact_domain) {
    return NULL;
  }
  return (const loom_function_symbol_facts_t*)facts;
}
