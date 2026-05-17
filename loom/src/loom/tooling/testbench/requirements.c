// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/requirements.h"

#include <string.h>

#include "loom/ops/check/ops.h"

static iree_string_view_t loom_testbench_requirement_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

void loom_testbench_requirement_provider_registry_initialize(
    const loom_testbench_requirement_provider_t* providers,
    iree_host_size_t provider_count,
    loom_testbench_requirement_provider_registry_t* out_registry) {
  IREE_ASSERT(provider_count == 0 || providers);
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_testbench_requirement_provider_registry_t){
      .providers = provider_count != 0 ? providers : NULL,
      .provider_count = provider_count,
  };
}

const loom_named_attr_t* loom_testbench_requirement_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  IREE_ASSERT_ARGUMENT(module);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_testbench_requirement_module_string(module, attr->name_id),
            name)) {
      return attr;
    }
  }
  return NULL;
}

iree_status_t loom_testbench_requirement_read_string_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, iree_string_view_t* out_value) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_value);
  const loom_named_attr_t* attr =
      loom_testbench_requirement_find_attr(module, attrs, name);
  if (!attr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requirement attr '%.*s' is required",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requirement attr '%.*s' must be string",
                            (int)name.size, name.data);
  }
  *out_value = loom_testbench_requirement_module_string(
      module, loom_attr_as_string_id(attr->value));
  return iree_ok_status();
}

iree_status_t loom_testbench_requirement_read_optional_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present, int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_present);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_present = false;
  *out_value = 0;
  const loom_named_attr_t* attr =
      loom_testbench_requirement_find_attr(module, attrs, name);
  if (!attr) {
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requirement attr '%.*s' must be i64",
                            (int)name.size, name.data);
  }
  *out_present = true;
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static const loom_testbench_requirement_provider_t*
loom_testbench_requirement_find_provider(
    const loom_testbench_requirement_provider_registry_t* registry,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
    const loom_testbench_requirement_provider_t* provider =
        &registry->providers[i];
    if (iree_string_view_equal(provider->name, name)) {
      return provider;
    }
  }
  return NULL;
}

static iree_status_t loom_testbench_requirement_evaluate_provider(
    const loom_module_t* module,
    const loom_testbench_requirement_provider_registry_t* registry,
    iree_string_view_t provider_name, loom_named_attr_slice_t attrs,
    bool* out_satisfied, iree_string_view_t* out_reason) {
  const loom_testbench_requirement_provider_t* provider =
      loom_testbench_requirement_find_provider(registry, provider_name);
  if (!provider || provider->query == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "requirement provider '%.*s' is not registered",
                            (int)provider_name.size, provider_name.data);
  }
  return provider->query(provider->user_data, module, attrs, out_satisfied,
                         out_reason);
}

static iree_string_view_t loom_testbench_requirement_skip_reason(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t provider_reason, iree_string_view_t fallback_reason) {
  if (loom_check_skip_if_isa(op)) {
    loom_attribute_t reason_attr =
        loom_op_attrs(op)[loom_check_skip_if_reason_ATTR_INDEX];
    if (reason_attr.kind == LOOM_ATTR_STRING) {
      return loom_testbench_requirement_module_string(
          module, loom_attr_as_string_id(reason_attr));
    }
  }
  if (!iree_string_view_is_empty(provider_reason)) {
    return provider_reason;
  }
  return fallback_reason;
}

static iree_status_t loom_testbench_requirement_evaluate_op(
    const loom_module_t* module,
    const loom_testbench_requirement_provider_registry_t* registry,
    const loom_op_t* op, loom_testbench_requirement_result_t* result) {
  iree_string_view_t provider_name = iree_string_view_empty();
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (loom_check_requires_isa(op)) {
    provider_name = loom_testbench_requirement_module_string(
        module, loom_check_requires_provider(op));
    attrs = loom_check_requires_attrs(op);
  } else if (loom_check_skip_if_isa(op)) {
    provider_name = loom_testbench_requirement_module_string(
        module, loom_check_skip_if_provider(op));
    attrs = loom_check_skip_if_attrs(op);
  } else {
    return iree_ok_status();
  }

  bool satisfied = false;
  iree_string_view_t provider_reason = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_evaluate_provider(
      module, registry, provider_name, attrs, &satisfied, &provider_reason));

  const bool should_skip = loom_check_requires_isa(op) ? !satisfied : satisfied;
  if (should_skip) {
    *result = (loom_testbench_requirement_result_t){
        .skipped = true,
        .op = op,
        .provider = provider_name,
        .reason = loom_testbench_requirement_skip_reason(
            module, op, provider_reason,
            loom_check_requires_isa(op) ? IREE_SV("requirement not satisfied")
                                        : IREE_SV("skip predicate matched")),
    };
  }
  return iree_ok_status();
}

iree_status_t loom_testbench_evaluate_case_requirements(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_requirement_provider_registry_t* registry,
    loom_testbench_requirement_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_result, 0, sizeof(*out_result));

  loom_region_t* body = loom_check_case_body(case_plan->op);
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_testbench_requirement_evaluate_op(
          module, registry, op, out_result));
      if (out_result->skipped) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}
