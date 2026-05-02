// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_contract.h"

#include <string.h>

#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/target/facts.h"
#include "loom/target/launch.h"

static iree_string_view_t loom_target_function_contract_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return IREE_SV("<unknown>");
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_target_function_contract_symbol_ref_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return loom_target_function_contract_string_from_id(module, symbol->name_id);
}

static iree_status_t loom_target_function_contract_emit(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_op_t* op,
    uint16_t code, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, code),
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_target_function_contract_reject(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, uint16_t code,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    bool* out_valid) {
  *out_valid = false;
  return loom_target_function_contract_emit(
      diagnostic_emitter, func_facts->func_op, code, params, param_count);
}

static bool loom_target_function_contract_mul_u64(uint64_t lhs, uint64_t rhs,
                                                  uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) {
    *out_result = 0;
    return false;
  }
  *out_result = lhs * rhs;
  return true;
}

static iree_status_t loom_target_function_contract_apply_abi_attrs(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, loom_named_attr_slice_t attrs,
    bool* out_valid) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t name =
        loom_target_function_contract_string_from_id(module, entry->name_id);
    const loom_diagnostic_param_t params[] = {
        loom_param_string(func_facts->name),
        loom_param_string(name),
    };
    return loom_target_function_contract_reject(
        diagnostic_emitter, func_facts, 39, params, IREE_ARRAYSIZE(params),
        out_valid);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_function_contract_lookup_target(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    const loom_target_symbol_facts_t** out_target) {
  *out_target = NULL;
  if (!loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(func_facts->name),
    };
    return loom_target_function_contract_reject(
        diagnostic_emitter, func_facts, 53, params, IREE_ARRAYSIZE(params),
        out_valid);
  }
  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, func_facts->target_symbol, &target_base_facts));
  const loom_target_symbol_facts_t* target =
      loom_target_symbol_facts_cast(target_base_facts);
  if (!target) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(func_facts->name),
        loom_param_string(loom_target_function_contract_symbol_ref_name(
            module, func_facts->target_symbol)),
    };
    return loom_target_function_contract_reject(
        diagnostic_emitter, func_facts, 38, params, IREE_ARRAYSIZE(params),
        out_valid);
  }
  *out_target = target;
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_target_function_contract_reject_with_target(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts,
    const loom_target_symbol_facts_t* target, uint16_t code, bool* out_valid) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(target->name),
  };
  return loom_target_function_contract_reject(
      diagnostic_emitter, func_facts, code, params, IREE_ARRAYSIZE(params),
      out_valid);
}

static iree_status_t loom_target_function_contract_reject_dimension_limit(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts,
    const loom_target_symbol_facts_t* target, iree_string_view_t axis,
    uint32_t size, uint32_t limit, bool* out_valid) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(target->name),
      loom_param_string(axis),
      loom_param_u32(size),
      loom_param_u32(limit),
  };
  return loom_target_function_contract_reject(
      diagnostic_emitter, func_facts, 48, params, IREE_ARRAYSIZE(params),
      out_valid);
}

static iree_status_t loom_target_function_contract_validate_hal_kernel(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts,
    const loom_target_symbol_facts_t* target,
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel, bool* out_valid) {
  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  if (loom_target_workgroup_size_is_partial(required)) {
    return loom_target_function_contract_reject_with_target(
        diagnostic_emitter, func_facts, target, 44, out_valid);
  }
  const uint32_t max_flat_workgroup_size = snapshot->max_flat_workgroup_size;
  const uint32_t flat_min = hal_kernel->flat_workgroup_size_min;
  const uint32_t flat_max = hal_kernel->flat_workgroup_size_max;
  if ((flat_min == 0) != (flat_max == 0)) {
    return loom_target_function_contract_reject_with_target(
        diagnostic_emitter, func_facts, target, 45, out_valid);
  }
  if (flat_min != 0) {
    if (flat_min > flat_max) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(func_facts->name),
          loom_param_string(target->name),
          loom_param_u32(flat_min),
          loom_param_u32(flat_max),
      };
      return loom_target_function_contract_reject(
          diagnostic_emitter, func_facts, 46, params, IREE_ARRAYSIZE(params),
          out_valid);
    }
    if (max_flat_workgroup_size != 0 && flat_max > max_flat_workgroup_size) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(func_facts->name),
          loom_param_string(target->name),
          loom_param_u32(flat_max),
          loom_param_u32(max_flat_workgroup_size),
      };
      return loom_target_function_contract_reject(
          diagnostic_emitter, func_facts, 47, params, IREE_ARRAYSIZE(params),
          out_valid);
    }
  }
  if (loom_target_workgroup_size_is_empty(required)) {
    return iree_ok_status();
  }
  const loom_target_workgroup_size_t* limit = &snapshot->max_workgroup_size;
  if (limit->x != 0 && required->x > limit->x) {
    return loom_target_function_contract_reject_dimension_limit(
        diagnostic_emitter, func_facts, target, IREE_SV("x"), required->x,
        limit->x, out_valid);
  }
  if (limit->y != 0 && required->y > limit->y) {
    return loom_target_function_contract_reject_dimension_limit(
        diagnostic_emitter, func_facts, target, IREE_SV("y"), required->y,
        limit->y, out_valid);
  }
  if (limit->z != 0 && required->z > limit->z) {
    return loom_target_function_contract_reject_dimension_limit(
        diagnostic_emitter, func_facts, target, IREE_SV("z"), required->z,
        limit->z, out_valid);
  }
  uint64_t flat_size = 1;
  if (!loom_target_function_contract_mul_u64(flat_size, required->x,
                                             &flat_size) ||
      !loom_target_function_contract_mul_u64(flat_size, required->y,
                                             &flat_size) ||
      !loom_target_function_contract_mul_u64(flat_size, required->z,
                                             &flat_size)) {
    return loom_target_function_contract_reject_with_target(
        diagnostic_emitter, func_facts, target, 49, out_valid);
  }
  if (max_flat_workgroup_size != 0 && flat_size > max_flat_workgroup_size) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(func_facts->name),
        loom_param_string(target->name),
        loom_param_u64(flat_size),
        loom_param_u32(max_flat_workgroup_size),
    };
    return loom_target_function_contract_reject(
        diagnostic_emitter, func_facts, 50, params, IREE_ARRAYSIZE(params),
        out_valid);
  }
  if (flat_min != 0 && (flat_size < flat_min || flat_size > flat_max)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(func_facts->name),
        loom_param_string(target->name),
        loom_param_u64(flat_size),
        loom_param_u32(flat_min),
        loom_param_u32(flat_max),
    };
    return loom_target_function_contract_reject(
        diagnostic_emitter, func_facts, 51, params, IREE_ARRAYSIZE(params),
        out_valid);
  }
  return iree_ok_status();
}

iree_status_t loom_target_function_contract_resolve(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_target_bundle_storage_t* out_bundle_storage) {
  memset(out_bundle_storage, 0, sizeof(*out_bundle_storage));
  *out_valid = false;

  const loom_target_symbol_facts_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_lookup_target(
      module, fact_table, func_facts, diagnostic_emitter, out_valid, &target));
  if (!*out_valid) {
    return iree_ok_status();
  }
  *out_bundle_storage = target->storage;
  loom_target_bundle_storage_rebind(out_bundle_storage);

  out_bundle_storage->export_plan.name = func_facts->name;
  out_bundle_storage->export_plan.export_symbol = iree_string_view_empty();
  if (func_facts->has_abi) {
    if (func_facts->is_kernel_entry) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(func_facts->name),
      };
      return loom_target_function_contract_reject(
          diagnostic_emitter, func_facts, 40, params, IREE_ARRAYSIZE(params),
          out_valid);
    }
    out_bundle_storage->export_plan.abi_kind = func_facts->abi_kind;
  }
  if (out_bundle_storage->export_plan.abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
    return loom_target_function_contract_reject_with_target(
        diagnostic_emitter, func_facts, target, 41, out_valid);
  }
  IREE_RETURN_IF_ERROR(loom_target_function_contract_apply_abi_attrs(
      module, func_facts, diagnostic_emitter, func_facts->abi_attrs,
      out_valid));
  if (!*out_valid) {
    return iree_ok_status();
  }
  if (func_facts->is_kernel_entry &&
      out_bundle_storage->export_plan.abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_target_function_contract_reject_with_target(
        diagnostic_emitter, func_facts, target, 42, out_valid);
  }
  if (func_facts->has_required_workgroup_size) {
    out_bundle_storage->export_plan.hal_kernel.required_workgroup_size =
        func_facts->required_workgroup_size;
  }
  if (out_bundle_storage->export_plan.abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    if (out_bundle_storage->export_plan.hal_kernel.binding_alignment == 0) {
      return loom_target_function_contract_reject_with_target(
          diagnostic_emitter, func_facts, target, 43, out_valid);
    }
    IREE_RETURN_IF_ERROR(loom_target_function_contract_validate_hal_kernel(
        diagnostic_emitter, func_facts, target, &out_bundle_storage->snapshot,
        &out_bundle_storage->export_plan.hal_kernel, out_valid));
    if (!*out_valid) {
      return iree_ok_status();
    }
  }
  if (!iree_string_view_is_empty(func_facts->export_symbol)) {
    out_bundle_storage->export_plan.export_symbol = func_facts->export_symbol;
  }
  if (func_facts->has_export_linkage) {
    out_bundle_storage->export_plan.linkage = func_facts->export_linkage;
  }
  *out_valid = true;
  return iree_ok_status();
}
