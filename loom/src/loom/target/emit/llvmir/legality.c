// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/legality.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/emit/llvmir/target_env.h"
#include "loom/target/launch.h"

struct loom_llvmir_target_legality_context_t {
  // Source Loom module being checked.
  const loom_module_t* module;
  // Caller-owned options for this check.
  const loom_llvmir_target_legality_options_t* options;
  // Generic target bundle view over the option records.
  loom_target_bundle_t bundle;
  // Derived LLVMIR target profile proving the target records are coherent.
  loom_llvmir_target_profile_storage_t profile_storage;
  // Optional first-failure diagnostic output.
  loom_llvmir_target_legality_diagnostic_t* diagnostic;
};

static void loom_llvmir_target_legality_reset_diagnostic(
    loom_llvmir_target_legality_diagnostic_t* diagnostic) {
  if (!diagnostic) return;
  *diagnostic = (loom_llvmir_target_legality_diagnostic_t){
      .code = LOOM_LLVMIR_TARGET_LEGALITY_OK,
  };
}

static iree_status_code_t loom_llvmir_target_legality_status_code(
    loom_llvmir_target_legality_code_t code) {
  switch (code) {
    case LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET:
      return IREE_STATUS_INVALID_ARGUMENT;
    case LOOM_LLVMIR_TARGET_LEGALITY_OK:
      return IREE_STATUS_OK;
    case LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI:
    case LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_FUNCTION:
    case LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE:
    case LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_CONTROL_FLOW:
    case LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP:
    case LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC:
    case LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT:
      return IREE_STATUS_UNIMPLEMENTED;
  }
  return IREE_STATUS_INVALID_ARGUMENT;
}

static iree_string_view_t loom_llvmir_target_legality_provider_name(
    const loom_llvmir_target_legality_provider_t* provider) {
  return provider ? provider->name : iree_string_view_empty();
}

static iree_string_view_t loom_llvmir_target_legality_op_name(
    const loom_llvmir_target_legality_context_t* context, const loom_op_t* op) {
  if (!op) return iree_string_view_empty();
  return loom_op_name(context->module, op);
}

iree_status_t loom_llvmir_target_legality_fail(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_code_t code, const loom_op_t* op,
    iree_string_view_t detail, iree_string_view_t target_detail) {
  iree_string_view_t provider_name =
      loom_llvmir_target_legality_provider_name(provider);
  iree_string_view_t op_name = loom_llvmir_target_legality_op_name(context, op);
  if (context->diagnostic) {
    *context->diagnostic = (loom_llvmir_target_legality_diagnostic_t){
        .code = code,
        .provider_name = provider_name,
        .op_name = op_name,
        .detail = detail,
        .target_detail = target_detail,
    };
  }
  const char* target_separator =
      iree_string_view_is_empty(target_detail) ? "" : ": ";
  iree_status_code_t status_code =
      loom_llvmir_target_legality_status_code(code);
  if (!iree_string_view_is_empty(provider_name) &&
      !iree_string_view_is_empty(op_name)) {
    return iree_make_status(
        status_code,
        "LLVMIR target legality failed in provider %.*s for op %.*s: "
        "%.*s%s%.*s",
        (int)provider_name.size, provider_name.data, (int)op_name.size,
        op_name.data, (int)detail.size, detail.data, target_separator,
        (int)target_detail.size, target_detail.data);
  }
  if (!iree_string_view_is_empty(op_name)) {
    return iree_make_status(
        status_code, "LLVMIR target legality failed for op %.*s: %.*s%s%.*s",
        (int)op_name.size, op_name.data, (int)detail.size, detail.data,
        target_separator, (int)target_detail.size, target_detail.data);
  }
  if (!iree_string_view_is_empty(provider_name)) {
    return iree_make_status(
        status_code,
        "LLVMIR target legality failed in provider %.*s: %.*s%s%.*s",
        (int)provider_name.size, provider_name.data, (int)detail.size,
        detail.data, target_separator, (int)target_detail.size,
        target_detail.data);
  }
  return iree_make_status(status_code,
                          "LLVMIR target legality failed: %.*s%s%.*s",
                          (int)detail.size, detail.data, target_separator,
                          (int)target_detail.size, target_detail.data);
}

const loom_module_t* loom_llvmir_target_legality_module(
    const loom_llvmir_target_legality_context_t* context) {
  return context->module;
}

const loom_llvmir_target_profile_t* loom_llvmir_target_legality_profile(
    const loom_llvmir_target_legality_context_t* context) {
  return &context->profile_storage.profile;
}

iree_status_t loom_llvmir_target_legality_string_attr(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_string_view_t attr_name, loom_string_id_t string_id,
    iree_string_view_t* out_string) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= context->module->strings.count) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP, op,
        IREE_SV("string attribute references an invalid string id"), attr_name);
  }
  *out_string = context->module->strings.entries[string_id];
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_legality_expect_intrinsic_shape(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_count, iree_host_size_t result_count,
    iree_string_view_t detail) {
  if (op->operand_count != operand_count || op->result_count != result_count) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  if (op->tied_result_count > 0) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, IREE_SV("llvmir.intrinsic does not support tied results"),
        iree_string_view_empty());
  }
  return iree_ok_status();
}

static loom_type_t loom_llvmir_target_legality_value_type(
    loom_llvmir_target_legality_context_t* context, loom_value_id_t value_id) {
  if (value_id >= context->module->values.count) return loom_type_none();
  return loom_module_value_type(context->module, value_id);
}

iree_status_t loom_llvmir_target_legality_expect_scalar_result(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    loom_scalar_type_t expected_type, iree_string_view_t detail) {
  if (op->result_count != 1) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  loom_type_t result_type = loom_llvmir_target_legality_value_type(
      context, loom_op_const_results(op)[0]);
  if (!loom_type_is_scalar(result_type) ||
      loom_type_element_type(result_type) != expected_type) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_legality_expect_scalar_operand(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_ordinal, loom_scalar_type_t expected_type,
    iree_string_view_t detail) {
  if (operand_ordinal >= op->operand_count) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  loom_type_t operand_type = loom_llvmir_target_legality_value_type(
      context, loom_op_const_operands(op)[operand_ordinal]);
  if (!loom_type_is_scalar(operand_type) ||
      loom_type_element_type(operand_type) != expected_type) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  return iree_ok_status();
}

static bool loom_llvmir_target_legality_integer_bit_width(
    loom_llvmir_target_legality_context_t* context, loom_type_t type,
    uint32_t* out_bit_width) {
  if (!loom_type_is_scalar(type)) return false;
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
      *out_bit_width = context->options->snapshot->index_bitwidth;
      return true;
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_bit_width = context->options->snapshot->offset_bitwidth;
      return true;
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
      *out_bit_width =
          (uint32_t)loom_scalar_type_bitwidth(loom_type_element_type(type));
      return true;
    default:
      return false;
  }
}

iree_status_t loom_llvmir_target_legality_expect_integer_operand_bit_width(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_ordinal, iree_string_view_t detail,
    uint32_t* out_bit_width) {
  if (operand_ordinal >= op->operand_count) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  loom_type_t operand_type = loom_llvmir_target_legality_value_type(
      context, loom_op_const_operands(op)[operand_ordinal]);
  if (!loom_llvmir_target_legality_integer_bit_width(context, operand_type,
                                                     out_bit_width)) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  return iree_ok_status();
}

static bool loom_llvmir_target_legality_value_is_source_constant(
    loom_llvmir_target_legality_context_t* context, loom_value_id_t value_id) {
  if (value_id >= context->module->values.count) return false;
  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* def_op = loom_value_def_op(value);
  return def_op &&
         (loom_scalar_constant_isa(def_op) || loom_index_constant_isa(def_op));
}

iree_status_t loom_llvmir_target_legality_expect_constant_operand(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_ordinal, iree_string_view_t detail) {
  if (operand_ordinal >= op->operand_count ||
      !loom_llvmir_target_legality_value_is_source_constant(
          context, loom_op_const_operands(op)[operand_ordinal])) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, detail, iree_string_view_empty());
  }
  return iree_ok_status();
}

static bool loom_llvmir_target_legality_artifact_is_object(
    loom_target_artifact_format_t artifact_format) {
  switch (artifact_format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      return true;
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
    case LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY:
      return false;
  }
  return false;
}

static bool loom_llvmir_target_legality_linkage_is_supported(
    loom_target_linkage_t linkage) {
  switch (linkage) {
    case LOOM_TARGET_LINKAGE_DEFAULT:
    case LOOM_TARGET_LINKAGE_DSO_LOCAL:
      return true;
  }
  return false;
}

static bool loom_llvmir_target_legality_mul_u64_overflows(uint64_t lhs,
                                                          uint64_t rhs,
                                                          uint64_t* out_value) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) {
    *out_value = 0;
    return true;
  }
  *out_value = lhs * rhs;
  return false;
}

static bool loom_llvmir_target_legality_workgroup_flat_size(
    const loom_target_workgroup_size_t* size, uint64_t* out_flat_size) {
  uint64_t flat_size = 1;
  const uint64_t factors[] = {size->x, size->y, size->z};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(factors); ++i) {
    if (loom_llvmir_target_legality_mul_u64_overflows(flat_size, factors[i],
                                                      &flat_size)) {
      *out_flat_size = 0;
      return false;
    }
  }
  *out_flat_size = flat_size;
  return true;
}

static iree_status_t loom_llvmir_target_legality_verify_hal_kernel_launch(
    loom_llvmir_target_legality_context_t* context) {
  const loom_target_snapshot_t* snapshot = context->options->snapshot;
  const loom_target_hal_kernel_abi_t* hal_kernel =
      &context->options->export_plan->hal_kernel;
  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  if (context->options->profile->kind !=
      LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI, NULL,
        IREE_SV("LLVMIR target projection does not support HAL kernel ABI"),
        context->options->profile->name);
  }
  if (snapshot->memory_spaces.global == UINT32_MAX) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel target global address space is unavailable"),
        snapshot->name);
  }
  if (hal_kernel->binding_alignment == 0) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel binding alignment must be non-zero"),
        context->options->export_plan->name);
  }
  if (loom_target_workgroup_size_is_partial(required)) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel required workgroup size must be all zero or all "
                "non-zero"),
        context->options->export_plan->name);
  }

  const uint32_t flat_min = hal_kernel->flat_workgroup_size_min;
  const uint32_t flat_max = hal_kernel->flat_workgroup_size_max;
  const uint32_t max_flat = snapshot->max_flat_workgroup_size;
  if ((flat_min == 0) != (flat_max == 0)) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV(
            "HAL kernel flat workgroup size range must be both zero or both "
            "non-zero"),
        context->options->export_plan->name);
  }
  if (flat_min != 0) {
    if (flat_min > flat_max) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
          IREE_SV("HAL kernel flat workgroup size range must be ordered"),
          context->options->export_plan->name);
    }
    if (max_flat != 0 && flat_max > max_flat) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
          IREE_SV("HAL kernel flat workgroup size max exceeds target limit"),
          snapshot->name);
    }
  } else if (max_flat == 0) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("LLVMIR HAL kernel requires a flat workgroup size limit"),
        snapshot->name);
  }

  if (loom_target_workgroup_size_is_empty(required)) {
    return iree_ok_status();
  }
  if (snapshot->max_workgroup_size.x != 0 &&
      required->x > snapshot->max_workgroup_size.x) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel required workgroup x size exceeds target limit"),
        snapshot->name);
  }
  if (snapshot->max_workgroup_size.y != 0 &&
      required->y > snapshot->max_workgroup_size.y) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel required workgroup y size exceeds target limit"),
        snapshot->name);
  }
  if (snapshot->max_workgroup_size.z != 0 &&
      required->z > snapshot->max_workgroup_size.z) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel required workgroup z size exceeds target limit"),
        snapshot->name);
  }
  uint64_t flat_size = 0;
  if (!loom_llvmir_target_legality_workgroup_flat_size(required, &flat_size)) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel required flat workgroup size overflows uint64"),
        context->options->export_plan->name);
  }
  if (max_flat != 0 && flat_size > max_flat) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("HAL kernel required flat workgroup size exceeds target limit"),
        snapshot->name);
  }
  if (flat_min != 0 && (flat_size < flat_min || flat_size > flat_max)) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV(
            "HAL kernel required flat workgroup size must be inside selected "
            "range"),
        context->options->export_plan->name);
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_validate_options(
    loom_llvmir_target_legality_context_t* context) {
  const loom_llvmir_target_legality_options_t* options = context->options;
  if (options->snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_LLVMIR) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("target snapshot is not an LLVMIR codegen target"),
        options->snapshot->name);
  }
  if (!loom_llvmir_target_legality_artifact_is_object(
          options->snapshot->artifact_format)) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("target artifact format is not an LLVM object"),
        loom_target_artifact_format_name(options->snapshot->artifact_format));
  }
  if (options->snapshot->default_pointer_bitwidth == 0 ||
      options->snapshot->index_bitwidth == 0 ||
      options->snapshot->offset_bitwidth == 0) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV(
            "target pointer, index, and offset bit widths must be non-zero"),
        options->snapshot->name);
  }
  if (options->snapshot->memory_spaces.generic == UINT32_MAX) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET, NULL,
        IREE_SV("target generic pointer address space is unavailable"),
        options->snapshot->name);
  }
  if (!loom_llvmir_target_legality_linkage_is_supported(
          options->export_plan->linkage)) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI, NULL,
        IREE_SV("target linkage has no LLVMIR mapping"),
        loom_target_linkage_name(options->export_plan->linkage));
  }
  if (options->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_target_legality_verify_hal_kernel_launch(context));
  } else if (options->export_plan->abi_kind !=
             LOOM_TARGET_ABI_OBJECT_FUNCTION) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI, NULL,
        IREE_SV("target ABI does not have an LLVMIR legality adapter"),
        options->export_plan->name);
  }
  context->bundle = (loom_target_bundle_t){
      .name = options->snapshot->name,
      .snapshot = options->snapshot,
      .export_plan = options->export_plan,
      .config = options->config,
  };
  loom_llvmir_target_profile_storage_initialize_from_bundle(
      &context->bundle, options->profile, &context->profile_storage);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_verify_scalar_type(
    loom_llvmir_target_legality_context_t* context, loom_type_t type) {
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE, NULL,
          IREE_SV("FP8 scalar types require explicit decode or target contract "
                  "lowering before LLVMIR"),
          IREE_SV("fp8"));
    case LOOM_SCALAR_TYPE_COUNT_:
      break;
  }
  return loom_llvmir_target_legality_fail(
      context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE, NULL,
      IREE_SV("unknown scalar type"), iree_string_view_empty());
}

static iree_status_t loom_llvmir_target_legality_verify_type(
    loom_llvmir_target_legality_context_t* context, loom_type_t type) {
  if (loom_type_is_scalar(type)) {
    return loom_llvmir_target_legality_verify_scalar_type(context, type);
  }
  if (loom_type_is_buffer(type) || loom_type_is_view(type)) {
    return iree_ok_status();
  }
  if (loom_type_is_vector(type)) {
    if (!loom_type_is_all_static(type) || loom_type_rank(type) != 1) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE, NULL,
          IREE_SV("only static one-dimensional vectors lower to LLVMIR today"),
          IREE_SV("vector"));
    }
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(type, &element_count) ||
        element_count > UINT32_MAX) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE, NULL,
          IREE_SV("vector lane count is not representable in LLVMIR"),
          IREE_SV("vector"));
    }
    return loom_llvmir_target_legality_verify_scalar_type(
        context, loom_type_scalar(loom_type_element_type(type)));
  }
  return loom_llvmir_target_legality_fail(
      context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE, NULL,
      IREE_SV("type has no LLVMIR mapping yet"), iree_string_view_empty());
}

static iree_status_t loom_llvmir_target_legality_verify_value_types(
    loom_llvmir_target_legality_context_t* context) {
  for (iree_host_size_t i = 0; i < context->module->values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_verify_type(
        context, context->module->values.entries[i].type));
  }
  return iree_ok_status();
}

static bool loom_llvmir_target_legality_type_is_pointer_like(
    loom_llvmir_target_legality_context_t* context, loom_value_id_t value_id) {
  loom_type_t type = loom_llvmir_target_legality_value_type(context, value_id);
  return loom_type_is_buffer(type) || loom_type_is_view(type);
}

static iree_status_t loom_llvmir_target_legality_expect_pointer_operand(
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    iree_host_size_t operand_ordinal, iree_string_view_t detail) {
  if (operand_ordinal >= op->operand_count ||
      !loom_llvmir_target_legality_type_is_pointer_like(
          context, loom_op_const_operands(op)[operand_ordinal])) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC, op,
        detail, iree_string_view_empty());
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_verify_builtin_intrinsic(
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t kind, bool* out_handled) {
  *out_handled = true;
  if (iree_string_view_equal(kind, IREE_SV("llvm.memcpy"))) {
    iree_string_view_t detail = IREE_SV(
        "llvm.memcpy expects (ptr target, ptr source, integer length, i1 "
        "constant volatile) -> ()");
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_intrinsic_shape(
        context, NULL, op, 4, 0, detail));
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_pointer_operand(
        context, op, 0, detail));
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_pointer_operand(
        context, op, 1, detail));
    uint32_t ignored_bit_width = 0;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_target_legality_expect_integer_operand_bit_width(
            context, NULL, op, 2, detail, &ignored_bit_width));
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_scalar_operand(
        context, NULL, op, 3, LOOM_SCALAR_TYPE_I1, detail));
    return loom_llvmir_target_legality_expect_constant_operand(context, NULL,
                                                               op, 3, detail);
  }
  if (iree_string_view_equal(kind, IREE_SV("llvm.memset"))) {
    iree_string_view_t detail = IREE_SV(
        "llvm.memset expects (ptr target, i8 value, integer length, i1 "
        "constant volatile) -> ()");
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_intrinsic_shape(
        context, NULL, op, 4, 0, detail));
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_pointer_operand(
        context, op, 0, detail));
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_scalar_operand(
        context, NULL, op, 1, LOOM_SCALAR_TYPE_I8, detail));
    uint32_t ignored_bit_width = 0;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_target_legality_expect_integer_operand_bit_width(
            context, NULL, op, 2, detail, &ignored_bit_width));
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_scalar_operand(
        context, NULL, op, 3, LOOM_SCALAR_TYPE_I1, detail));
    return loom_llvmir_target_legality_expect_constant_operand(context, NULL,
                                                               op, 3, detail);
  }
  if (iree_string_view_equal(kind, IREE_SV("llvm.lifetime.start")) ||
      iree_string_view_equal(kind, IREE_SV("llvm.lifetime.end"))) {
    iree_string_view_t detail = IREE_SV(
        "llvm.lifetime intrinsic expects (i64 constant size, ptr) -> ()");
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_intrinsic_shape(
        context, NULL, op, 2, 0, detail));
    uint32_t size_bit_width = 0;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_target_legality_expect_integer_operand_bit_width(
            context, NULL, op, 0, detail, &size_bit_width));
    if (size_bit_width != 64) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC, op,
          detail, iree_string_view_empty());
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_constant_operand(
        context, NULL, op, 0, detail));
    return loom_llvmir_target_legality_expect_pointer_operand(context, op, 1,
                                                              detail);
  }
  *out_handled = false;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_try_provider_op(
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  for (iree_host_size_t i = 0; i < context->options->provider_count; ++i) {
    const loom_llvmir_target_legality_provider_t* provider =
        context->options->providers[i];
    bool handled = false;
    IREE_RETURN_IF_ERROR(
        provider->try_verify_op(provider, context, op, &handled));
    if (handled) {
      *out_handled = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_verify_intrinsic(
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op) {
  loom_string_id_t kind_id = loom_llvmir_intrinsic_kind(op);
  iree_string_view_t kind = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_string_attr(
      context, NULL, op, IREE_SV("kind"), kind_id, &kind));
  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_verify_builtin_intrinsic(
      context, op, kind, &handled));
  if (handled) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_llvmir_target_legality_try_provider_op(context, op, &handled));
  if (handled) return iree_ok_status();
  return loom_llvmir_target_legality_fail(
      context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC, op,
      IREE_SV("unknown llvmir.intrinsic kind"), kind);
}

static iree_status_t loom_llvmir_target_legality_verify_inline_asm(
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op) {
  if (op->result_count > 1) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP, op,
        IREE_SV("inline asm supports at most one direct result"),
        iree_string_view_empty());
  }
  if (op->tied_result_count > 0) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP, op,
        IREE_SV("inline asm does not support tied results"),
        iree_string_view_empty());
  }
  uint8_t supported_flags = LOOM_LLVMIR_ASMFLAGS_SIDEEFFECT |
                            LOOM_LLVMIR_ASMFLAGS_ALIGNSTACK |
                            LOOM_LLVMIR_ASMFLAGS_INTELDIALECT;
  if ((loom_llvmir_inline_asm_flags(op) & ~supported_flags) != 0) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP, op,
        IREE_SV("inline asm has unknown instance flags"),
        iree_string_view_empty());
  }
  iree_string_view_t ignored = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_string_attr(
      context, NULL, op, IREE_SV("asm_template"),
      loom_llvmir_inline_asm_asm_template(op), &ignored));
  return loom_llvmir_target_legality_string_attr(
      context, NULL, op, IREE_SV("constraints"),
      loom_llvmir_inline_asm_constraints(op), &ignored);
}

static iree_status_t loom_llvmir_target_legality_verify_provider_contract_op(
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t unsupported_detail) {
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_target_legality_try_provider_op(context, op, &handled));
  if (handled) return iree_ok_status();
  return loom_llvmir_target_legality_fail(
      context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT,
      op, unsupported_detail, iree_string_view_empty());
}

static bool loom_llvmir_target_legality_op_is_supported_core(
    const loom_llvmir_target_legality_context_t* context, const loom_op_t* op) {
  if (loom_traits_are_fact_identity(
          loom_op_effective_traits(context->module, op))) {
    return true;
  }
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_SCALAR_DIVSI:
    case LOOM_OP_SCALAR_DIVUI:
    case LOOM_OP_SCALAR_REMSI:
    case LOOM_OP_SCALAR_REMUI:
    case LOOM_OP_SCALAR_ADDF:
    case LOOM_OP_SCALAR_SUBF:
    case LOOM_OP_SCALAR_MULF:
    case LOOM_OP_SCALAR_DIVF:
    case LOOM_OP_SCALAR_REMF:
    case LOOM_OP_SCALAR_ANDI:
    case LOOM_OP_SCALAR_ORI:
    case LOOM_OP_SCALAR_XORI:
    case LOOM_OP_SCALAR_SHLI:
    case LOOM_OP_SCALAR_SHRSI:
    case LOOM_OP_SCALAR_SHRUI:
    case LOOM_OP_SCALAR_NEGF:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_SCALAR_CMPI:
    case LOOM_OP_SCALAR_CMPF:
    case LOOM_OP_SCALAR_SITOFP:
    case LOOM_OP_SCALAR_UITOFP:
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTOUI:
    case LOOM_OP_SCALAR_EXTF:
    case LOOM_OP_SCALAR_FPTRUNC:
    case LOOM_OP_SCALAR_EXTSI:
    case LOOM_OP_SCALAR_EXTUI:
    case LOOM_OP_SCALAR_TRUNCI:
    case LOOM_OP_SCALAR_BITCAST:
    case LOOM_OP_INDEX_ADD:
    case LOOM_OP_INDEX_SUB:
    case LOOM_OP_INDEX_MUL:
    case LOOM_OP_INDEX_DIV:
    case LOOM_OP_INDEX_REM:
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_INDEX_MADD:
    case LOOM_OP_INDEX_MIN:
    case LOOM_OP_INDEX_MAX:
    case LOOM_OP_INDEX_CMP:
    case LOOM_OP_INDEX_CAST:
    case LOOM_OP_VECTOR_ADDF:
    case LOOM_OP_VECTOR_SUBF:
    case LOOM_OP_VECTOR_MULF:
    case LOOM_OP_VECTOR_DIVF:
    case LOOM_OP_VECTOR_REMF:
    case LOOM_OP_VECTOR_ADDI:
    case LOOM_OP_VECTOR_SUBI:
    case LOOM_OP_VECTOR_MULI:
    case LOOM_OP_VECTOR_DIVSI:
    case LOOM_OP_VECTOR_DIVUI:
    case LOOM_OP_VECTOR_REMSI:
    case LOOM_OP_VECTOR_REMUI:
    case LOOM_OP_VECTOR_ANDI:
    case LOOM_OP_VECTOR_ORI:
    case LOOM_OP_VECTOR_XORI:
    case LOOM_OP_VECTOR_SHLI:
    case LOOM_OP_VECTOR_SHRSI:
    case LOOM_OP_VECTOR_SHRUI:
    case LOOM_OP_VECTOR_NEGF:
    case LOOM_OP_VECTOR_CONSTANT:
    case LOOM_OP_VECTOR_POISON:
    case LOOM_OP_VECTOR_SPLAT:
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
    case LOOM_OP_VECTOR_EXTRACT:
    case LOOM_OP_VECTOR_INSERT:
    case LOOM_OP_VECTOR_SHUFFLE:
    case LOOM_OP_VECTOR_CMPI:
    case LOOM_OP_VECTOR_CMPF:
    case LOOM_OP_VECTOR_SELECT:
    case LOOM_OP_VECTOR_SITOFP:
    case LOOM_OP_VECTOR_UITOFP:
    case LOOM_OP_VECTOR_FPTOSI:
    case LOOM_OP_VECTOR_FPTOUI:
    case LOOM_OP_VECTOR_EXTF:
    case LOOM_OP_VECTOR_FPTRUNC:
    case LOOM_OP_VECTOR_EXTSI:
    case LOOM_OP_VECTOR_EXTUI:
    case LOOM_OP_VECTOR_TRUNCI:
    case LOOM_OP_VECTOR_BITCAST:
    case LOOM_OP_BUFFER_ALLOCA:
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
    case LOOM_OP_BUFFER_ASSUME_NOALIAS:
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_BUFFER_VIEW:
    case LOOM_OP_VIEW_SUBVIEW:
    case LOOM_OP_VIEW_LOAD:
    case LOOM_OP_VIEW_STORE:
    case LOOM_OP_VIEW_PREFETCH:
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE:
    case LOOM_OP_SCF_SELECT:
    case LOOM_OP_FUNC_CALL:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_target_legality_verify_op(
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_FUNC_DEF:
    case LOOM_OP_FUNC_DECL:
      return iree_ok_status();
    case LOOM_OP_TARGET_ARTIFACT:
    case LOOM_OP_TARGET_GENERIC:
      if (op->parent_op != NULL) {
        return loom_llvmir_target_legality_fail(
            context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP, op,
            IREE_SV("target record ops are module metadata and cannot appear "
                    "inside executable regions"),
            iree_string_view_empty());
      }
      return iree_ok_status();
    case LOOM_OP_LLVMIR_INLINE_ASM:
      return loom_llvmir_target_legality_verify_inline_asm(context, op);
    case LOOM_OP_LLVMIR_INTRINSIC:
      return loom_llvmir_target_legality_verify_intrinsic(context, op);
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I:
      return loom_llvmir_target_legality_verify_provider_contract_op(
          context, op,
          IREE_SV("vector dot op requires an explicit target legality "
                  "provider"));
    case LOOM_OP_VECTOR_DOT8I4:
      return loom_llvmir_target_legality_fail(
          context, NULL,
          LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT, op,
          IREE_SV("packed i4 dot needs explicit unpack/reference expansion or "
                  "target-contract lowering"),
          iree_string_view_empty());
    case LOOM_OP_VECTOR_DOT4F8:
      return loom_llvmir_target_legality_fail(
          context, NULL,
          LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT, op,
          IREE_SV("packed fp8/bf8 dot needs explicit decode/reference "
                  "expansion or target-contract lowering"),
          iree_string_view_empty());
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_CONTROL_FLOW,
          op,
          IREE_SV("structured SCF control flow must be lowered to CFG before "
                  "LLVMIR lowering"),
          iree_string_view_empty());
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_YIELD:
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_CONTROL_FLOW,
          op,
          IREE_SV("SCF terminators must be lowered with their parent "
                  "structured control-flow op before LLVMIR lowering"),
          iree_string_view_empty());
    default:
      break;
  }
  if (loom_llvmir_target_legality_op_is_supported_core(context, op)) {
    return iree_ok_status();
  }
  return loom_llvmir_target_legality_fail(
      context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP, op,
      IREE_SV("no LLVMIR lowering rule is registered"),
      iree_string_view_empty());
}

static iree_status_t loom_llvmir_target_legality_verify_region(
    loom_llvmir_target_legality_context_t* context,
    const loom_region_t* region) {
  if (!region) return iree_ok_status();
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_verify_op(context, op));
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_verify_region(
            context, loom_op_regions(op)[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_verify_function_body(
    loom_llvmir_target_legality_context_t* context, loom_func_like_t func) {
  loom_region_t* body = loom_func_like_body(func);
  if (!body) return iree_ok_status();
  if (iree_any_bit_set(body->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return iree_ok_status();
  }
  if (body->block_count != 1) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_CONTROL_FLOW,
        func.op,
        IREE_SV("multi-block function body must use CFG successor terminators "
                "before LLVMIR lowering"),
        iree_string_view_empty());
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_verify_function_abi(
    loom_llvmir_target_legality_context_t* context, loom_func_like_t func) {
  if (func.op->result_count > 1) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_FUNCTION,
        func.op,
        IREE_SV("LLVM functions with multiple direct results need an "
                "aggregate or sret ABI decision"),
        iree_string_view_empty());
  }
  if (context->profile_storage.profile.kind !=
      LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL) {
    return iree_ok_status();
  }
  bool is_public = loom_func_like_visibility(func) != 0;
  if (is_public && loom_func_like_cc(func) != LOOM_FUNC_CC_DEVICE) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI, func.op,
        IREE_SV("HAL kernel profile can only export public device entry "
                "points"),
        iree_string_view_empty());
  }
  if (!is_public) return iree_ok_status();
  if (!loom_func_like_body(func)) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI, func.op,
        IREE_SV("HAL kernel entry points must be function definitions"),
        iree_string_view_empty());
  }
  if (func.op->result_count != 0) {
    return loom_llvmir_target_legality_fail(
        context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI, func.op,
        IREE_SV("HAL kernel entry points must return void"),
        iree_string_view_empty());
  }
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = loom_func_like_arg_ids(func, &arg_count);
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_type_t arg_type =
        loom_llvmir_target_legality_value_type(context, arg_ids[i]);
    if (loom_type_is_view(arg_type)) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI, func.op,
          IREE_SV("HAL kernel entry point view parameters need explicit "
                  "resource materialization"),
          iree_string_view_empty());
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_legality_verify_functions(
    loom_llvmir_target_legality_context_t* context) {
  for (iree_host_size_t i = 0; i < context->module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &context->module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    iree_string_view_t symbol_name = iree_string_view_empty();
    if (symbol->name_id != LOOM_STRING_ID_INVALID &&
        symbol->name_id < context->module->strings.count) {
      symbol_name = context->module->strings.entries[symbol->name_id];
    }
    if (!symbol->defining_op) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_FUNCTION, NULL,
          IREE_SV("function-like symbol has no defining op"), symbol_name);
    }
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    if (bytecode_kind != LOOM_SYMBOL_FUNC_DEF &&
        bytecode_kind != LOOM_SYMBOL_FUNC_DECL) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_FUNCTION,
          symbol->defining_op,
          IREE_SV("LLVMIR lowering only supports func.def and func.decl "
                  "symbols today"),
          symbol_name);
    }
    loom_func_like_t func =
        loom_func_like_cast(context->module, symbol->defining_op);
    if (!loom_func_like_isa(func)) {
      return loom_llvmir_target_legality_fail(
          context, NULL, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_FUNCTION,
          symbol->defining_op,
          IREE_SV("function-like symbol does not implement FuncLike"),
          symbol_name);
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_target_legality_verify_function_abi(context, func));
    IREE_RETURN_IF_ERROR(
        loom_llvmir_target_legality_verify_function_body(context, func));
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_verify_target_legality(
    const loom_module_t* module,
    const loom_llvmir_target_legality_options_t* options,
    loom_llvmir_target_legality_diagnostic_t* out_diagnostic) {
  loom_llvmir_target_legality_reset_diagnostic(out_diagnostic);
  loom_llvmir_target_legality_context_t context = {
      .module = module,
      .options = options,
      .diagnostic = out_diagnostic,
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_validate_options(&context));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_target_legality_verify_value_types(&context));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_verify_functions(&context));
  return loom_llvmir_target_legality_verify_region(&context, module->body);
}
