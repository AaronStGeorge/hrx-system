// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel_resources.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_KERNEL_RESOURCES_STAT_FUNCTIONS_NORMALIZED = 0,
  LOOM_KERNEL_RESOURCES_STAT_VIEW_ARGS_NORMALIZED = 1,
};

static const loom_pass_statistic_def_t kKernelResourcesStatistics[] = {
    {IREE_SVL("functions-normalized"),
     IREE_SVL("Number of exported device functions changed.")},
    {IREE_SVL("view-args-normalized"),
     IREE_SVL("Number of view-typed ABI arguments rewritten to buffer roots.")},
};

static const loom_pass_info_t loom_kernel_resources_pass_info_storage = {
    .name = IREE_SVL("normalize-kernel-resources"),
    .description =
        IREE_SVL("Normalize exported device resource args to buffer roots."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kKernelResourcesStatistics,
    .statistic_count = IREE_ARRAYSIZE(kKernelResourcesStatistics),
};

const loom_pass_info_t* loom_normalize_kernel_resources_pass_info(void) {
  return &loom_kernel_resources_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

static bool loom_kernel_resources_is_exported_device_function(
    loom_func_like_t function) {
  return loom_func_like_isa(function) &&
         loom_func_like_visibility(function) == LOOM_FUNC_VISIBILITY_PUBLIC &&
         loom_func_like_cc(function) == LOOM_FUNC_CC_DEVICE &&
         loom_func_like_body(function) != NULL;
}

static iree_status_t loom_kernel_resources_build_zero_offset(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t* out_zero_offset) {
  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      location, &zero_op));
  *out_zero_offset = loom_index_constant_result(zero_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_resources_normalize_view_arg(
    loom_pass_t* pass, loom_builder_t* builder, loom_module_t* module,
    loom_value_id_t arg_id, loom_value_id_t zero_offset, loom_type_t view_type,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(module, arg_id, loom_type_buffer()));

  loom_op_t* view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(builder, arg_id, zero_offset,
                                              view_type, location, &view_op));
  loom_value_id_t view_id = loom_buffer_view_result(view_op);
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_except(module, arg_id, view_id, view_op));
  if (pass->statistics) {
    loom_pass_statistic_add(pass,
                            LOOM_KERNEL_RESOURCES_STAT_VIEW_ARGS_NORMALIZED, 1);
  }
  return iree_ok_status();
}

iree_status_t loom_normalize_kernel_resources_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function) {
  if (!loom_kernel_resources_is_exported_device_function(function)) {
    return iree_ok_status();
  }

  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = loom_func_like_arg_ids(function, &arg_count);
  if (arg_count == 0 || !arg_ids) return iree_ok_status();

  loom_region_t* body = loom_func_like_body(function);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  if (entry_block->first_op) {
    loom_builder_set_before(&builder, entry_block->first_op);
  } else {
    builder.ip.parent_op = function.op;
  }

  bool changed_function = false;
  loom_value_id_t zero_offset = LOOM_VALUE_ID_INVALID;
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_value_id_t arg_id = arg_ids[i];
    if (arg_id == LOOM_VALUE_ID_INVALID || arg_id >= module->values.count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "function argument %u has invalid value id %%%u",
                              (unsigned)i, (unsigned)arg_id);
    }
    loom_type_t arg_type = loom_module_value_type(module, arg_id);
    if (!loom_type_is_view(arg_type)) continue;
    if (zero_offset == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_kernel_resources_build_zero_offset(
          &builder, function.op->location, &zero_offset));
    }
    IREE_RETURN_IF_ERROR(loom_kernel_resources_normalize_view_arg(
        pass, &builder, module, arg_id, zero_offset, arg_type,
        function.op->location));
    changed_function = true;
  }

  if (changed_function && pass->statistics) {
    loom_pass_statistic_add(pass,
                            LOOM_KERNEL_RESOURCES_STAT_FUNCTIONS_NORMALIZED, 1);
  }
  return iree_ok_status();
}
