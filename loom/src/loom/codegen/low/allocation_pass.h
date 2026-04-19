// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass wrappers for low allocation and materialization.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_PASS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_PASS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_low_materialize_allocation_pass_info(void);

typedef struct loom_low_materialize_allocation_pass_config_t {
  // Selected target-low descriptor registry used to interpret low ops.
  const loom_low_descriptor_registry_t* descriptor_registry;
} loom_low_materialize_allocation_pass_config_t;

iree_status_t loom_low_materialize_allocation_create(
    loom_pass_t* pass, iree_string_view_t options);

iree_status_t loom_low_materialize_allocation_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_PASS_H_
