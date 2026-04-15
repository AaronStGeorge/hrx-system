// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared LLVM IR module fixtures for writer tests.
//
// These fixtures build representative modules through the public C builder API
// so every writer sink can exercise the same feature inventory. Production
// lowering code is C, so the fixture implementation intentionally uses the same
// designated-initializer style expected from real lowering code.

#ifndef LOOM_TARGET_LLVMIR_TEST_MODULES_H_
#define LOOM_TARGET_LLVMIR_TEST_MODULES_H_

#include "iree/base/api.h"
#include "loom/target/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_llvmir_test_module_scenario_e {
  // Host object-style vector add function with parameter attrs and
  // GEP/load/store.
  LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4 = 0,
  // Host function calling imported functions with literal constants.
  LOOM_LLVMIR_TEST_MODULE_CALL_CONSTANTS = 1,
  // CFG-shaped scalar function with conditional branches and a phi.
  LOOM_LLVMIR_TEST_MODULE_CFG_PHI = 2,
  // Function using structured inline asm.
  LOOM_LLVMIR_TEST_MODULE_INLINE_ASM = 3,
  // AMDGPU kernel boundary using attrs, metadata, and AMDGCN intrinsics.
  LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS = 4,
  // Scalar arithmetic function with a value-producing binary op.
  LOOM_LLVMIR_TEST_MODULE_SCALAR_BINOP = 5,
  // Host function using target-independent memory intrinsics.
  LOOM_LLVMIR_TEST_MODULE_BUILTIN_INTRINSICS = 6,
} loom_llvmir_test_module_scenario_t;

iree_host_size_t loom_llvmir_test_module_scenario_count(void);

iree_string_view_t loom_llvmir_test_module_scenario_name(
    loom_llvmir_test_module_scenario_t scenario);

iree_status_t loom_llvmir_test_module_build(
    loom_llvmir_test_module_scenario_t scenario, iree_allocator_t allocator,
    loom_llvmir_module_t** out_module);

iree_string_view_t loom_llvmir_test_module_expected_text(
    loom_llvmir_test_module_scenario_t scenario);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TEST_MODULES_H_
