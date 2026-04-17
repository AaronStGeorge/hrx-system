// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-neutral LLVMIR test target fixtures.
//
// These records deliberately do not model a real ISA, feature catalog, ABI
// extension, or target-provider contract. Core LLVMIR tests and loom-check
// default emission use them when the behavior under test should not select a
// real target package.

#ifndef LOOM_TARGET_LLVMIR_TEST_TARGET_H_
#define LOOM_TARGET_LLVMIR_TEST_TARGET_H_

#include "loom/target/emit/llvmir/target_env.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_llvmir_target_env_t* loom_llvmir_target_env_test_object(void);

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_test_object(
    void);

const loom_target_bundle_t* loom_llvmir_target_bundle_test_object(void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TEST_TARGET_H_
