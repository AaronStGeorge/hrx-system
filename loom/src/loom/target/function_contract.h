// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-aware function contract materialization.
//
// Generic func symbol facts carry only function-local structure: target symbol,
// ABI/export syntax, kernel launch structure, and signature/import facts. This
// target layer resolves the referenced target.profile facts and overlays the
// func-like-owned contract onto the selected profile bundle for lowering,
// packaging, and execution.

#ifndef LOOM_TARGET_FUNCTION_CONTRACT_H_
#define LOOM_TARGET_FUNCTION_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves |func_facts|'s target profile and materializes the effective target
// bundle selected by the func-like symbol.
//
// The target snapshot and config come from target.profile facts. The export
// plan starts from the target profile defaults and is then overlaid with
// function-owned ABI/export attrs or kernel-owned launch/export attrs.
// |out_bundle_storage| owns the copied payload fields and its embedded bundle
// points at those copies. Returns status only for infrastructure failures.
// Invalid user IR emits a structured diagnostic, sets |out_valid| to false,
// and returns OK.
iree_status_t loom_target_function_contract_resolve(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_target_bundle_storage_t* out_bundle_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FUNCTION_CONTRACT_H_
