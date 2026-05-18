// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Read-only Loom module and attribute queries used by iree-tune-loom.

#ifndef LOOM_TOOLS_IREE_TUNE_LOOM_MODULE_QUERY_H_
#define LOOM_TOOLS_IREE_TUNE_LOOM_MODULE_QUERY_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-tune-loom/options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a borrowed module string or an empty view when |string_id| is
// invalid.
iree_string_view_t iree_tune_loom_module_string(const loom_module_t* module,
                                                loom_string_id_t string_id);

// Returns a borrowed value name or an empty view when unnamed/invalid.
iree_string_view_t iree_tune_loom_value_name(const loom_module_t* module,
                                             loom_value_id_t value_id);

// Trims a symbol name and removes one leading '@' when present.
iree_string_view_t iree_tune_loom_normalize_symbol_name(
    iree_string_view_t symbol_name);

// Returns true when |case_plan| matches an empty or named selection.
bool iree_tune_loom_case_matches_selection(
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t selected_case_name);

// Returns true when |benchmark_plan| matches an empty or named selection.
bool iree_tune_loom_benchmark_matches_selection(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_string_view_t selected_benchmark_name);

// Resolves a symbol reference attribute to the target symbol name.
iree_status_t iree_tune_loom_module_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name);

// Finds a named attribute by decoded string name.
const loom_named_attr_t* iree_tune_loom_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name);

// Reads an optional i64 attribute from |attrs|.
iree_status_t iree_tune_loom_read_optional_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, int64_t default_value, int64_t* out_value);

// Reads an optional bool attribute from |attrs|.
iree_status_t iree_tune_loom_read_optional_bool_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool default_value, bool* out_value);

// Reads an i64 policy value from either CLI override or benchmark attribute.
iree_status_t iree_tune_loom_read_i64_policy_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, const iree_tune_loom_i32_flag_t* flag,
    int64_t* out_value);

// Reads a bool policy value from either CLI override or benchmark attribute.
iree_status_t iree_tune_loom_read_bool_policy_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, const iree_tune_loom_bool_flag_t* flag,
    bool* out_value);

// Reads an optional string attribute or returns |default_value|.
iree_status_t iree_tune_loom_read_optional_string_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, iree_string_view_t default_value,
    iree_string_view_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TUNE_LOOM_MODULE_QUERY_H_
