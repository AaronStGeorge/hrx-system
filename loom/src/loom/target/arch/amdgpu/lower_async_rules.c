// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/kernel/ops.h"

enum {
  LOOM_AMDGPU_ASYNC_VALUE_REF_GROUP_RESULT = 0,
};

enum {
  LOOM_AMDGPU_ASYNC_DIAGNOSTIC_WAIT_GROUP_COUNT = 0,
};

static const loom_low_lower_value_ref_t kAmdgpuAsyncValueRefs[] = {
    [LOOM_AMDGPU_ASYNC_VALUE_REF_GROUP_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
};

static const loom_low_lower_diagnostic_t kAmdgpuAsyncDiagnostics[] = {
    [LOOM_AMDGPU_ASYNC_DIAGNOSTIC_WAIT_GROUP_COUNT] =
        {
            .subject_kind = IREE_SVL("async"),
            .subject_name = IREE_SVL("wait"),
            .reason = IREE_SVL("AMDGPU source-to-low currently supports only "
                               "newer_groups = 0 async waits"),
        },
};

static const loom_low_lower_guard_t kAmdgpuAsyncGuards[] = {
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE,
        .attr_index = 0,
        .diagnostic_index = LOOM_AMDGPU_ASYNC_DIAGNOSTIC_WAIT_GROUP_COUNT,
        .minimum_i64 = 0,
        .maximum_i64 = 0,
    },
};

static const loom_low_lower_rule_t kAmdgpuAsyncRules[] = {
    {
        .source_op_kind = LOOM_OP_KERNEL_ASYNC_GROUP,
        .elide_ref_start = LOOM_AMDGPU_ASYNC_VALUE_REF_GROUP_RESULT,
        .elide_ref_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_KERNEL_ASYNC_WAIT,
        .guard_start = 0,
        .guard_count = 1,
    },
};

static const loom_low_lower_rule_span_t kAmdgpuAsyncRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_KERNEL_ASYNC_GROUP,
        .rule_start = 0,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_KERNEL_ASYNC_WAIT,
        .rule_start = 1,
        .rule_count = 1,
    },
};

const loom_low_lower_rule_set_t loom_amdgpu_async_rule_set = {
    .spans = kAmdgpuAsyncRuleSpans,
    .span_count = IREE_ARRAYSIZE(kAmdgpuAsyncRuleSpans),
    .rules = kAmdgpuAsyncRules,
    .rule_count = IREE_ARRAYSIZE(kAmdgpuAsyncRules),
    .value_refs = kAmdgpuAsyncValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kAmdgpuAsyncValueRefs),
    .guards = kAmdgpuAsyncGuards,
    .guard_count = IREE_ARRAYSIZE(kAmdgpuAsyncGuards),
    .diagnostics = kAmdgpuAsyncDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kAmdgpuAsyncDiagnostics),
};
