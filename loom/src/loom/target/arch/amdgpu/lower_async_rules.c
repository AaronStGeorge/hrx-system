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

static const loom_low_lower_value_ref_t kAmdgpuAsyncValueRefs[] = {
    [LOOM_AMDGPU_ASYNC_VALUE_REF_GROUP_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
};

static const loom_low_lower_rule_t kAmdgpuAsyncRules[] = {
    {
        .source_op_kind = LOOM_OP_KERNEL_ASYNC_GROUP,
        .elide_ref_start = LOOM_AMDGPU_ASYNC_VALUE_REF_GROUP_RESULT,
        .elide_ref_count = 1,
    },
};

static const loom_low_lower_rule_span_t kAmdgpuAsyncRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_KERNEL_ASYNC_GROUP,
        .rule_start = 0,
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
};
