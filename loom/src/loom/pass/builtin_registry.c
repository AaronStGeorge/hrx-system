// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/builtin_registry.h"

#include <stdint.h>

#include "loom/codegen/low/allocation_pass.h"
#include "loom/transforms/branch_fusion.h"
#include "loom/transforms/branch_sink.h"
#include "loom/transforms/canonicalize.h"
#include "loom/transforms/cfg_simplify.h"
#include "loom/transforms/cse.h"
#include "loom/transforms/dce.h"
#include "loom/transforms/kernel_async_legality.h"
#include "loom/transforms/kernel_resources.h"
#include "loom/transforms/licm.h"
#include "loom/transforms/loop_fusion.h"
#include "loom/transforms/refine_boundaries.h"
#include "loom/transforms/scf_to_cfg.h"
#include "loom/transforms/strip_hints.h"
#include "loom/transforms/symbol_dce.h"
#include "loom/transforms/vector_memory_footprint.h"
#include "loom/transforms/vector_to_scalar.h"

static const loom_pass_option_schema_t kCanonicalizeOptionSchema[] = {
    {
        .name = IREE_SVL("max-iterations"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 1,
        .maximum_uint32 = UINT32_MAX,
    },
};

static const loom_pass_option_enum_value_t
    kLowMaterializeAllocationDiagnosticsValues[] = {
        {.value = IREE_SVL("none")},
        {.value = IREE_SVL("spills")},
};

static const loom_pass_option_schema_t kLowMaterializeAllocationOptionSchema[] =
    {
        {
            .name = IREE_SVL("budgets"),
            .kind = LOOM_PASS_OPTION_SCHEMA_STRING,
        },
        {
            .name = IREE_SVL("diagnostics"),
            .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
            .enum_values = kLowMaterializeAllocationDiagnosticsValues,
            .enum_value_count =
                IREE_ARRAYSIZE(kLowMaterializeAllocationDiagnosticsValues),
        },
};

static const loom_pass_option_schema_t kRefineBoundariesOptionSchema[] = {
    {
        .name = IREE_SVL("max-iterations"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 1,
        .maximum_uint32 = UINT32_MAX,
    },
};

static const loom_pass_descriptor_t kBuiltinPassDescriptors[] = {
    {
        .key = IREE_SVL("branch-fusion"),
        .info = loom_branch_fusion_pass_info,
        .function_run = loom_branch_fusion_run,
    },
    {
        .key = IREE_SVL("branch-sink"),
        .info = loom_branch_sink_pass_info,
        .function_run = loom_branch_sink_run,
    },
    {
        .key = IREE_SVL("canonicalize"),
        .info = loom_canonicalize_pass_info,
        .function_run = loom_canonicalize_run,
        .create = loom_canonicalize_create,
        .option_schema = kCanonicalizeOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kCanonicalizeOptionSchema),
    },
    {
        .key = IREE_SVL("cfg-simplify"),
        .info = loom_cfg_simplify_pass_info,
        .function_run = loom_cfg_simplify_run,
    },
    {
        .key = IREE_SVL("cse"),
        .info = loom_cse_pass_info,
        .function_run = loom_cse_run,
    },
    {
        .key = IREE_SVL("dce"),
        .info = loom_dce_pass_info,
        .function_run = loom_dce_run,
    },
    {
        .key = IREE_SVL("kernel-async-legality"),
        .info = loom_kernel_async_legality_pass_info,
        .function_run = loom_kernel_async_legality_run,
    },
    {
        .key = IREE_SVL("licm"),
        .info = loom_licm_pass_info,
        .function_run = loom_licm_run,
    },
    {
        .key = IREE_SVL("loop-fusion"),
        .info = loom_loop_fusion_pass_info,
        .function_run = loom_loop_fusion_run,
    },
    {
        .key = IREE_SVL("low-materialize-allocation"),
        .info = loom_low_materialize_allocation_pass_info,
        .function_run = loom_low_materialize_allocation_run,
        .create = loom_low_materialize_allocation_create,
        .option_schema = kLowMaterializeAllocationOptionSchema,
        .option_schema_count =
            IREE_ARRAYSIZE(kLowMaterializeAllocationOptionSchema),
    },
    {
        .key = IREE_SVL("normalize-kernel-resources"),
        .info = loom_normalize_kernel_resources_pass_info,
        .function_run = loom_normalize_kernel_resources_run,
    },
    {
        .key = IREE_SVL("refine-boundaries"),
        .info = loom_refine_boundaries_pass_info,
        .module_run = loom_refine_boundaries_run,
        .create = loom_refine_boundaries_create,
        .option_schema = kRefineBoundariesOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kRefineBoundariesOptionSchema),
    },
    {
        .key = IREE_SVL("scf-to-cfg"),
        .info = loom_scf_to_cfg_pass_info,
        .function_run = loom_scf_to_cfg_run,
    },
    {
        .key = IREE_SVL("strip-hints"),
        .info = loom_strip_hints_pass_info,
        .function_run = loom_strip_hints_run,
    },
    {
        .key = IREE_SVL("symbol-dce"),
        .info = loom_symbol_dce_pass_info,
        .module_run = loom_symbol_dce_run,
    },
    {
        .key = IREE_SVL("vector-memory-footprint"),
        .info = loom_vector_memory_footprint_pass_info,
        .function_run = loom_vector_memory_footprint_run,
    },
    {
        .key = IREE_SVL("vector-to-scalar"),
        .info = loom_vector_to_scalar_pass_info,
        .function_run = loom_vector_to_scalar_run,
    },
};

static const loom_pass_registry_t kBuiltinPassRegistry = {
    .descriptors = kBuiltinPassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kBuiltinPassDescriptors),
};

const loom_pass_registry_t* loom_pass_builtin_registry(void) {
  return &kBuiltinPassRegistry;
}
