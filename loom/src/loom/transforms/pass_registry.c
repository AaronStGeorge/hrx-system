// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pass_registry.h"

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

static bool loom_pass_descriptor_key_matches_info(
    const loom_pass_descriptor_t* descriptor) {
  if (!descriptor->info) return false;
  const loom_pass_info_t* info = descriptor->info();
  return info && iree_string_view_equal(descriptor->key, info->name);
}

iree_status_t loom_pass_registry_verify(const loom_pass_registry_t* registry) {
  if (!registry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass registry is required");
  }
  if (registry->descriptor_count > 0 && !registry->descriptors) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass registry descriptors are required");
  }
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    const loom_pass_descriptor_t* descriptor = &registry->descriptors[i];
    if (iree_string_view_is_empty(descriptor->key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor %zu has no key", i);
    }
    if (!descriptor->info) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor '%.*s' has no info",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    const loom_pass_info_t* info = descriptor->info();
    if (!info) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor '%.*s' returned no info",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    if (!loom_pass_descriptor_key_matches_info(descriptor)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor key '%.*s' does not match pass info name '%.*s'",
          (int)descriptor->key.size, descriptor->key.data, (int)info->name.size,
          info->name.data);
    }
    switch (info->kind) {
      case LOOM_PASS_MODULE:
        if (!descriptor->module_run) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "module pass descriptor '%.*s' has no run callback",
              (int)descriptor->key.size, descriptor->key.data);
        }
        break;
      case LOOM_PASS_FUNCTION:
        if (!descriptor->function_run) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "function pass descriptor '%.*s' has no run callback",
              (int)descriptor->key.size, descriptor->key.data);
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "pass descriptor '%.*s' has unknown kind %d",
                                (int)descriptor->key.size, descriptor->key.data,
                                (int)info->kind);
    }
    if (i > 0) {
      const loom_pass_descriptor_t* previous = &registry->descriptors[i - 1];
      int comparison = iree_string_view_compare(previous->key, descriptor->key);
      if (comparison >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass registry keys must be strictly sorted; '%.*s' precedes "
            "'%.*s'",
            (int)previous->key.size, previous->key.data,
            (int)descriptor->key.size, descriptor->key.data);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pass_registry_lookup(
    const loom_pass_registry_t* registry, iree_string_view_t key,
    const loom_pass_descriptor_t** out_descriptor) {
  if (!registry || !out_descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass registry and output descriptor are required");
  }
  *out_descriptor = NULL;
  iree_host_size_t low = 0;
  iree_host_size_t high = registry->descriptor_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const loom_pass_descriptor_t* descriptor = &registry->descriptors[mid];
    int comparison = iree_string_view_compare(key, descriptor->key);
    if (comparison == 0) {
      *out_descriptor = descriptor;
      return iree_ok_status();
    } else if (comparison < 0) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return iree_ok_status();
}

bool loom_pass_descriptor_is_available(
    const loom_pass_descriptor_t* descriptor) {
  return descriptor &&
         !iree_any_bit_set(descriptor->flags, LOOM_PASS_DESCRIPTOR_UNAVAILABLE);
}

iree_status_t loom_pass_manager_add_descriptor(
    loom_pass_manager_t* manager, const loom_pass_descriptor_t* descriptor,
    iree_string_view_t options, void* user_data) {
  if (!manager || !descriptor || !descriptor->info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass manager and descriptor with pass info are required");
  }
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (!loom_pass_descriptor_is_available(descriptor)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pass '%.*s' is unavailable: %.*s",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)descriptor->unavailable_reason.size,
                            descriptor->unavailable_reason.data);
  }
  if (info->kind == LOOM_PASS_MODULE) {
    return loom_pass_manager_add_module_pass(
        manager, info, descriptor->module_run, descriptor->create,
        descriptor->destroy, options, user_data);
  }
  if (info->kind == LOOM_PASS_FUNCTION) {
    return loom_pass_manager_add_function_pass(
        manager, info, descriptor->function_run, descriptor->create,
        descriptor->destroy, options, user_data);
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "pass '%.*s' has unknown kind %d",
      (int)descriptor->key.size, descriptor->key.data, (int)info->kind);
}
