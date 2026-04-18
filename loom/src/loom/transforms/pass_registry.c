// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pass_registry.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

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

static bool loom_pass_descriptor_key_matches_info(
    const loom_pass_descriptor_t* descriptor) {
  if (!descriptor->info) return false;
  const loom_pass_info_t* info = descriptor->info();
  return info && iree_string_view_equal(descriptor->key, info->name);
}

static bool loom_pass_info_has_option_def(const loom_pass_info_t* info,
                                          iree_string_view_t option_name) {
  for (uint16_t i = 0; i < info->option_count; ++i) {
    if (iree_string_view_equal(info->option_defs[i].name, option_name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_pass_registry_verify_option_schema(
    const loom_pass_descriptor_t* descriptor, const loom_pass_info_t* info) {
  if (descriptor->option_schema_count != info->option_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' has %u option schemas but pass info declares "
        "%u options",
        (int)descriptor->key.size, descriptor->key.data,
        (unsigned)descriptor->option_schema_count,
        (unsigned)info->option_count);
  }
  if (info->option_count > 0 && !info->option_defs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' pass info declares options without option defs",
        (int)descriptor->key.size, descriptor->key.data);
  }
  if (descriptor->option_schema_count > 0 && !descriptor->option_schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' has no option schema",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (descriptor->option_schema_count > 0 && !descriptor->create) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' declares options but has no create callback",
        (int)descriptor->key.size, descriptor->key.data);
  }
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_string_view_is_empty(schema->name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' option schema %u has no name",
          (int)descriptor->key.size, descriptor->key.data, (unsigned)i);
    }
    if (!loom_pass_info_has_option_def(info, schema->name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' option schema '%.*s' has no matching pass "
          "info option",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data);
    }
    if (iree_any_bit_set(schema->flags, ~(loom_pass_option_schema_flags_t)
                                            LOOM_PASS_OPTION_SCHEMA_REQUIRED)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' option schema '%.*s' has unsupported flags "
          "0x%" PRIx32,
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, schema->flags);
    }
    if (i > 0) {
      const loom_pass_option_schema_t* previous =
          &descriptor->option_schema[i - 1];
      int comparison = iree_string_view_compare(previous->name, schema->name);
      if (comparison >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' option schemas must be strictly sorted; "
            "'%.*s' precedes '%.*s'",
            (int)descriptor->key.size, descriptor->key.data,
            (int)previous->name.size, previous->name.data,
            (int)schema->name.size, schema->name.data);
      }
    }
    switch (schema->kind) {
      case LOOM_PASS_OPTION_SCHEMA_STRING:
        if (schema->enum_values || schema->enum_value_count != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' string option schema '%.*s' has enum "
              "values",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        break;
      case LOOM_PASS_OPTION_SCHEMA_UINT32:
        if (schema->minimum_uint32 > schema->maximum_uint32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' uint32 option schema '%.*s' has invalid "
              "range",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        if (schema->enum_values || schema->enum_value_count != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' uint32 option schema '%.*s' has enum "
              "values",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        break;
      case LOOM_PASS_OPTION_SCHEMA_ENUM:
        if (schema->enum_value_count == 0 || !schema->enum_values) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' enum option schema '%.*s' has no values",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        for (uint16_t j = 0; j < schema->enum_value_count; ++j) {
          const loom_pass_option_enum_value_t* value = &schema->enum_values[j];
          if (iree_string_view_is_empty(value->value)) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "pass descriptor '%.*s' enum option schema '%.*s' value "
                "%u is empty",
                (int)descriptor->key.size, descriptor->key.data,
                (int)schema->name.size, schema->name.data, (unsigned)j);
          }
          if (j > 0) {
            const loom_pass_option_enum_value_t* previous =
                &schema->enum_values[j - 1];
            int comparison =
                iree_string_view_compare(previous->value, value->value);
            if (comparison >= 0) {
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "pass descriptor '%.*s' enum option schema '%.*s' values "
                  "must be strictly sorted; '%.*s' precedes '%.*s'",
                  (int)descriptor->key.size, descriptor->key.data,
                  (int)schema->name.size, schema->name.data,
                  (int)previous->value.size, previous->value.data,
                  (int)value->value.size, value->value.data);
            }
          }
        }
        break;
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' option schema '%.*s' has unknown kind %d",
            (int)descriptor->key.size, descriptor->key.data,
            (int)schema->name.size, schema->name.data, (int)schema->kind);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_registry_verify_requirements(
    const loom_pass_descriptor_t* descriptor) {
  if (descriptor->requirement_count > 0 && !descriptor->requirement_defs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' has no requirement defs",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  for (uint16_t i = 0; i < descriptor->requirement_count; ++i) {
    const loom_pass_requirement_def_t* requirement =
        &descriptor->requirement_defs[i];
    if (iree_string_view_is_empty(requirement->key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' requirement %u has no key",
          (int)descriptor->key.size, descriptor->key.data, (unsigned)i);
    }
    if (iree_string_view_is_empty(requirement->description)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' requirement '%.*s' has no description",
          (int)descriptor->key.size, descriptor->key.data,
          (int)requirement->key.size, requirement->key.data);
    }
    if (i > 0) {
      const loom_pass_requirement_def_t* previous =
          &descriptor->requirement_defs[i - 1];
      int comparison =
          iree_string_view_compare(previous->key, requirement->key);
      if (comparison >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' requirements must be strictly sorted; "
            "'%.*s' precedes '%.*s'",
            (int)descriptor->key.size, descriptor->key.data,
            (int)previous->key.size, previous->key.data,
            (int)requirement->key.size, requirement->key.data);
      }
    }
  }
  return iree_ok_status();
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
    IREE_RETURN_IF_ERROR(
        loom_pass_registry_verify_option_schema(descriptor, info));
    IREE_RETURN_IF_ERROR(loom_pass_registry_verify_requirements(descriptor));
    if (iree_any_bit_set(
            descriptor->flags,
            ~(loom_pass_descriptor_flags_t)LOOM_PASS_DESCRIPTOR_UNAVAILABLE)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' has unsupported flags 0x%" PRIx32,
          (int)descriptor->key.size, descriptor->key.data, descriptor->flags);
    }
    if (iree_any_bit_set(descriptor->flags, LOOM_PASS_DESCRIPTOR_UNAVAILABLE) &&
        iree_string_view_is_empty(descriptor->unavailable_reason)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' is unavailable without a reason",
          (int)descriptor->key.size, descriptor->key.data);
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

typedef struct loom_pass_option_validation_context_t {
  // Descriptor whose option schema is being applied.
  const loom_pass_descriptor_t* descriptor;
  // Seen bit per descriptor option schema entry.
  bool* seen_options;
} loom_pass_option_validation_context_t;

static const loom_pass_option_schema_t* loom_pass_descriptor_find_option_schema(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t option_name,
    uint16_t* out_schema_index) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_string_view_equal(schema->name, option_name)) {
      *out_schema_index = i;
      return schema;
    }
  }
  return NULL;
}

static bool loom_pass_option_schema_find_enum_value(
    const loom_pass_option_schema_t* schema, iree_string_view_t option_value,
    uint16_t* out_enum_value_index) {
  for (uint16_t i = 0; i < schema->enum_value_count; ++i) {
    if (iree_string_view_equal(schema->enum_values[i].value, option_value)) {
      *out_enum_value_index = i;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_pass_option_schema_decode_uint32(
    const loom_pass_descriptor_t* descriptor,
    const loom_pass_option_schema_t* schema, iree_string_view_t option_value,
    uint32_t* out_value) {
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
      descriptor->key, schema->name, option_value, &value));
  if (value < schema->minimum_uint32 || value > schema->maximum_uint32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' option '%.*s' must be in range %" PRIu32 "..%" PRIu32,
        (int)descriptor->key.size, descriptor->key.data, (int)schema->name.size,
        schema->name.data, schema->minimum_uint32, schema->maximum_uint32);
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_pass_option_schema_validate_assignment(
    void* user_data, iree_string_view_t option_name,
    iree_string_view_t option_value) {
  loom_pass_option_validation_context_t* context =
      (loom_pass_option_validation_context_t*)user_data;
  const loom_pass_descriptor_t* descriptor = context->descriptor;
  uint16_t schema_index = 0;
  const loom_pass_option_schema_t* schema =
      loom_pass_descriptor_find_option_schema(descriptor, option_name,
                                              &schema_index);
  if (!schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (context->seen_options[schema_index]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  context->seen_options[schema_index] = true;

  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      return iree_ok_status();
    case LOOM_PASS_OPTION_SCHEMA_UINT32: {
      uint32_t uint32_value = 0;
      return loom_pass_option_schema_decode_uint32(descriptor, schema,
                                                   option_value, &uint32_value);
    }
    case LOOM_PASS_OPTION_SCHEMA_ENUM: {
      uint16_t enum_value_index = 0;
      if (loom_pass_option_schema_find_enum_value(schema, option_value,
                                                  &enum_value_index)) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has invalid enum value '%.*s'",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)option_value.size,
          option_value.data);
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has unknown schema kind %d",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)schema->kind);
  }
}

static iree_status_t loom_pass_option_schema_validate_required(
    const loom_pass_descriptor_t* descriptor, const bool* seen_options) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_all_bits_set(schema->flags,
                          (loom_pass_option_schema_flags_t)
                              LOOM_PASS_OPTION_SCHEMA_REQUIRED) &&
        !seen_options[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing required option '%.*s' for pass '%.*s'",
                              (int)schema->name.size, schema->name.data,
                              (int)descriptor->key.size, descriptor->key.data);
    }
  }
  return iree_ok_status();
}

static bool loom_pass_option_schema_has_required(
    const loom_pass_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_all_bits_set(schema->flags,
                          (loom_pass_option_schema_flags_t)
                              LOOM_PASS_OPTION_SCHEMA_REQUIRED)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_pass_descriptor_validate_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options) {
  if (!descriptor || !descriptor->info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor with pass info is required for option validation");
  }
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  IREE_RETURN_IF_ERROR(
      loom_pass_registry_verify_option_schema(descriptor, info));
  iree_string_view_t trimmed_options = iree_string_view_trim(options);
  bool has_options = !iree_string_view_is_empty(trimmed_options);
  if (has_options && !descriptor->create) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options, got '{%.*s}'",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)options.size, options.data);
  }
  if (descriptor->option_schema_count == 0) {
    if (!has_options) return iree_ok_status();
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options, got '{%.*s}'",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)options.size, options.data);
  }
  if (!descriptor->option_schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' has no option schema",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (!has_options && !loom_pass_option_schema_has_required(descriptor)) {
    return iree_ok_status();
  }

  bool* seen_options = NULL;
  iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_allocator_malloc(
      allocator, descriptor->option_schema_count * sizeof(*seen_options),
      (void**)&seen_options);
  if (iree_status_is_ok(status)) {
    memset(seen_options, 0,
           descriptor->option_schema_count * sizeof(*seen_options));
  }

  loom_pass_option_validation_context_t context = {
      .descriptor = descriptor,
      .seen_options = seen_options,
  };
  if (iree_status_is_ok(status) && has_options) {
    status = loom_pass_options_parse(
        info->name, trimmed_options,
        loom_pass_option_schema_validate_assignment, &context);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_pass_option_schema_validate_required(descriptor, seen_options);
  }

  if (seen_options) {
    iree_allocator_free(allocator, seen_options);
  }
  return status;
}

typedef struct loom_pass_option_decode_context_t {
  // Descriptor whose option schema is being applied.
  const loom_pass_descriptor_t* descriptor;
  // Mutable decoded options indexed by descriptor option schema order.
  loom_pass_decoded_option_t* decoded_options;
} loom_pass_option_decode_context_t;

static iree_status_t loom_pass_option_schema_decode_assignment(
    void* user_data, iree_string_view_t option_name,
    iree_string_view_t option_value) {
  loom_pass_option_decode_context_t* context =
      (loom_pass_option_decode_context_t*)user_data;
  const loom_pass_descriptor_t* descriptor = context->descriptor;
  uint16_t schema_index = 0;
  const loom_pass_option_schema_t* schema =
      loom_pass_descriptor_find_option_schema(descriptor, option_name,
                                              &schema_index);
  if (!schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }

  loom_pass_decoded_option_t* decoded = &context->decoded_options[schema_index];
  if (decoded->present) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  decoded->present = true;

  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      decoded->string_value = option_value;
      return iree_ok_status();
    case LOOM_PASS_OPTION_SCHEMA_UINT32:
      return loom_pass_option_schema_decode_uint32(
          descriptor, schema, option_value, &decoded->uint32_value);
    case LOOM_PASS_OPTION_SCHEMA_ENUM:
      if (loom_pass_option_schema_find_enum_value(schema, option_value,
                                                  &decoded->enum_value_index)) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has invalid enum value '%.*s'",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)option_value.size,
          option_value.data);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has unknown schema kind %d",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)schema->kind);
  }
}

static iree_status_t loom_pass_option_schema_validate_decoded_required(
    const loom_pass_descriptor_t* descriptor,
    const loom_pass_decoded_option_t* decoded_options) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_all_bits_set(schema->flags,
                          (loom_pass_option_schema_flags_t)
                              LOOM_PASS_OPTION_SCHEMA_REQUIRED) &&
        !decoded_options[i].present) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing required option '%.*s' for pass '%.*s'",
                              (int)schema->name.size, schema->name.data,
                              (int)descriptor->key.size, descriptor->key.data);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pass_descriptor_decode_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options,
    iree_arena_allocator_t* arena, loom_pass_decoded_options_t* out_options) {
  if (!descriptor || !descriptor->info || !arena || !out_options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor, arena, and output options are required");
  }
  memset(out_options, 0, sizeof(*out_options));
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  IREE_RETURN_IF_ERROR(
      loom_pass_registry_verify_option_schema(descriptor, info));

  iree_string_view_t trimmed_options = iree_string_view_trim(options);
  bool has_options = !iree_string_view_is_empty(trimmed_options);
  if (has_options && descriptor->option_schema_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options, got '{%.*s}'",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)options.size, options.data);
  }

  loom_pass_decoded_option_t* decoded_options = NULL;
  iree_status_t status = iree_ok_status();
  if (descriptor->option_schema_count > 0) {
    status = iree_arena_allocate_array(arena, descriptor->option_schema_count,
                                       sizeof(*decoded_options),
                                       (void**)&decoded_options);
    if (iree_status_is_ok(status)) {
      memset(decoded_options, 0,
             descriptor->option_schema_count * sizeof(*decoded_options));
      for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
        decoded_options[i].schema = &descriptor->option_schema[i];
      }
    }
  }

  loom_pass_option_decode_context_t context = {
      .descriptor = descriptor,
      .decoded_options = decoded_options,
  };
  if (iree_status_is_ok(status) && has_options) {
    status = loom_pass_options_parse(info->name, trimmed_options,
                                     loom_pass_option_schema_decode_assignment,
                                     &context);
  }
  if (iree_status_is_ok(status) && decoded_options) {
    status = loom_pass_option_schema_validate_decoded_required(descriptor,
                                                               decoded_options);
  }
  if (iree_status_is_ok(status)) {
    *out_options = (loom_pass_decoded_options_t){
        .descriptor = descriptor,
        .options = decoded_options,
        .option_count = descriptor->option_schema_count,
    };
  }
  return status;
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
  IREE_RETURN_IF_ERROR(
      loom_pass_descriptor_validate_options(descriptor, options));
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

iree_status_t loom_pass_manager_add_pipeline(
    loom_pass_manager_t* manager, const loom_pass_registry_t* registry,
    iree_string_view_t pipeline,
    loom_pass_pipeline_configure_callback_t configure) {
  if (!manager || !registry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass manager and registry are required");
  }
  IREE_RETURN_IF_ERROR(loom_pass_registry_verify(registry));

  iree_host_size_t initial_manager_count = manager->count;
  iree_string_view_t remaining = pipeline;
  iree_host_size_t pipeline_index = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    loom_pass_pipeline_entry_spec_t spec = {0};
    bool has_entry = false;
    status = loom_pass_pipeline_consume_entry(&remaining, &spec, &has_entry);
    if (!iree_status_is_ok(status) || !has_entry) break;

    const loom_pass_descriptor_t* descriptor = NULL;
    status = loom_pass_registry_lookup(registry, spec.name, &descriptor);
    if (iree_status_is_ok(status) && !descriptor) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "unknown pass '%.*s' at pipeline entry %zu",
                           (int)spec.name.size, spec.name.data, pipeline_index);
    }

    void* pass_user_data = NULL;
    if (iree_status_is_ok(status) && configure.fn) {
      const loom_pass_pipeline_descriptor_entry_t entry = {
          .spec = spec,
          .descriptor = descriptor,
          .pipeline_index = pipeline_index,
      };
      status = configure.fn(configure.user_data, &entry, &pass_user_data);
    }
    if (iree_status_is_ok(status)) {
      status = loom_pass_manager_add_descriptor(manager, descriptor,
                                                spec.options, pass_user_data);
    }
    if (iree_status_is_ok(status)) ++pipeline_index;
  }
  if (!iree_status_is_ok(status)) {
    manager->count = initial_manager_count;
  }
  return status;
}
