// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv.h"

#include "diagnostic.h"
#include "loom/error/error_defs.h"
#include "loom/target/arch/spirv/provider.h"
#include "loom/target/emit/spirv/module_emitter.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
#include "target.h"
#include "workspace.h"

typedef struct loomc_spirv_emit_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;
} loomc_spirv_emit_diagnostic_capture_t;

static loomc_status_t loomc_spirv_emit_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_emit_validate_options(
    const loomc_spirv_emit_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V emit options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V emit options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "SPIR-V emit option extensions are not supported");
  }
  return loomc_spirv_emit_validate_string_view(options->identifier);
}

static iree_status_t loomc_spirv_emit_capture_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_spirv_emit_diagnostic_capture_t* capture =
      (loomc_spirv_emit_diagnostic_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
}

static loomc_string_view_t loomc_spirv_emit_identifier(
    const loomc_spirv_emit_options_t* options) {
  if (options == NULL || loomc_string_view_is_empty(options->identifier)) {
    return loomc_make_cstring_view("module.spv");
  }
  return options->identifier;
}

loomc_status_t loomc_target_environment_create_spirv(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set(
      &loom_spirv_target_provider_set, allocator, out_target_environment);
}

loomc_status_t loomc_spirv_emit_module(
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module,
    const loomc_spirv_emit_options_t* options, loomc_allocator_t allocator,
    loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (target_environment == NULL || workspace == NULL || module == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target_environment, workspace, and module must not be NULL");
  }
  loom_module_t* internal_module = loomc_module_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_emit_validate_options(options));

  allocator = loomc_allocator_or_system(allocator);
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loom_spirv_module_binary_t binary = {0};
  loomc_spirv_emit_diagnostic_capture_t capture = {
      .result = result,
  };
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);

  const loomc_target_pass_environment_t* pass_environment =
      loomc_target_environment_pass_environment(target_environment);
  loomc_status_t status = loomc_status_from_iree(loom_spirv_emit_low_module(
      internal_module, &pass_environment->low_descriptor_registry.registry,
      loom_target_selection_empty(),
      (iree_diagnostic_emitter_t){
          .fn = loomc_spirv_emit_capture_diagnostic,
          .user_data = &capture,
      },
      &scratch_arena, &binary, iree_allocator_from_loomc(allocator)));
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    status = loomc_result_fail_status_diagnostic_consume(
        result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
        loomc_make_cstring_view("SPIRV/EMIT"), status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
        loomc_spirv_emit_identifier(options),
        loomc_make_byte_span(binary.words,
                             binary.word_count * sizeof(uint32_t)));
    if (loomc_status_is_ok(status)) {
      binary = (loom_spirv_module_binary_t){0};
    }
  }

  loom_spirv_module_binary_deinitialize(&binary,
                                        iree_allocator_from_loomc(allocator));
  iree_arena_deinitialize(&scratch_arena);
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}
