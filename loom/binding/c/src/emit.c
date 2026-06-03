// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/emit.h"

#include <string.h>

#include "diagnostic.h"
#include "iree/base/internal/arena.h"
#include "loom/error/error_defs.h"
#include "loom/target/provider.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
#include "target.h"
#include "workspace.h"

typedef struct loomc_descriptor_prefix_t {
  // Structure type identifying the descriptor.
  loomc_structure_type_t type;

  // Size of the descriptor in bytes.
  loomc_host_size_t structure_size;

  // Next descriptor in the option extension chain.
  const void* next;
} loomc_descriptor_prefix_t;

typedef struct loomc_emit_resolved_options_t {
  // Artifact format requested by the caller.
  loomc_string_view_t artifact_format;

  // Artifact identifier requested by the caller.
  loomc_string_view_t identifier;

  // Artifact classes requested by the caller.
  loomc_emit_artifact_flags_t artifact_flags;

  // Target selection requested by the caller.
  loomc_target_selection_t* target_selection;

  // Full extension chain passed to target-specific emitters.
  const void* option_chain;
} loomc_emit_resolved_options_t;

typedef struct loomc_emit_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;
} loomc_emit_diagnostic_capture_t;

static loomc_status_t loomc_emit_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_result_fail_message(
    loomc_result_t* result, loomc_string_view_t code,
    loomc_string_view_t message) {
  loomc_diagnostic_t diagnostic = {
      .severity = LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      .code = code,
      .message = message,
  };
  LOOMC_RETURN_IF_ERROR(loomc_result_add_diagnostic(result, &diagnostic));
  return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
}

static loomc_status_t loomc_emit_result_fail_cstring(loomc_result_t* result,
                                                     const char* code,
                                                     const char* message) {
  return loomc_emit_result_fail_message(result, loomc_make_cstring_view(code),
                                        loomc_make_cstring_view(message));
}

static loomc_status_t loomc_emit_result_fail_unknown_option(
    loomc_result_t* result, loomc_string_view_t key,
    loomc_allocator_t allocator) {
  iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  loomc_status_t status = loomc_status_from_iree(
      iree_string_builder_append_cstring(&builder, "unknown emit option key"));
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(key)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, " '"));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(key)) {
    status = loomc_status_from_iree(iree_string_builder_append_string(
        &builder, iree_string_view_from_loomc(key)));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(key)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, "'"));
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_result_fail_message(
        result, loomc_make_cstring_view("EMIT/OPTION"),
        loomc_string_view_from_iree(iree_string_builder_view(&builder)));
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}

static loomc_status_t loomc_emit_validate_options(
    const loomc_emit_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "emit options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "emit options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_emit_validate_string_view(options->artifact_format));
  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_string_view(options->identifier));
  const loomc_emit_artifact_flags_t known_artifact_flags =
      LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY;
  if ((options->artifact_flags & ~known_artifact_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "emit options contain unknown artifact flags");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_validate_option_dict(
    const loomc_option_dict_t* dict) {
  if (dict->type != LOOMC_STRUCTURE_TYPE_OPTION_DICT) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "option dictionary has an unknown structure type");
  }
  if (dict->structure_size != 0 && dict->structure_size < sizeof(*dict)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "option dictionary structure_size is too small");
  }
  if (dict->entry_count != 0 && dict->entries == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "option dictionary entry_count is non-zero but entries is NULL");
  }
  for (loomc_host_size_t i = 0; i < dict->entry_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_emit_validate_string_view(dict->entries[i].key));
    LOOMC_RETURN_IF_ERROR(
        loomc_emit_validate_string_view(dict->entries[i].value));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_validate_unknown_descriptor(
    const loomc_descriptor_prefix_t* prefix) {
  if (prefix->structure_size != 0 && prefix->structure_size < sizeof(*prefix)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "option extension structure_size is too small");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_apply_option_entry(
    const loomc_option_entry_t* entry, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_emit_resolved_options_t* options) {
  if (loomc_string_view_equal(
          entry->key,
          loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER))) {
    options->identifier = entry->value;
    return loomc_ok_status();
  }
  return loomc_emit_result_fail_unknown_option(result, entry->key, allocator);
}

static loomc_status_t loomc_emit_apply_option_dict(
    const loomc_option_dict_t* dict, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_emit_resolved_options_t* options) {
  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_option_dict(dict));
  loomc_status_t status = loomc_ok_status();
  for (loomc_host_size_t i = 0;
       i < dict->entry_count && loomc_status_is_ok(status); ++i) {
    status = loomc_emit_apply_option_entry(&dict->entries[i], result, allocator,
                                           options);
  }
  return status;
}

static loomc_status_t loomc_emit_resolve_options(
    const loomc_emit_options_t* options, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_emit_resolved_options_t* out_options) {
  *out_options = (loomc_emit_resolved_options_t){
      .artifact_format =
          options ? options->artifact_format : loomc_string_view_empty(),
      .identifier = options ? options->identifier : loomc_string_view_empty(),
      .artifact_flags = options ? options->artifact_flags : 0,
      .option_chain = options ? options->next : NULL,
  };

  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_options(options));

  const void* next = options ? options->next : NULL;
  while (next != NULL) {
    const loomc_descriptor_prefix_t* prefix =
        (const loomc_descriptor_prefix_t*)next;
    switch (prefix->type) {
      case LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS: {
        if (out_options->target_selection != NULL) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "emit option chain contains duplicate target selection options");
        }
        const loomc_target_selection_options_t* target_options =
            (const loomc_target_selection_options_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_target_selection_options_validate(target_options));
        out_options->target_selection = target_options->target_selection;
        next = target_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_OPTION_DICT: {
        const loomc_option_dict_t* dict = (const loomc_option_dict_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_emit_apply_option_dict(dict, result, allocator, out_options));
        next = dict->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_NONE:
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "emit option extension is missing a structure type");
      default:
        LOOMC_RETURN_IF_ERROR(loomc_emit_validate_unknown_descriptor(prefix));
        next = prefix->next;
        break;
    }
  }

  return loomc_ok_status();
}

static loomc_status_t loomc_emit_result_fail_format_message(
    loomc_result_t* result, const char* message, loomc_string_view_t format,
    loomc_allocator_t allocator) {
  iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  loomc_status_t status = loomc_status_from_iree(
      iree_string_builder_append_cstring(&builder, message));
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(format)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, " '"));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(format)) {
    status = loomc_status_from_iree(iree_string_builder_append_string(
        &builder, iree_string_view_from_loomc(format)));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(format)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, "'"));
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_result_fail_message(
        result, loomc_make_cstring_view("EMIT/TARGET"),
        loomc_string_view_from_iree(iree_string_builder_view(&builder)));
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}

static loomc_status_t loomc_emit_select_emitter(
    const loom_target_environment_t* target_environment,
    loomc_string_view_t artifact_format, loomc_result_t* result,
    loomc_allocator_t allocator, const loom_target_emitter_t** out_emitter) {
  *out_emitter = NULL;
  loom_target_emitter_list_t emitters =
      loom_target_environment_emitter_list(target_environment);
  if (loomc_string_view_is_empty(artifact_format)) {
    if (emitters.count == 0) {
      return loomc_emit_result_fail_cstring(
          result, "EMIT/TARGET",
          "target environment does not contain a linked emitter");
    }
    if (emitters.count > 1) {
      return loomc_emit_result_fail_cstring(
          result, "EMIT/TARGET",
          "artifact_format is required when multiple emitters are linked");
    }
    *out_emitter = emitters.values[0];
    return loomc_ok_status();
  }

  for (iree_host_size_t i = 0; i < emitters.count; ++i) {
    const loom_target_emitter_t* emitter = emitters.values[i];
    if (iree_string_view_equal(emitter->public_artifact_format,
                               iree_string_view_from_loomc(artifact_format))) {
      if (*out_emitter != NULL) {
        return loomc_make_status(
            LOOMC_STATUS_INTERNAL,
            "target environment contains duplicate emitter artifact formats");
      }
      *out_emitter = emitter;
    }
  }
  if (*out_emitter == NULL) {
    return loomc_emit_result_fail_format_message(
        result, "no linked emitter supports artifact format", artifact_format,
        allocator);
  }
  return loomc_ok_status();
}

static iree_status_t loomc_emit_capture_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_emit_diagnostic_capture_t* capture =
      (loomc_emit_diagnostic_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
}

static void loomc_emit_artifact_deinitialize(
    loom_target_emit_artifact_t* artifact, iree_allocator_t allocator) {
  if (artifact == NULL) {
    return;
  }
  if (artifact->storage != NULL && artifact->deinitialize != NULL) {
    artifact->deinitialize(artifact->storage, allocator);
  }
  *artifact = (loom_target_emit_artifact_t){0};
}

static loomc_string_view_t loomc_emit_identifier(
    const loomc_emit_resolved_options_t* options,
    const loom_target_emitter_t* emitter) {
  if (!loomc_string_view_is_empty(options->identifier)) {
    return options->identifier;
  }
  return loomc_string_view_from_iree(emitter->default_identifier);
}

static loomc_status_t loomc_emit_add_artifact(
    loomc_result_t* result, const loomc_emit_resolved_options_t* options,
    const loom_target_emitter_t* emitter,
    loom_target_emit_artifact_t* target_artifact) {
  if (target_artifact->contents.data == NULL &&
      target_artifact->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INTERNAL,
                             "emitter returned artifact length with no data");
  }
  if (target_artifact->target_artifact_format !=
      LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    if (target_artifact->target_artifact_format !=
        emitter->target_artifact_format) {
      return loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "emitter returned an unexpected target artifact format");
    }
  }

  loomc_status_t status = loomc_result_add_artifact_take_contents(
      result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
      loomc_string_view_from_iree(emitter->public_artifact_format),
      loomc_emit_identifier(options, emitter),
      loomc_byte_span_from_iree(target_artifact->contents));
  if (loomc_status_is_ok(status)) {
    target_artifact->contents = iree_const_byte_span_empty();
    target_artifact->storage = NULL;
    target_artifact->deinitialize = NULL;
  }
  return status;
}

loomc_status_t loomc_emit_module(loomc_target_environment_t* target_environment,
                                 loomc_workspace_t* workspace,
                                 loomc_module_t* module,
                                 const loomc_emit_options_t* options,
                                 loomc_allocator_t allocator,
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

  allocator = loomc_allocator_or_system(allocator);
  iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  loomc_result_t* result = NULL;
  loomc_status_t status =
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result);

  loomc_emit_resolved_options_t resolved_options = {0};
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_resolve_options(options, result, allocator,
                                        &resolved_options);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_target_selection_validate_environment(
        resolved_options.target_selection, target_environment);
  }

  const loom_target_environment_t* internal_target_environment =
      loomc_target_environment_loom_target_environment(target_environment);
  const loom_target_emitter_t* emitter = NULL;
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_emit_select_emitter(internal_target_environment,
                                       resolved_options.artifact_format, result,
                                       allocator, &emitter);
  }

  loom_target_emit_artifact_t target_artifact = {0};
  iree_arena_allocator_t scratch_arena;
  bool scratch_arena_initialized = false;
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      emitter != NULL) {
    if (emitter->emit == NULL) {
      status = loomc_make_status(LOOMC_STATUS_INTERNAL,
                                 "target emitter has no emit function");
    } else {
      iree_arena_initialize(loomc_workspace_block_pool(workspace),
                            &scratch_arena);
      scratch_arena_initialized = true;
      const loomc_target_pass_environment_t* pass_environment =
          loomc_target_environment_pass_environment(target_environment);
      loomc_emit_diagnostic_capture_t capture = {
          .result = result,
      };
      const loom_target_emit_request_t request = {
          .target_environment = internal_target_environment,
          .low_descriptor_registry =
              &pass_environment->low_descriptor_registry.registry,
          .module = internal_module,
          .target_selection = loomc_target_selection_loom_target_selection(
              resolved_options.target_selection),
          .option_chain = resolved_options.option_chain,
          .identifier = iree_string_view_from_loomc(
              loomc_emit_identifier(&resolved_options, emitter)),
          .diagnostic_emitter =
              {
                  .fn = loomc_emit_capture_diagnostic,
                  .user_data = &capture,
              },
          .scratch_arena = &scratch_arena,
          .allocator = host_allocator,
      };
      status =
          loomc_status_from_iree(emitter->emit(&request, &target_artifact));
      if (!loomc_status_is_ok(status) &&
          loomc_status_is_result_diagnostic(status)) {
        status = loomc_result_fail_status_diagnostic_consume(
            result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
            loomc_make_cstring_view("EMIT/TARGET"), status);
      }
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      emitter != NULL) {
    status = loomc_emit_add_artifact(result, &resolved_options, emitter,
                                     &target_artifact);
  }

  loomc_emit_artifact_deinitialize(&target_artifact, host_allocator);
  if (scratch_arena_initialized) {
    iree_arena_deinitialize(&scratch_arena);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}
