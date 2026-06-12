// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu.h"

#include <string.h>

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loomc/iree.h"
#include "module.h"
#include "target.h"

static char loomc_amdgpu_profile_payload_type;

static void loomc_amdgpu_profile_payload_deinitialize(
    void* payload, loomc_allocator_t allocator) {
  (void)payload;
  (void)allocator;
}

static loomc_status_t loomc_amdgpu_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_amdgpu_profile_options_validate(
    const loomc_amdgpu_profile_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU profile options are required");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "AMDGPU profile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "AMDGPU profile options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "AMDGPU profile option extensions are not supported");
  }
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(options->identifier));
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(options->processor));
  if (loomc_string_view_is_empty(options->processor)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU profile options require processor");
  }
  return loomc_ok_status();
}

static iree_status_t loomc_amdgpu_count_targetless_kernels(
    loom_module_t* module, uint32_t* out_count) {
  if (out_count == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU targetless kernel count requires an output count");
  }
  *out_count = 0;
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU targetless kernel count requires a module with a body block");
  }

  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (loom_kernel_def_isa(op) &&
        !loom_symbol_ref_is_valid(loom_kernel_def_target(op))) {
      ++*out_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loomc_amdgpu_validate_target_symbol(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_amdgpu_processor_info_t* processor, loom_symbol_ref_t target_ref,
    bool* out_reusable) {
  *out_reusable = false;

  const loom_symbol_t* symbol = &module->symbols.entries[target_ref.symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_target_isa(symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU target assignment symbol '@%.*s' already names a non-AMDGPU "
        "target op",
        (int)symbol_name.size, symbol_name.data);
  }

  const iree_string_view_t existing_processor =
      loom_amdgpu_target_record_processor_name(module, symbol->defining_op);
  if (!iree_string_view_equal(existing_processor, processor->processor)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU target assignment symbol '@%.*s' selects processor '%.*s', "
        "but the selected profile requires '%.*s'",
        (int)symbol_name.size, symbol_name.data, (int)existing_processor.size,
        existing_processor.data, (int)processor->processor.size,
        processor->processor.data);
  }

  *out_reusable = true;
  return iree_ok_status();
}

static iree_status_t loomc_amdgpu_resolve_profile_target_ref(
    loom_module_t* module, const loom_amdgpu_processor_info_t* processor,
    loom_symbol_ref_t* out_target_ref) {
  if (out_target_ref == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU target resolver requires an output target ref");
  }
  *out_target_ref = loom_symbol_ref_null();
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU target resolver requires a module with a body block");
  }
  if (processor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target resolver requires a selected processor");
  }
  if (iree_string_view_is_empty(processor->processor)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target resolver selected a processor row with no processor "
        "name");
  }

  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, processor->processor, &symbol_name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(module, symbol_name_id, &symbol_id));
  }
  *out_target_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};

  bool reusable = false;
  IREE_RETURN_IF_ERROR(loomc_amdgpu_validate_target_symbol(
      module, processor->processor, processor, *out_target_ref, &reusable));
  if (reusable) {
    return iree_ok_status();
  }

  loom_block_t* module_block = loom_module_block(module);
  loom_builder_t builder = {0};
  loom_builder_initialize(module, &module->arena, module_block, &builder);
  if (module_block->first_op != NULL) {
    loom_builder_set_before(&builder, module_block->first_op);
  }

  loom_op_t* target_op = NULL;
  return loom_amdgpu_target_record_build_for_processor(
      &builder, processor, *out_target_ref, LOOM_LOCATION_UNKNOWN, &target_op);
}

static iree_status_t loomc_amdgpu_assign_targetless_kernel_targets(
    loom_module_t* module, const loom_amdgpu_processor_info_t* processor,
    loomc_amdgpu_target_assignment_t* out_assignment) {
  if (out_assignment != NULL) {
    *out_assignment = (loomc_amdgpu_target_assignment_t){0};
  }
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU target assignment requires a module with a body block");
  }

  uint32_t targetless_kernel_count = 0;
  IREE_RETURN_IF_ERROR(
      loomc_amdgpu_count_targetless_kernels(module, &targetless_kernel_count));
  if (out_assignment != NULL) {
    out_assignment->targetless_kernel_count = targetless_kernel_count;
  }
  if (targetless_kernel_count == 0) {
    return iree_ok_status();
  }

  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loomc_amdgpu_resolve_profile_target_ref(module, processor, &target_ref));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, &module->arena));
  iree_status_t status = iree_ok_status();
  uint32_t assigned_kernel_count = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (!loom_kernel_def_isa(op) ||
        loom_symbol_ref_is_valid(loom_kernel_def_target(op))) {
      continue;
    }
    status =
        loom_rewriter_set_attr(&rewriter, op, loom_kernel_def_target_ATTR_INDEX,
                               loom_attr_symbol(target_ref));
    if (!iree_status_is_ok(status)) {
      break;
    }
    ++assigned_kernel_count;
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (out_assignment != NULL) {
    out_assignment->changed = assigned_kernel_count != 0;
  }
  return iree_ok_status();
}

static void loomc_amdgpu_emit_artifact_release(void* storage,
                                               iree_allocator_t allocator) {
  iree_allocator_free(allocator, storage);
}

typedef struct loomc_amdgpu_emit_library_storage_t {
  // AMDGPU kernel library storage.
  loom_amdgpu_hal_kernel_library_t library;
} loomc_amdgpu_emit_library_storage_t;

static void loomc_amdgpu_emit_library_release(void* storage,
                                              iree_allocator_t allocator) {
  loomc_amdgpu_emit_library_storage_t* library_storage =
      (loomc_amdgpu_emit_library_storage_t*)storage;
  loom_amdgpu_hal_kernel_library_deinitialize(&library_storage->library,
                                              allocator);
  iree_allocator_free(allocator, library_storage);
}

static iree_status_t loomc_amdgpu_forward_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  iree_diagnostic_emitter_t* emitter = (iree_diagnostic_emitter_t*)user_data;
  if (diagnostic == NULL || diagnostic->error == NULL) {
    return iree_ok_status();
  }
  const loom_diagnostic_emission_t emission = {
      .error = diagnostic->error,
      .params = diagnostic->params,
      .param_count = diagnostic->param_count,
  };
  return iree_diagnostic_emit(*emitter, &emission);
}

static iree_status_t loomc_amdgpu_emit_module_artifact(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};

  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)request->target_selection.data;
  iree_diagnostic_emitter_t diagnostic_emitter = request->diagnostic_emitter;
  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .processor = processor ? processor->processor : iree_string_view_empty(),
      .target_selection = request->target_selection,
      .diagnostic_sink =
          {
              .fn = loomc_amdgpu_forward_diagnostic,
              .user_data = &diagnostic_emitter,
          },
      .max_errors = 20,
      .artifact_name = request->identifier,
      .artifact_manifest_identifier = request->artifact_manifest.identifier,
      .artifact_manifest =
          {
              .mode = request->artifact_manifest.mode,
          },
  };
  bool emitted = false;
  loom_amdgpu_hal_kernel_library_t library = {0};
  iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
      request->module, &library_options, request->allocator, &emitted,
      &library);
  if (iree_status_is_ok(status) && !emitted) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "AMDGPU HSACO emission produced no executable bytes");
  }
  if (iree_status_is_ok(status) &&
      library.artifact_manifest.contents.data == NULL) {
    out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    out_artifact->contents = iree_make_const_byte_span(
        library.hsaco_data, library.hsaco_data_length);
    out_artifact->storage = library.hsaco_data;
    out_artifact->release = loomc_amdgpu_emit_artifact_release;
    library.hsaco_data = NULL;
    library.hsaco_data_length = 0;
  } else if (iree_status_is_ok(status)) {
    loomc_amdgpu_emit_library_storage_t* storage = NULL;
    status = iree_allocator_malloc(request->allocator, sizeof(*storage),
                                   (void**)&storage);
    if (iree_status_is_ok(status)) {
      *storage = (loomc_amdgpu_emit_library_storage_t){
          .library = library,
      };
      out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
      out_artifact->contents = iree_make_const_byte_span(
          storage->library.hsaco_data, storage->library.hsaco_data_length);
      out_artifact->sidecars = &storage->library.artifact_manifest;
      out_artifact->sidecar_count = 1;
      out_artifact->storage = storage;
      out_artifact->release = loomc_amdgpu_emit_library_release;
      library = (loom_amdgpu_hal_kernel_library_t){0};
    }
  }
  loom_amdgpu_hal_kernel_library_deinitialize(&library, request->allocator);
  return status;
}

static const loom_target_emitter_t loomc_amdgpu_hsaco_emitter = {
    .name = {"amdgpu-hsaco", 12},
    .public_artifact_format = {LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO,
                               sizeof(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO) - 1},
    .default_identifier = {"module.hsaco", 12},
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .emit = loomc_amdgpu_emit_module_artifact,
};

static const loom_target_emitter_t* const kLoomcAmdgpuEmitters[] = {
    &loomc_amdgpu_hsaco_emitter,
};

static const loom_target_provider_t loomc_amdgpu_emit_target_provider = {
    .emitter_list =
        {
            .values = kLoomcAmdgpuEmitters,
            .count = IREE_ARRAYSIZE(kLoomcAmdgpuEmitters),
        },
};

static const loom_target_provider_t* const kLoomcAmdgpuTargetProviders[] = {
    &loom_amdgpu_target_provider,
    &loomc_amdgpu_emit_target_provider,
};

static const loom_target_provider_set_t loomc_amdgpu_target_provider_set = {
    .providers = kLoomcAmdgpuTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomcAmdgpuTargetProviders),
};

loomc_status_t loomc_target_environment_create_amdgpu(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set(
      &loomc_amdgpu_target_provider_set, allocator, out_target_environment);
}

loomc_status_t loomc_target_profile_create_amdgpu(
    loomc_target_environment_t* target_environment,
    const loomc_amdgpu_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile) {
  if (out_profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_profile must not be NULL");
  }
  *out_profile = NULL;
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target_environment must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_profile_options_validate(options));

  const loom_amdgpu_processor_info_t* processor = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_lookup_processor(
          iree_string_view_from_loomc(options->processor), &processor)));
  bool hsaco_supported = false;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_processor_supports_hsaco(
          processor, &hsaco_supported)));
  if (!hsaco_supported) {
    return loomc_make_status(LOOMC_STATUS_UNAVAILABLE,
                             "AMDGPU processor cannot be emitted as HSACO");
  }
  const loom_target_bundle_t* target_bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->descriptor_set_ordinal);
  if (target_bundle == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNAVAILABLE,
        "AMDGPU processor has no Loom target bundle for its descriptor set");
  }

  const loom_target_selection_t selection = {
      .bundle = target_bundle,
      .data = (void*)processor,
  };
  const loomc_target_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_string_view_is_empty(options->identifier)
                        ? options->processor
                        : options->identifier,
  };
  return loomc_target_profile_create_from_selection(
      target_environment, &profile_options, selection,
      &loomc_amdgpu_profile_payload_type, (void*)processor,
      loomc_amdgpu_profile_payload_deinitialize, allocator, out_profile);
}

loomc_string_view_t loomc_amdgpu_target_profile_processor(
    const loomc_target_profile_t* profile) {
  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)loomc_target_profile_payload(
          profile, &loomc_amdgpu_profile_payload_type);
  return processor ? loomc_string_view_from_iree(processor->processor)
                   : loomc_string_view_empty();
}

loomc_status_t loomc_amdgpu_processor_from_hsa_isa_name(
    loomc_string_view_t hsa_isa_name, loomc_string_view_t* out_processor) {
  if (out_processor == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_processor must not be NULL");
  }
  *out_processor = loomc_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(hsa_isa_name));

  loom_amdgpu_amdhsa_target_id_t target_id = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_parse_amdhsa_target_id(
          iree_string_view_from_loomc(hsa_isa_name), &target_id)));
  *out_processor = loomc_string_view_from_iree(target_id.processor->processor);
  return loomc_ok_status();
}

loomc_status_t loomc_amdgpu_module_assign_targetless_kernel_targets(
    loomc_module_t* module, loomc_target_profile_t* profile,
    loomc_amdgpu_target_assignment_t* out_assignment) {
  if (out_assignment != NULL) {
    *out_assignment = (loomc_amdgpu_target_assignment_t){0};
  }
  if (module == NULL || profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and profile must not be NULL");
  }
  loom_module_t* internal_module = loomc_module_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  loom_target_selection_t selection =
      loomc_target_profile_loom_target_selection(profile);
  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)selection.data;
  if (processor == NULL || selection.bundle == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "profile is not an AMDGPU processor profile");
  }

  return loomc_status_from_iree(loomc_amdgpu_assign_targetless_kernel_targets(
      internal_module, processor, out_assignment));
}
