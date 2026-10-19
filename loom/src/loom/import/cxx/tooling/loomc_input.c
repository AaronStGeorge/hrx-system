// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/tooling/loomc_input.h"

#include "loom/import/cxx/tooling/options.h"
#include "loomc/iree.h"

typedef struct loom_cxx_input_source_path_mapper_t {
  // Command-line source path policy borrowed during this import.
  const loom_tooling_source_path_options_t* options;
  // Invocation-local storage retaining each mapped identifier.
  iree_arena_allocator_t* arena;
} loom_cxx_input_source_path_mapper_t;

static loomc_status_t loom_cxx_input_map_source_path(
    void* user_data, loomc_string_view_t path,
    loomc_string_view_t* out_identifier) {
  loom_cxx_input_source_path_mapper_t* mapper =
      (loom_cxx_input_source_path_mapper_t*)user_data;
  iree_string_view_t identifier = iree_string_view_empty();
  char* identifier_storage = NULL;
  loomc_status_t status = loomc_status_from_iree(loom_tooling_source_path_remap(
      iree_string_view_from_loomc(path), mapper->options,
      iree_arena_allocator(mapper->arena), &identifier, &identifier_storage));
  if (loomc_status_is_ok(status)) {
    *out_identifier = loomc_string_view_from_iree(identifier);
  }
  return status;
}

iree_status_t loom_cxx_input_import_loomc(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source, iree_string_view_t input_options,
    const loom_tooling_source_path_options_t* source_path_options,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    loomc_module_t** out_module, loomc_result_t** out_result) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  loomc_cxx_import_options_t options;
  iree_status_t status = loom_cxx_input_parse_options(
      iree_string_view_from_loomc(loomc_source_identifier(source)),
      input_options, &arena, host_allocator, &options);
  loom_cxx_input_source_path_mapper_t source_path_mapper = {
      .options = source_path_options,
      .arena = &arena,
  };
  if (iree_status_is_ok(status)) {
    if (source_path_options->prefix_maps.count != 0) {
      options.source_path_mapper = (loomc_cxx_source_path_mapper_t){
          .fn = loom_cxx_input_map_source_path,
          .user_data = &source_path_mapper,
      };
    }
    status = iree_status_from_loomc(loomc_module_import_cxx(
        context, workspace, source, &options,
        loomc_allocator_from_iree(host_allocator), out_module, out_result));
  }

  iree_arena_deinitialize(&arena);
  return status;
}
