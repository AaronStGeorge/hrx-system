// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/tooling/input.h"

#include "loom/import/cxx/import.h"
#include "loomc/iree.h"

static iree_status_t loom_cxx_input_copy_strings(
    const loomc_string_view_t* source, iree_host_size_t count,
    iree_arena_allocator_t* arena, const iree_string_view_t** out_target) {
  *out_target = NULL;
  if (!count) {
    return iree_ok_status();
  }
  iree_string_view_t* target = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*target),
                                                 (void**)&target));
  for (iree_host_size_t i = 0; i < count; ++i) {
    target[i] = iree_string_view_from_loomc(source[i]);
  }
  *out_target = target;
  return iree_ok_status();
}

static iree_status_t loom_cxx_input_load(const loom_input_request_t* request,
                                         loom_input_source_capture_t capture,
                                         loom_context_t* context,
                                         iree_arena_block_pool_t* block_pool,
                                         iree_allocator_t host_allocator,
                                         loom_module_t** out_module) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loomc_cxx_import_options_t parsed_options;
  iree_status_t status = loom_cxx_input_parse_options(
      request->path, request->options, &arena, host_allocator, &parsed_options);
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.standard = iree_string_view_from_loomc(parsed_options.standard);
  options.triple = iree_string_view_from_loomc(parsed_options.triple);
  options.data_model = (loom_cxx_data_model_t)parsed_options.data_model;
  options.fp_model = (loom_cxx_fp_model_t)parsed_options.fp_model;
  options.flags = parsed_options.flags;
  options.include_path_count = parsed_options.include_path_count;
  options.system_include_path_count = parsed_options.system_include_path_count;
  options.root_count = parsed_options.root_count;
  options.define_count = parsed_options.define_count;
  if (iree_status_is_ok(status)) {
    status = loom_cxx_input_copy_strings(parsed_options.include_paths,
                                         parsed_options.include_path_count,
                                         &arena, &options.include_paths);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_cxx_input_copy_strings(parsed_options.system_include_paths,
                                    parsed_options.system_include_path_count,
                                    &arena, &options.system_include_paths);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cxx_input_copy_strings(parsed_options.roots,
                                         parsed_options.root_count, &arena,
                                         &options.roots);
  }
  loom_cxx_define_t* defines = NULL;
  if (iree_status_is_ok(status) && parsed_options.define_count) {
    status = iree_arena_allocate_array(&arena, parsed_options.define_count,
                                       sizeof(*defines), (void**)&defines);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < parsed_options.define_count; ++i) {
    defines[i] = (loom_cxx_define_t){
        .name = iree_string_view_from_loomc(parsed_options.defines[i].name),
        .value = iree_string_view_from_loomc(parsed_options.defines[i].value),
    };
  }
  options.defines = defines;
  if (iree_status_is_ok(status)) {
    options.diagnostic_sink = request->parse_options.diagnostic_sink;
    options.low_asm_environment = request->parse_options.low_asm_environment;
    if (capture.fn) {
      options.source_observer = (loom_cxx_source_observer_t){
          .fn = capture.fn, .user_data = capture.user_data};
    }
    status = loom_cxx_import(request->source, request->path, context,
                             block_pool, &options, host_allocator, out_module);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

const loom_input_provider_t loom_cxx_input_provider = {
    .format = &loom_cxx_input_format,
    .load = loom_cxx_input_load,
};
