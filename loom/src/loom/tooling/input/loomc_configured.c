// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/input/loomc_configured.h"

#if LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX
#include "loom/import/cxx/tooling/loomc_input.h"
#include "loom/import/cxx/tooling/options.h"

static iree_status_t loom_configured_input_import_loomc(
    void* user_data, iree_string_view_t format,
    iree_string_view_t input_options,
    const loom_tooling_source_path_options_t* source_path_options,
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  (void)user_data;
  if (iree_string_view_equal(format, IREE_SV("cxx"))) {
    return loom_cxx_input_import_loomc(
        context, workspace, source, input_options, source_path_options,
        block_pool, host_allocator, out_module, out_result);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "input format '%.*s' has no configured LoomC importer", (int)format.size,
      format.data);
}

loom_tooling_input_import_loomc_fn_t loom_configured_input_loomc_importer(
    void) {
  return loom_configured_input_import_loomc;
}
#endif  // LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX

loom_input_format_list_t loom_configured_input_loomc_formats(void) {
#if LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX
  static const loom_input_format_t* const formats[] = {&loom_cxx_input_format};
  return (loom_input_format_list_t){formats, IREE_ARRAYSIZE(formats)};
#else
  return (loom_input_format_list_t){0};
#endif
}
