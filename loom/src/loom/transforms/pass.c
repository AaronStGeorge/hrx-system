// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pass.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Pass manager
//===----------------------------------------------------------------------===//

iree_status_t loom_pass_manager_initialize(iree_arena_block_pool_t* block_pool,
                                           loom_pass_manager_flags_t flags,
                                           iree_allocator_t allocator,
                                           loom_pass_manager_t* out_manager) {
  memset(out_manager, 0, sizeof(*out_manager));
  out_manager->block_pool = block_pool;
  out_manager->flags = flags;
  out_manager->allocator = allocator;
  return iree_ok_status();
}

static iree_status_t loom_pass_manager_ensure_capacity(
    loom_pass_manager_t* manager) {
  if (manager->count < manager->capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      manager->capacity > 0 ? manager->capacity * 2 : 8;
  loom_pipeline_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      manager->allocator, new_capacity * sizeof(loom_pipeline_entry_t),
      (void**)&new_entries));
  if (manager->count > 0) {
    memcpy(new_entries, manager->entries,
           manager->count * sizeof(loom_pipeline_entry_t));
  }
  if (manager->entries) {
    iree_allocator_free(manager->allocator, manager->entries);
  }
  manager->entries = new_entries;
  manager->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t loom_pass_manager_add_module_pass(loom_pass_manager_t* manager,
                                                const loom_pass_info_t* info,
                                                loom_module_pass_fn_t run,
                                                loom_pass_create_fn_t create,
                                                loom_pass_destroy_fn_t destroy,
                                                iree_string_view_t options) {
  IREE_RETURN_IF_ERROR(loom_pass_manager_ensure_capacity(manager));
  manager->entries[manager->count++] = (loom_pipeline_entry_t){
      .info = info,
      .module_run = run,
      .create = create,
      .destroy = destroy,
      .options = options,
  };
  return iree_ok_status();
}

iree_status_t loom_pass_manager_add_function_pass(
    loom_pass_manager_t* manager, const loom_pass_info_t* info,
    loom_function_pass_fn_t run, loom_pass_create_fn_t create,
    loom_pass_destroy_fn_t destroy, iree_string_view_t options) {
  IREE_RETURN_IF_ERROR(loom_pass_manager_ensure_capacity(manager));
  manager->entries[manager->count++] = (loom_pipeline_entry_t){
      .info = info,
      .function_run = run,
      .create = create,
      .destroy = destroy,
      .options = options,
  };
  return iree_ok_status();
}

// Runs a function pass once per bodyful function-like symbol. The pass's
// callback scratch arena is reset before each function so one-shot region
// stacks and worklists are bounded by the largest single function.
static iree_status_t loom_pass_manager_run_function_entry(
    loom_pass_manager_t* manager, loom_pipeline_entry_t* entry,
    loom_pass_t* pass, loom_module_t* module) {
  iree_arena_allocator_t function_arena;
  iree_arena_initialize(manager->block_pool, &function_arena);

  pass->function_run = entry->function_run;
  pass->arena = &function_arena;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < module->symbols.count && iree_status_is_ok(status); ++i) {
    loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loom_symbol_kind_is_function_like(symbol->kind)) continue;
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_body(function)) continue;

    iree_arena_reset(&function_arena);
    status = entry->function_run(pass, module, function);
  }

  pass->arena = pass->instance_arena;
  iree_arena_deinitialize(&function_arena);
  return status;
}

// Runs a single pass on the module (or on each function for function passes).
static iree_status_t loom_pass_manager_run_entry(loom_pass_manager_t* manager,
                                                 loom_pipeline_entry_t* entry,
                                                 loom_module_t* module) {
  // Create a stable arena for this pipeline entry's pass instance.
  iree_arena_allocator_t instance_arena;
  iree_arena_initialize(manager->block_pool, &instance_arena);

  // Build the pass instance.
  loom_pass_t pass = {
      .info = entry->info,
      .instance_arena = &instance_arena,
      .arena = &instance_arena,
  };

  // Allocate statistics counters from the stable instance arena.
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status) && entry->info->statistic_count > 0) {
    iree_host_size_t statistics_size =
        (iree_host_size_t)entry->info->statistic_count * sizeof(int64_t);
    status = iree_arena_allocate(&instance_arena, statistics_size,
                                 (void**)&pass.statistics);
    if (iree_status_is_ok(status)) {
      memset(pass.statistics, 0, statistics_size);
    }
  }

  // Initialize the pass (parse options, allocate state).
  bool created = false;
  if (iree_status_is_ok(status) && entry->create) {
    status = entry->create(&pass, entry->options);
    created = iree_status_is_ok(status);
  }

  // Dispatch based on pass kind.
  if (iree_status_is_ok(status)) {
    if (entry->info->kind == LOOM_PASS_MODULE) {
      pass.module_run = entry->module_run;
      status = entry->module_run(&pass, module);
    } else {
      status =
          loom_pass_manager_run_function_entry(manager, entry, &pass, module);
    }
  }

  // Single cleanup path.
  if (created && entry->destroy) {
    entry->destroy(&pass);
  }
  iree_arena_deinitialize(&instance_arena);
  return status;
}

iree_status_t loom_pass_manager_run(loom_pass_manager_t* manager,
                                    loom_module_t* module) {
  for (iree_host_size_t i = 0; i < manager->count; ++i) {
    loom_pipeline_entry_t* entry = &manager->entries[i];
    iree_status_t status = loom_pass_manager_run_entry(manager, entry, module);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(
          status, "pass '%.*s' failed (pipeline entry %" PRIhsz ")",
          (int)entry->info->name.size, entry->info->name.data, i);
    }
  }
  return iree_ok_status();
}

void loom_pass_manager_deinitialize(loom_pass_manager_t* manager) {
  if (manager->entries) {
    iree_allocator_free(manager->allocator, manager->entries);
  }
  memset(manager, 0, sizeof(*manager));
}
