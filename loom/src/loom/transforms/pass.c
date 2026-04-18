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
// Pass option parsing
//===----------------------------------------------------------------------===//

iree_status_t loom_pass_pipeline_consume_entry(
    iree_string_view_t* pipeline, loom_pass_pipeline_entry_spec_t* out_entry,
    bool* out_has_entry) {
  IREE_ASSERT_ARGUMENT(pipeline);
  IREE_ASSERT_ARGUMENT(out_entry);
  IREE_ASSERT_ARGUMENT(out_has_entry);

  memset(out_entry, 0, sizeof(*out_entry));
  *out_has_entry = false;

  iree_string_view_t remaining = iree_string_view_trim(*pipeline);
  if (iree_string_view_is_empty(remaining)) {
    *pipeline = remaining;
    return iree_ok_status();
  }

  iree_host_size_t separator_position = IREE_STRING_VIEW_NPOS;
  uint32_t brace_depth = 0;
  for (iree_host_size_t i = 0; i < remaining.size; ++i) {
    char c = remaining.data[i];
    if (c == '{') {
      if (brace_depth != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "nested pass option dictionaries are not "
                                "supported");
      }
      ++brace_depth;
    } else if (c == '}') {
      if (brace_depth == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unexpected '}' in pass pipeline");
      }
      --brace_depth;
    } else if (c == ',' && brace_depth == 0) {
      separator_position = i;
      break;
    }
  }
  if (brace_depth != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated pass option dictionary");
  }

  iree_string_view_t entry = remaining;
  if (separator_position != IREE_STRING_VIEW_NPOS) {
    entry = iree_string_view_substr(remaining, 0, separator_position);
    iree_string_view_t next = iree_string_view_substr(
        remaining, separator_position + 1, IREE_STRING_VIEW_NPOS);
    if (iree_string_view_is_empty(iree_string_view_trim(next))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty pass pipeline entry");
    }
    *pipeline = next;
  } else {
    *pipeline = iree_string_view_empty();
  }
  entry = iree_string_view_trim(entry);
  if (iree_string_view_is_empty(entry)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty pass pipeline entry");
  }

  iree_host_size_t open_position = iree_string_view_find_char(entry, '{', 0);
  iree_host_size_t close_position = iree_string_view_find_char(entry, '}', 0);
  if (open_position == IREE_STRING_VIEW_NPOS) {
    if (close_position != IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unexpected '}' in pass pipeline entry");
    }
    out_entry->name = entry;
    *out_has_entry = true;
    return iree_ok_status();
  }

  if (close_position != entry.size - 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass option dictionary must terminate the pass "
                            "pipeline entry");
  }

  out_entry->name =
      iree_string_view_trim(iree_string_view_substr(entry, 0, open_position));
  out_entry->options = iree_string_view_trim(iree_string_view_substr(
      entry, open_position + 1, close_position - open_position - 1));
  if (iree_string_view_is_empty(out_entry->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass name required before option dictionary");
  }
  *out_has_entry = true;
  return iree_ok_status();
}

iree_status_t loom_pass_options_parse(iree_string_view_t pass_name,
                                      iree_string_view_t options,
                                      loom_pass_option_parse_fn_t parse,
                                      void* user_data) {
  if (iree_string_view_is_empty(iree_string_view_trim(options))) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(parse);

  iree_string_view_t remaining = options;
  while (!iree_string_view_is_empty(iree_string_view_trim(remaining))) {
    iree_string_view_t assignment = iree_string_view_empty();
    intptr_t separator_position =
        iree_string_view_split(remaining, ',', &assignment, &remaining);
    if (separator_position >= 0 &&
        iree_string_view_is_empty(iree_string_view_trim(remaining))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty option assignment in pass '%.*s'",
                              (int)pass_name.size, pass_name.data);
    }
    assignment = iree_string_view_trim(assignment);
    if (iree_string_view_is_empty(assignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty option assignment in pass '%.*s'",
                              (int)pass_name.size, pass_name.data);
    }

    iree_string_view_t name = iree_string_view_empty();
    iree_string_view_t value = iree_string_view_empty();
    if (iree_string_view_split(assignment, '=', &name, &value) < 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' must use name=value syntax",
          (int)pass_name.size, pass_name.data, (int)assignment.size,
          assignment.data);
    }

    name = iree_string_view_trim(name);
    value = iree_string_view_trim(value);
    if (iree_string_view_is_empty(name) || iree_string_view_is_empty(value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option assignments require non-empty names and values",
          (int)pass_name.size, pass_name.data);
    }
    IREE_RETURN_IF_ERROR(parse(user_data, name, value));
  }
  return iree_ok_status();
}

iree_status_t loom_pass_option_parse_uint32(iree_string_view_t pass_name,
                                            iree_string_view_t option_name,
                                            iree_string_view_t option_value,
                                            uint32_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  uint32_t value = 0;
  if (!iree_string_view_atoi_uint32(option_value, &value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' option '%.*s' expected a uint32 value, got '%.*s'",
        (int)pass_name.size, pass_name.data, (int)option_name.size,
        option_name.data, (int)option_value.size, option_value.data);
  }
  *out_value = value;
  return iree_ok_status();
}

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
                                                iree_string_view_t options,
                                                void* user_data) {
  IREE_RETURN_IF_ERROR(loom_pass_manager_ensure_capacity(manager));
  manager->entries[manager->count++] = (loom_pipeline_entry_t){
      .info = info,
      .module_run = run,
      .create = create,
      .destroy = destroy,
      .options = options,
      .user_data = user_data,
  };
  return iree_ok_status();
}

iree_status_t loom_pass_manager_add_function_pass(
    loom_pass_manager_t* manager, const loom_pass_info_t* info,
    loom_function_pass_fn_t run, loom_pass_create_fn_t create,
    loom_pass_destroy_fn_t destroy, iree_string_view_t options,
    void* user_data) {
  IREE_RETURN_IF_ERROR(loom_pass_manager_ensure_capacity(manager));
  manager->entries[manager->count++] = (loom_pipeline_entry_t){
      .info = info,
      .function_run = run,
      .create = create,
      .destroy = destroy,
      .options = options,
      .user_data = user_data,
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
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
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
      .diagnostic_emitter = manager->diagnostic_emitter,
      .user_data = entry->user_data,
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
