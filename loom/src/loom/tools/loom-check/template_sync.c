// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/template_sync.h"

#include <string.h>

#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"

typedef struct loom_check_template_sync_case_t {
  // Arena-owned function symbol name used as the case key.
  iree_string_view_t key;

  // Parsed case carrying the input to synchronize.
  const loom_check_case_t* test_case;
} loom_check_template_sync_case_t;

static iree_status_t loom_check_template_sync_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_string) {
  if (source.size == 0) {
    *out_string = source;
    return iree_ok_status();
  }
  char* target_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, source.size, (void**)&target_data));
  memcpy(target_data, source.data, source.size);
  *out_string = iree_make_string_view(target_data, source.size);
  return iree_ok_status();
}

static bool loom_check_template_sync_input_is_empty(
    const loom_check_case_t* test_case) {
  return iree_string_view_is_empty(iree_string_view_trim(test_case->input));
}

static iree_status_t loom_check_template_sync_append_with_trailing_newline(
    iree_string_builder_t* builder, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, text));
  if (text.size == 0 || text.data[text.size - 1] != '\n') {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_ensure_trailing_newline(
    iree_string_builder_t* builder) {
  const iree_host_size_t size = iree_string_builder_size(builder);
  if (size > 0 && iree_string_builder_buffer(builder)[size - 1] != '\n') {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_extract_case_key(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();

  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(host_allocator, &stripped_input);
  iree_status_t status =
      loom_check_strip_comments(test_case->input, &stripped_input);

  loom_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_text_parse(iree_string_builder_view(&stripped_input),
                             filename, context, block_pool, NULL, &module);
  }
  if (iree_status_is_ok(status) && module == NULL) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template synchronization requires each case input to parse cleanly");
  }

  iree_string_view_t key = iree_string_view_empty();
  iree_host_size_t func_def_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &module->symbols.entries[i];
      if (!symbol->defining_op || !loom_func_def_isa(symbol->defining_op)) {
        continue;
      }
      ++func_def_count;
      if (symbol->name_id < module->strings.count) {
        key = module->strings.entries[symbol->name_id];
      }
    }
    if (func_def_count != 1 || iree_string_view_is_empty(key)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template synchronization requires each case to contain exactly one "
          "func.def, got %" PRIhsz,
          func_def_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_template_sync_copy_string(arena, key, out_key);
  }

  loom_module_free(module);
  iree_string_builder_deinitialize(&stripped_input);
  return status;
}

static iree_status_t loom_check_template_sync_collect_cases(
    const loom_check_file_t* file, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    bool allow_empty_cases, bool reject_case_run_directives,
    loom_check_template_sync_case_t** out_cases,
    iree_host_size_t* out_case_count) {
  *out_cases = NULL;
  *out_case_count = 0;
  if (file->case_count == 0) {
    return iree_ok_status();
  }

  loom_check_template_sync_case_t* cases = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, file->case_count, sizeof(*cases), (void**)&cases));
  iree_host_size_t case_count = 0;
  for (iree_host_size_t i = 0; i < file->case_count; ++i) {
    const loom_check_case_t* test_case = &file->cases[i];
    if (allow_empty_cases &&
        loom_check_template_sync_input_is_empty(test_case)) {
      continue;
    }
    if (reject_case_run_directives && test_case->has_run_directive) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template-synchronized target cases must inherit the file-level RUN "
          "directive");
    }
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_check_template_sync_extract_case_key(
        test_case, filename, context, block_pool, arena, host_allocator, &key));
    for (iree_host_size_t j = 0; j < case_count; ++j) {
      if (iree_string_view_equal(cases[j].key, key)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate template-synchronized case for func @%.*s",
            (int)key.size, key.data);
      }
    }
    cases[case_count++] = (loom_check_template_sync_case_t){
        .key = key,
        .test_case = test_case,
    };
  }

  *out_cases = cases;
  *out_case_count = case_count;
  return iree_ok_status();
}

static const loom_check_template_sync_case_t*
loom_check_template_sync_find_case(const loom_check_template_sync_case_t* cases,
                                   iree_host_size_t case_count,
                                   iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < case_count; ++i) {
    if (iree_string_view_equal(cases[i].key, key)) {
      return &cases[i];
    }
  }
  return NULL;
}

static iree_status_t loom_check_template_sync_append_source_range(
    iree_string_view_t source, loom_check_source_range_t range,
    iree_string_builder_t* builder) {
  if (range.end_byte < range.start_byte || range.end_byte > source.size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "template sync source range is out of bounds");
  }
  return iree_string_builder_append_string(
      builder, iree_string_view_substr(source, range.start_byte,
                                       range.end_byte - range.start_byte));
}

static iree_status_t loom_check_template_sync_append_target_directive(
    iree_string_view_t target_source, loom_check_source_range_t range,
    iree_string_builder_t* builder) {
  if (loom_check_source_range_is_empty(range)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_source_range(
      target_source, range, builder));
  return iree_string_builder_append_cstring(builder, "\n");
}

static iree_status_t loom_check_template_sync_append_target_directives(
    iree_string_view_t target_source, const loom_check_case_t* target_case,
    iree_string_builder_t* builder) {
  if (!target_case) {
    return iree_ok_status();
  }
  loom_check_source_range_t first = target_case->requires_directive_range;
  loom_check_source_range_t second = target_case->xfail_directive_range;
  if (!loom_check_source_range_is_empty(first) &&
      !loom_check_source_range_is_empty(second) &&
      second.start_byte < first.start_byte) {
    loom_check_source_range_t temporary = first;
    first = second;
    second = temporary;
  }
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_directive(
      target_source, first, builder));
  return loom_check_template_sync_append_target_directive(target_source, second,
                                                          builder);
}

static iree_status_t loom_check_template_sync_append_target_annotations(
    iree_string_view_t target_source, const loom_check_case_t* target_case,
    iree_string_builder_t* builder) {
  if (!target_case) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < target_case->annotation_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_directive(
        target_source, target_case->annotations[i].source_range, builder));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_append_case(
    iree_string_view_t target_source, const loom_check_case_t* template_case,
    const loom_check_case_t* target_case, iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_check_template_sync_ensure_trailing_newline(builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "// ====\n"));
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_directives(
      target_source, target_case, builder));
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_annotations(
      target_source, target_case, builder));
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_with_trailing_newline(
      builder, template_case->input));
  if (target_case && target_case->has_expected_section) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "// ----\n"));
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_with_trailing_newline(
        builder, target_case->expected));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_parse_template(
    iree_string_view_t template_source, iree_arena_allocator_t* arena,
    loom_check_file_t* out_template_file) {
  IREE_RETURN_IF_ERROR(
      loom_check_parse(template_source, arena, out_template_file));
  if (out_template_file->has_template_directive) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "a loom-check TEMPLATE source must not itself declare TEMPLATE");
  }
  return iree_ok_status();
}

iree_status_t loom_check_template_sync_build_source(
    iree_string_view_t target_source, const loom_check_file_t* target_file,
    iree_string_view_t target_filename, iree_string_view_t template_source,
    iree_string_view_t template_filename, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    iree_allocator_t host_allocator, iree_string_builder_t* new_source,
    bool* out_changed) {
  IREE_ASSERT_ARGUMENT(target_file);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(new_source);
  IREE_ASSERT_ARGUMENT(out_changed);
  iree_string_builder_reset(new_source);
  *out_changed = false;

  if (!target_file->has_template_directive) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template synchronization requires a TEMPLATE directive");
  }
  if (target_file->case_count == 0 ||
      loom_check_source_range_is_empty(target_file->cases[0].separator_range)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template synchronization requires a pure preamble "
                            "followed by // ====");
  }

  loom_check_file_t template_file = {0};
  IREE_RETURN_IF_ERROR(loom_check_template_sync_parse_template(
      template_source, arena, &template_file));

  loom_check_template_sync_case_t* template_cases = NULL;
  iree_host_size_t template_case_count = 0;
  IREE_RETURN_IF_ERROR(loom_check_template_sync_collect_cases(
      &template_file, template_filename, context, block_pool, arena,
      host_allocator, /*allow_empty_cases=*/false,
      /*reject_case_run_directives=*/false, &template_cases,
      &template_case_count));
  if (template_case_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template synchronization requires at least one template case");
  }

  loom_check_template_sync_case_t* target_cases = NULL;
  iree_host_size_t target_case_count = 0;
  IREE_RETURN_IF_ERROR(loom_check_template_sync_collect_cases(
      target_file, target_filename, context, block_pool, arena, host_allocator,
      /*allow_empty_cases=*/true, /*reject_case_run_directives=*/true,
      &target_cases, &target_case_count));

  const loom_check_source_range_t preamble_range = {
      .start_byte = 0,
      .end_byte = target_file->cases[0].separator_range.start_byte,
  };
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_source_range(
      target_source, preamble_range, new_source));

  for (iree_host_size_t i = 0; i < template_case_count; ++i) {
    const loom_check_template_sync_case_t* template_record = &template_cases[i];
    const loom_check_template_sync_case_t* target_record =
        loom_check_template_sync_find_case(target_cases, target_case_count,
                                           template_record->key);
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_case(
        target_source, template_record->test_case,
        target_record ? target_record->test_case : NULL, new_source));
  }

  iree_string_view_t rebuilt_source = iree_string_builder_view(new_source);
  *out_changed = !iree_string_view_equal(target_source, rebuilt_source);
  return iree_ok_status();
}
