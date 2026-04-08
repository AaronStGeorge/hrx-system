// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/verify.h"

#include "loom/error/diagnostic.h"
#include "loom/error/renderer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/util/stream.h"

//===----------------------------------------------------------------------===//
// Diagnostic collector
//===----------------------------------------------------------------------===//

// One collected diagnostic from either the parser or verifier.
typedef struct loom_collected_diagnostic_t {
  loom_diagnostic_severity_t severity;
  loom_error_domain_t domain;
  uint16_t code;
  // 1-based source line where the diagnostic was emitted. 0 if unknown.
  uint32_t origin_line;
  // Rendered message text, arena-allocated.
  iree_string_view_t message;
  // Set during the matching pass.
  bool matched;
} loom_collected_diagnostic_t;

// Accumulates diagnostics emitted during parse and verify phases.
// All storage is arena-allocated (no per-entry cleanup needed).
typedef struct loom_diagnostic_collector_t {
  loom_collected_diagnostic_t* diagnostics;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_arena_allocator_t* arena;
  iree_allocator_t host_allocator;
} loom_diagnostic_collector_t;

// Grows the diagnostics array by doubling capacity. Arena-allocated.
static iree_status_t loom_collector_grow(
    loom_diagnostic_collector_t* collector) {
  iree_host_size_t new_capacity =
      collector->capacity == 0 ? 16 : collector->capacity * 2;
  loom_collected_diagnostic_t* new_diagnostics = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      collector->arena, new_capacity * sizeof(loom_collected_diagnostic_t),
      (void**)&new_diagnostics));
  if (collector->count > 0) {
    memcpy(new_diagnostics, collector->diagnostics,
           collector->count * sizeof(loom_collected_diagnostic_t));
  }
  collector->diagnostics = new_diagnostics;
  collector->capacity = new_capacity;
  return iree_ok_status();
}

// Diagnostic sink callback. Renders the message, stores the entry.
static iree_status_t loom_collector_sink(void* user_data,
                                         const loom_diagnostic_t* diagnostic) {
  loom_diagnostic_collector_t* collector =
      (loom_diagnostic_collector_t*)user_data;
  if (collector->count >= collector->capacity) {
    IREE_RETURN_IF_ERROR(loom_collector_grow(collector));
  }

  // Render the message template into a temporary builder, then copy
  // the result into the arena for zero-cost ownership.
  iree_string_builder_t message_builder;
  iree_string_builder_initialize(collector->host_allocator, &message_builder);
  loom_output_stream_t message_stream;
  loom_output_stream_for_builder(&message_builder, &message_stream);
  loom_type_formatter_t type_formatter = {loom_type_format_minimal, NULL};
  iree_status_t render_status = loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count,
      type_formatter, &message_stream);
  if (!iree_status_is_ok(render_status)) {
    iree_string_builder_deinitialize(&message_builder);
    return render_status;
  }

  // Arena-copy the rendered message so it outlives the builder.
  iree_string_view_t rendered = iree_string_builder_view(&message_builder);
  char* arena_copy = NULL;
  iree_status_t copy_status =
      iree_arena_allocate(collector->arena, rendered.size, (void**)&arena_copy);
  if (!iree_status_is_ok(copy_status)) {
    iree_string_builder_deinitialize(&message_builder);
    return copy_status;
  }
  memcpy(arena_copy, rendered.data, rendered.size);
  iree_string_builder_deinitialize(&message_builder);

  collector->diagnostics[collector->count++] = (loom_collected_diagnostic_t){
      .severity = diagnostic->severity,
      .domain = diagnostic->error->domain,
      .code = diagnostic->error->code,
      .origin_line = diagnostic->origin.start_line,
      .message = iree_make_string_view(arena_copy, rendered.size),
  };
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Annotation matching
//===----------------------------------------------------------------------===//

// Returns true if a collected diagnostic matches an annotation.
static bool loom_diagnostic_matches_annotation(
    const loom_collected_diagnostic_t* diagnostic,
    const loom_check_annotation_t* annotation) {
  // Severity must match exactly.
  if (diagnostic->severity != annotation->severity) return false;

  // Domain: LOOM_ERROR_DOMAIN_COUNT_ is the wildcard sentinel.
  if (annotation->domain != LOOM_ERROR_DOMAIN_COUNT_ &&
      diagnostic->domain != annotation->domain) {
    return false;
  }

  // Code: 0 is the wildcard.
  if (annotation->code != 0 && diagnostic->code != annotation->code) {
    return false;
  }

  // Line: must match exactly. Annotation target_line is 1-based,
  // diagnostic origin_line is 1-based (0 means unknown — never matches
  // a targeted annotation).
  if (diagnostic->origin_line != (uint32_t)annotation->target_line) {
    return false;
  }

  // Substrings: zero count means "match any message"; non-zero count
  // requires every entry to appear somewhere in the diagnostic message
  // (order independent).
  for (uint8_t i = 0; i < annotation->message_substring_count; ++i) {
    if (iree_string_view_find(diagnostic->message,
                              annotation->message_substrings[i],
                              0) == IREE_STRING_VIEW_NPOS) {
      return false;
    }
  }

  return true;
}

// Greedy O(A×D) matching. For each annotation, find the first unmatched
// diagnostic that matches. Greedy is correct because annotations are
// line-targeted — ambiguity is effectively impossible in well-written tests.
static void loom_match_annotations(loom_collected_diagnostic_t* diagnostics,
                                   iree_host_size_t diagnostic_count,
                                   const loom_check_annotation_t* annotations,
                                   iree_host_size_t annotation_count,
                                   bool* out_annotation_matched) {
  for (iree_host_size_t a = 0; a < annotation_count; ++a) {
    out_annotation_matched[a] = false;
    for (iree_host_size_t d = 0; d < diagnostic_count; ++d) {
      if (diagnostics[d].matched) continue;
      if (loom_diagnostic_matches_annotation(&diagnostics[d],
                                             &annotations[a])) {
        diagnostics[d].matched = true;
        out_annotation_matched[a] = true;
        break;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Result assembly
//===----------------------------------------------------------------------===//

// Appends detail for all unmatched annotations and unexpected diagnostics.
static iree_status_t loom_assemble_verify_detail(
    const loom_collected_diagnostic_t* diagnostics,
    iree_host_size_t diagnostic_count,
    const loom_check_annotation_t* annotations,
    iree_host_size_t annotation_count, const bool* annotation_matched,
    iree_string_builder_t* detail) {
  // Report unmatched annotations.
  for (iree_host_size_t a = 0; a < annotation_count; ++a) {
    if (annotation_matched[a]) continue;
    const loom_check_annotation_t* ann = &annotations[a];
    const char* severity_name = loom_diagnostic_severity_name(ann->severity);
    if (ann->domain != LOOM_ERROR_DOMAIN_COUNT_ && ann->code != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s %s/%03u",
          ann->target_line, severity_name, loom_error_domain_name(ann->domain),
          ann->code));
    } else if (ann->domain != LOOM_ERROR_DOMAIN_COUNT_) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s %s/*",
          ann->target_line, severity_name,
          loom_error_domain_name(ann->domain)));
    } else if (ann->code != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s */%03u",
          ann->target_line, severity_name, ann->code));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s",
          ann->target_line, severity_name));
    }
    for (uint8_t i = 0; i < ann->message_substring_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, " \"%.*s\"", (int)ann->message_substrings[i].size,
          ann->message_substrings[i].data));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(detail, "\n"));
  }

  // Report unexpected diagnostics.
  for (iree_host_size_t d = 0; d < diagnostic_count; ++d) {
    if (diagnostics[d].matched) continue;
    const loom_collected_diagnostic_t* diag = &diagnostics[d];
    const char* severity_name = loom_diagnostic_severity_name(diag->severity);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        detail, "unexpected %s at line %u: [%s/%03u] %.*s\n", severity_name,
        diag->origin_line, loom_error_domain_name(diag->domain), diag->code,
        (int)diag->message.size, diag->message.data));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Verify execution
//===----------------------------------------------------------------------===//

iree_status_t loom_check_execute_verify(const loom_check_case_t* test_case,
                                        iree_string_view_t filename,
                                        loom_context_t* context,
                                        iree_arena_block_pool_t* block_pool,
                                        iree_allocator_t allocator,
                                        loom_check_result_t* result) {
  // Arena for collector storage (diagnostics array, message copies).
  // Deinitialized at the end — no per-entry cleanup needed.
  iree_arena_allocator_t collector_arena;
  iree_arena_initialize(block_pool, &collector_arena);

  loom_diagnostic_collector_t collector = {
      .arena = &collector_arena,
      .host_allocator = allocator,
  };

  // Strip standalone comment lines from input. Comments become blank
  // lines to preserve line count for diagnostic source locations.
  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  iree_status_t status =
      loom_check_strip_comments(test_case->input, &stripped_input);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&collector_arena);
    return status;
  }

  // Parse the stripped input with the collector as the diagnostic sink.
  // Parse errors are emitted as diagnostics (not status failures).
  loom_module_t* module = NULL;
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_collector_sink, .user_data = &collector},
      .max_errors = 100,
  };
  iree_string_view_t stripped_view = iree_string_builder_view(&stripped_input);
  status = loom_text_parse(stripped_view, filename, context, block_pool,
                           &parse_options, &module);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&collector_arena);
    return status;
  }

  // If parsing succeeded (module != NULL), run the verifier to collect
  // additional diagnostics. The source resolver uses the stripped input
  // so verifier diagnostics get the same line numbers as parse diagnostics.
  if (module) {
    loom_source_entry_t source_entry = {
        .source_id = 0,
        .source = stripped_view,
        .filename = filename,
    };
    loom_source_table_resolver_t resolver_data = {
        .entries = &source_entry,
        .count = 1,
    };
    loom_verify_options_t verify_options = {
        .sink = {.fn = loom_collector_sink, .user_data = &collector},
        .max_errors = 100,
        .source_resolver = {.fn = loom_source_table_resolve,
                            .user_data = &resolver_data},
    };

    loom_verify_result_t verify_result;
    status = loom_verify_module(module, &verify_options, &verify_result);
    loom_module_free(module);
    if (!iree_status_is_ok(status)) {
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&collector_arena);
      return status;
    }
  }

  // The stripped input builder is no longer needed (diagnostics already
  // collected with arena-copied messages).
  iree_string_builder_deinitialize(&stripped_input);

  // Match collected diagnostics against annotations.
  bool* annotation_matched = NULL;
  if (test_case->annotation_count > 0) {
    annotation_matched =
        (bool*)iree_alloca(test_case->annotation_count * sizeof(bool));
  }
  loom_match_annotations(collector.diagnostics, collector.count,
                         test_case->annotations, test_case->annotation_count,
                         annotation_matched);

  // Determine outcome: all annotations matched and no unexpected diagnostics.
  bool all_annotations_matched = true;
  for (iree_host_size_t a = 0; a < test_case->annotation_count; ++a) {
    if (!annotation_matched[a]) {
      all_annotations_matched = false;
      break;
    }
  }
  bool all_diagnostics_matched = true;
  for (iree_host_size_t d = 0; d < collector.count; ++d) {
    if (!collector.diagnostics[d].matched) {
      all_diagnostics_matched = false;
      break;
    }
  }

  if (all_annotations_matched && all_diagnostics_matched) {
    result->raw_outcome = LOOM_CHECK_PASS;
  } else {
    result->raw_outcome = LOOM_CHECK_FAIL;
    status = loom_assemble_verify_detail(
        collector.diagnostics, collector.count, test_case->annotations,
        test_case->annotation_count, annotation_matched, &result->detail);
  }

  iree_arena_deinitialize(&collector_arena);
  return status;
}
