// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/tooling/options.h"

#include "iree/base/internal/path.h"
#include "loom/tooling/io/file.h"
#include "loomc/iree.h"

static iree_status_t loom_cxx_input_token(iree_string_view_t* remaining,
                                          iree_string_builder_t* token) {
  iree_string_builder_reset(token);
  *remaining = iree_string_view_trim(*remaining);
  char quote = 0;
  while (remaining->size) {
    char c = remaining->data[0];
    *remaining = iree_string_view_substr(*remaining, 1, IREE_HOST_SIZE_MAX);
    if (quote) {
      if (c == quote) {
        quote = 0;
        continue;
      }
      if (c == '\\' && remaining->size &&
          (remaining->data[0] == quote || remaining->data[0] == '\\')) {
        c = remaining->data[0];
        *remaining = iree_string_view_substr(*remaining, 1, IREE_HOST_SIZE_MAX);
      }
    } else if (c == '\'' || c == '"') {
      quote = c;
      continue;
    } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      break;
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(token, iree_make_string_view(&c, 1)));
  }
  if (quote) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated quote in C/C++ input options");
  }
  return iree_ok_status();
}

iree_status_t loom_cxx_input_parse_options(
    iree_string_view_t source_path, iree_string_view_t input_options,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    loomc_cxx_import_options_t* out_options) {
  *out_options = (loomc_cxx_import_options_t){
      .type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS,
      .structure_size = sizeof(*out_options),
  };
  iree_string_view_t* tokens = NULL;
  iree_host_size_t count = 0;
  iree_host_size_t capacity = 0;
  iree_string_builder_t token;
  iree_string_builder_initialize(host_allocator, &token);
  iree_string_view_t remaining = iree_string_view_trim(input_options);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && remaining.size) {
    status = loom_cxx_input_token(&remaining, &token);
    if (iree_status_is_ok(status) && count == capacity) {
      status = iree_arena_grow_array(arena, count, count + 1, sizeof(*tokens),
                                     &capacity, (void**)&tokens);
    }
    if (iree_status_is_ok(status)) {
      char* storage = NULL;
      status = iree_arena_allocate(arena, token.size, (void**)&storage);
      if (iree_status_is_ok(status)) {
        memcpy(storage, token.buffer, token.size);
        tokens[count++] = iree_make_string_view(storage, token.size);
      }
    }
    remaining = iree_string_view_trim(remaining);
  }
  iree_string_builder_deinitialize(&token);
  IREE_RETURN_IF_ERROR(status);

  loomc_string_view_t* include_paths = NULL;
  loomc_string_view_t* system_paths = NULL;
  loomc_string_view_t* roots = NULL;
  loomc_cxx_define_t* defines = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*include_paths), (void**)&include_paths));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*system_paths), (void**)&system_paths));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*roots), (void**)&roots));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*defines),
                                                 (void**)&defines));
  out_options->include_paths = include_paths;
  out_options->system_include_paths = system_paths;
  out_options->roots = roots;
  out_options->defines = defines;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_string_view_t name, value;
    if (iree_string_view_split(tokens[i], '=', &name, &value) < 0 ||
        iree_string_view_is_empty(value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "expected C/C++ input option key=value, got '%.*s'",
          (int)tokens[i].size, tokens[i].data);
    }
    if (iree_string_view_equal(name, IREE_SV("std"))) {
      out_options->standard = loomc_string_view_from_iree(value);
    } else if (iree_string_view_equal(name, IREE_SV("triple"))) {
      out_options->triple = loomc_string_view_from_iree(value);
    } else if (iree_string_view_equal(name, IREE_SV("data-model"))) {
      if (iree_string_view_equal(value, IREE_SV("lp64"))) {
        out_options->data_model = LOOMC_CXX_DATA_MODEL_LP64;
      } else if (iree_string_view_equal(value, IREE_SV("llp64"))) {
        out_options->data_model = LOOMC_CXX_DATA_MODEL_LLP64;
      } else if (iree_string_view_equal(value, IREE_SV("ilp32"))) {
        out_options->data_model = LOOMC_CXX_DATA_MODEL_ILP32;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown source data model '%.*s'",
                                (int)value.size, value.data);
      }
    } else if (iree_string_view_equal(name, IREE_SV("fp-model"))) {
      if (iree_string_view_equal(value, IREE_SV("precise"))) {
        out_options->fp_model = LOOMC_CXX_FP_MODEL_PRECISE;
      } else if (iree_string_view_equal(value, IREE_SV("fast"))) {
        out_options->fp_model = LOOMC_CXX_FP_MODEL_FAST;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown floating-point model '%.*s'",
                                (int)value.size, value.data);
      }
    } else if (iree_string_view_equal(name, IREE_SV("root"))) {
      roots[out_options->root_count++] = loomc_string_view_from_iree(value);
    } else if (iree_string_view_equal(name, IREE_SV("I")) ||
               iree_string_view_equal(name, IREE_SV("isystem"))) {
      char* path = NULL;
      IREE_RETURN_IF_ERROR(loom_tooling_file_path_join(
          iree_file_path_dirname(source_path), value,
          iree_arena_allocator(arena), &path));
      if (iree_string_view_equal(name, IREE_SV("I"))) {
        include_paths[out_options->include_path_count++] =
            loomc_make_cstring_view(path);
      } else {
        system_paths[out_options->system_include_path_count++] =
            loomc_make_cstring_view(path);
      }
    } else if (iree_string_view_equal(name, IREE_SV("D"))) {
      iree_string_view_t define_name, define_value;
      if (iree_string_view_split(value, '=', &define_name, &define_value) < 0) {
        define_value = IREE_SV("1");
      }
      defines[out_options->define_count++] = (loomc_cxx_define_t){
          .name = loomc_string_view_from_iree(define_name),
          .value = loomc_string_view_from_iree(define_value),
      };
    } else if (iree_string_view_equal(name, IREE_SV("builtin-includes"))) {
      bool enabled = iree_string_view_equal(value, IREE_SV("true"));
      if (!enabled && !iree_string_view_equal(value, IREE_SV("false"))) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "C/C++ input option '%.*s' expects true or false", (int)name.size,
            name.data);
      }
      if (enabled) {
        out_options->flags &= ~LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
      } else {
        out_options->flags |= LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown C/C++ input option '%.*s'",
                              (int)name.size, name.data);
    }
  }
  return iree_ok_status();
}

static const iree_string_view_t loom_cxx_input_suffixes[] = {
    IREE_SVL(".cxx-test"), IREE_SVL(".cxx"), IREE_SVL(".cpp"),
    IREE_SVL(".cc"),       IREE_SVL(".c"),
};

const loom_input_format_t loom_cxx_input_format = {
    .name = IREE_SVL("cxx"),
    .suffixes = {IREE_ARRAYSIZE(loom_cxx_input_suffixes),
                 loom_cxx_input_suffixes},
};
