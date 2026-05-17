// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-tune-loom/module_query.h"

#include "loom/ops/special_values.h"

iree_string_view_t iree_tune_loom_module_string(const loom_module_t* module,
                                                loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_string_view_t iree_tune_loom_value_name(const loom_module_t* module,
                                             loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = module->values.entries[value_id].name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[name_id];
}

iree_string_view_t iree_tune_loom_normalize_symbol_name(
    iree_string_view_t symbol_name) {
  symbol_name = iree_string_view_trim(symbol_name);
  if (iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  }
  return symbol_name;
}

iree_status_t iree_tune_loom_module_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u is outside the module symbol table",
                            (unsigned)ref.symbol_id);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u has an invalid name",
                            (unsigned)ref.symbol_id);
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

const loom_named_attr_t* iree_tune_loom_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            iree_tune_loom_module_string(module, attr->name_id), name)) {
      return attr;
    }
  }
  return NULL;
}

iree_status_t iree_tune_loom_read_optional_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, int64_t default_value, int64_t* out_value) {
  const loom_named_attr_t* attr =
      iree_tune_loom_find_named_attr(module, attrs, name);
  if (!attr) {
    *out_value = default_value;
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark attr '%.*s' must be i64", (int)name.size,
                            name.data);
  }
  *out_value = attr->value.i64;
  return iree_ok_status();
}

iree_status_t iree_tune_loom_read_optional_bool_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool default_value, bool* out_value) {
  const loom_named_attr_t* attr =
      iree_tune_loom_find_named_attr(module, attrs, name);
  if (!attr) {
    *out_value = default_value;
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_BOOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark attr '%.*s' must be bool",
                            (int)name.size, name.data);
  }
  *out_value = loom_attr_as_bool(attr->value);
  return iree_ok_status();
}

iree_status_t iree_tune_loom_read_i64_policy_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, const iree_tune_loom_i32_flag_t* command_line_flag,
    int64_t* out_value) {
  if (command_line_flag->specified) {
    *out_value = command_line_flag->value;
    return iree_ok_status();
  }
  return iree_tune_loom_read_optional_i64_attr(
      module, attrs, name, command_line_flag->value, out_value);
}

iree_status_t iree_tune_loom_read_bool_policy_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name,
    const iree_tune_loom_bool_flag_t* command_line_flag, bool* out_value) {
  if (command_line_flag->specified) {
    *out_value = command_line_flag->value;
    return iree_ok_status();
  }
  return iree_tune_loom_read_optional_bool_attr(
      module, attrs, name, command_line_flag->value, out_value);
}

iree_status_t iree_tune_loom_read_optional_string_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, iree_string_view_t default_value,
    iree_string_view_t* out_value) {
  const loom_named_attr_t* attr =
      iree_tune_loom_find_named_attr(module, attrs, name);
  if (!attr) {
    *out_value = default_value;
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark attr '%.*s' must be string",
                            (int)name.size, name.data);
  }
  *out_value =
      iree_tune_loom_module_string(module, loom_attr_as_string_id(attr->value));
  return iree_ok_status();
}
