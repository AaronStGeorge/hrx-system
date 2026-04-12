// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/storage.h"

#include "loom/error/error_defs.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"

static iree_string_view_t loom_encoding_physical_storage_name(void) {
  return IREE_SV("physical_storage");
}

static iree_string_view_t loom_encoding_layout_param_name(void) {
  return IREE_SV("layout");
}

static iree_string_view_t loom_encoding_schema_param_name(void) {
  return IREE_SV("schema");
}

static bool loom_encoding_string_id_equal(const loom_module_t* module,
                                          loom_string_id_t string_id,
                                          iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static bool loom_encoding_name_equal(const loom_module_t* module,
                                     const loom_encoding_t* encoding,
                                     iree_string_view_t expected) {
  if (!encoding) return false;
  return loom_encoding_string_id_equal(module, encoding->name_id, expected);
}

static const loom_named_attr_t* loom_encoding_find_param(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_encoding_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

static bool loom_encoding_dynamic_param_value(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, loom_value_id_t* out_value) {
  if (!name_entry || name_entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t ordinal = name_entry->value.i64;
  if (ordinal < 0 || ordinal >= params->dynamic_values.count) return false;
  *out_value = params->dynamic_values.values[ordinal];
  return true;
}

static bool loom_encoding_address_layout_op_isa(const loom_op_t* op) {
  return op && (loom_encoding_layout_dense_isa(op) ||
                loom_encoding_layout_strided_isa(op));
}

static bool loom_encoding_physical_storage_define_isa(
    const loom_module_t* module, const loom_op_t* op) {
  if (!module || !op || !loom_encoding_define_isa(op)) return false;
  const loom_encoding_t* spec =
      loom_module_encoding(module, loom_encoding_define_spec(op));
  return loom_encoding_name_equal(module, spec,
                                  loom_encoding_physical_storage_name());
}

static bool loom_encoding_resolve_address_layout_op_rec(
    const loom_module_t* module, loom_value_id_t value_id, uint8_t depth,
    const loom_op_t** out_layout_op) {
  if (!out_layout_op) return false;
  *out_layout_op = NULL;
  if (!module || depth > 4 || value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= module->values.count) {
    return false;
  }

  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_encoding(type)) return false;

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;

  const loom_op_t* op = loom_value_def_op(value);
  if (loom_encoding_address_layout_op_isa(op)) {
    *out_layout_op = op;
    return true;
  }
  if (!loom_encoding_physical_storage_define_isa(module, op)) return false;

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, op);
  const loom_named_attr_t* layout_entry = loom_encoding_find_param(
      module, params.dynamic_names, loom_encoding_layout_param_name());
  loom_value_id_t layout_value = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_dynamic_param_value(&params, layout_entry,
                                         &layout_value)) {
    return false;
  }
  return loom_encoding_resolve_address_layout_op_rec(
      module, layout_value, (uint8_t)(depth + 1), out_layout_op);
}

bool loom_encoding_resolve_address_layout_op(const loom_module_t* module,
                                             loom_value_id_t value_id,
                                             const loom_op_t** out_layout_op) {
  return loom_encoding_resolve_address_layout_op_rec(
      module, value_id, /*depth=*/0, out_layout_op);
}

static iree_status_t loom_encoding_emit(iree_diagnostic_emitter_t emitter,
                                        const loom_op_t* op,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_encoding_emit_param_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, iree_string_view_t param_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_kind_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, loom_attr_kind_t actual_kind,
    iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_u32(actual_kind),
      loom_param_string(expected_kind),
  };
  return loom_encoding_emit(emitter, op, &loom_err_encoding_010, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t param_name,
    loom_value_id_t value_id) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("encoding")),
  };
  return loom_encoding_emit(emitter, op, &loom_err_encoding_009, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, iree_string_view_t expected_role) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_string(expected_role),
  };
  return loom_encoding_emit(emitter, op, &loom_err_encoding_011, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_physical_storage_verify_static_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, const loom_named_attr_t* entry,
    iree_string_view_t param_name, loom_encoding_role_t expected_role,
    iree_string_view_t expected_role_name) {
  if (entry->value.kind != LOOM_ATTR_ENCODING) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, param_name, (loom_attr_kind_t)entry->value.kind,
        IREE_SV("encoding"));
  }

  const loom_encoding_t* nested =
      loom_module_encoding(module, loom_attr_as_encoding_id(entry->value));
  loom_encoding_role_t actual_role = loom_encoding_static_role(module, nested);
  if (actual_role != expected_role) {
    return loom_encoding_emit_role_error(emitter, op, param_name,
                                         expected_role_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter,
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* entry, iree_string_view_t param_name,
    loom_encoding_role_t expected_role, iree_string_view_t expected_role_name) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_dynamic_param_value(params, entry, &value_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter '%.*s'",
        (int)param_name.size, param_name.data);
  }

  if (!loom_type_is_encoding(loom_module_value_type(module, value_id))) {
    return loom_encoding_emit_dynamic_type_error(module, emitter, op,
                                                 param_name, value_id);
  }

  loom_encoding_role_t actual_role = loom_encoding_value_role(module, value_id);
  if (actual_role != expected_role) {
    return loom_encoding_emit_role_error(emitter, op, param_name,
                                         expected_role_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(emitter, op, &loom_err_encoding_008,
                                            param_name);
    }
  }
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(emitter, op, &loom_err_encoding_008,
                                            param_name);
    }
  }

  const loom_named_attr_t* static_layout = loom_encoding_find_param(
      module, params->static_attrs, loom_encoding_layout_param_name());
  const loom_named_attr_t* dynamic_layout = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_layout_param_name());
  if (!static_layout && !dynamic_layout) {
    return loom_encoding_emit_param_error(emitter, op, &loom_err_encoding_007,
                                          loom_encoding_layout_param_name());
  }

  const loom_named_attr_t* static_schema = loom_encoding_find_param(
      module, params->static_attrs, loom_encoding_schema_param_name());
  const loom_named_attr_t* dynamic_schema = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_schema_param_name());
  if (!static_schema && !dynamic_schema) {
    return loom_encoding_emit_param_error(emitter, op, &loom_err_encoding_007,
                                          loom_encoding_schema_param_name());
  }

  if (static_layout) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_static_param(
        module, op, emitter, static_layout, loom_encoding_layout_param_name(),
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }
  if (dynamic_layout) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_layout,
        loom_encoding_layout_param_name(), LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }

  if (static_schema) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_static_param(
        module, op, emitter, static_schema, loom_encoding_schema_param_name(),
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }
  if (dynamic_schema) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_schema,
        loom_encoding_schema_param_name(), LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }

  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_static(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  const loom_named_attr_t* layout = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_layout_param_name());
  const loom_named_attr_t* schema = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_schema_param_name());

  for (iree_host_size_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* entry = &encoding->attributes[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' does not support parameter '%.*s'",
          (int)param_name.size, param_name.data);
    }
  }

  if (layout && layout->value.kind != LOOM_ATTR_ENCODING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding 'physical_storage' parameter 'layout' must be an encoding");
  }
  if (schema && schema->value.kind != LOOM_ATTR_ENCODING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding 'physical_storage' parameter 'schema' must be an encoding");
  }

  if (layout) {
    const loom_encoding_t* layout_encoding =
        loom_module_encoding(module, loom_attr_as_encoding_id(layout->value));
    if (loom_encoding_static_role(module, layout_encoding) !=
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' parameter 'layout' must be an address "
          "layout encoding");
    }
  }

  if (schema) {
    const loom_encoding_t* schema_encoding =
        loom_module_encoding(module, loom_attr_as_encoding_id(schema->value));
    if (loom_encoding_static_role(module, schema_encoding) !=
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' parameter 'schema' must be a storage "
          "schema encoding");
    }
  }

  return iree_ok_status();
}

const loom_encoding_vtable_t loom_encoding_physical_storage_vtable = {
    .name = IREE_SVL("physical_storage"),
    .role = LOOM_ENCODING_ROLE_PHYSICAL_STORAGE,
    .verify = loom_encoding_physical_storage_verify_static,
    .verify_define = loom_encoding_physical_storage_verify_define,
};
