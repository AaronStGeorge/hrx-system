// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/tool_xml.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"
#include "loom/serve/service/text/chat_fields.h"
#include "loom/serve/service/text/tool_schemas.h"
#include "loom/util/json.h"

static iree_status_t loom_serve_text_tool_xml_quote(
    iree_string_builder_t* output, iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static iree_status_t loom_serve_text_tool_xml_consume(
    iree_string_view_t* cursor, iree_string_view_t literal) {
  *cursor = iree_string_view_trim(*cursor);
  if (!iree_string_view_consume_prefix(cursor, literal)) {
    // Bound model-generated diagnostics and keep terminal control bytes out of
    // server logs. The request still fails without publishing a partial call.
    char preview[33];
    const iree_host_size_t length = iree_min(cursor->size, sizeof(preview) - 1);
    for (iree_host_size_t i = 0; i < length; ++i) {
      const unsigned char c = (unsigned char)cursor->data[i];
      preview[i] = c >= 0x20 && c <= 0x7E ? (char)c : '.';
    }
    preview[length] = 0;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated tool call expected '%.*s'; got '%s%s'",
                            (int)literal.size, literal.data, preview,
                            cursor->size > length ? "..." : "");
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_tool_xml_name(
    iree_string_view_t* cursor, iree_string_view_t prefix,
    iree_string_view_t* out_name) {
  IREE_RETURN_IF_ERROR(loom_serve_text_tool_xml_consume(cursor, prefix));
  const iree_host_size_t end = iree_string_view_find_char(*cursor, '>', 0);
  if (end == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated generated tool name");
  }
  *out_name = iree_string_view_substr(*cursor, 0, end);
  *cursor = iree_string_view_substr(*cursor, end + 1, IREE_HOST_SIZE_MAX);
  if (!loom_serve_text_chat_is_identifier(*out_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated tool name is not an identifier");
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_tool_xml_parameter_value(
    const loom_serve_text_chat_tool_t* tool, iree_string_view_t name,
    iree_string_view_t value, iree_string_builder_t* arguments) {
  char name_storage[129];
  memcpy(name_storage, name.data, name.size);
  name_storage[name.size] = 0;
  loom_serve_text_chat_field_t property = {name_storage,
                                           LOOM_SERVE_TEXT_CHAT_OBJECT};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      tool->properties, 1, &property,
      LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN));
  if (!property.value.data) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "generated tool '%.*s' uses undeclared parameter '%.*s'",
        (int)tool->name.size, tool->name.data, (int)name.size, name.data);
  }
  loom_serve_text_chat_field_t schema[] = {
      {"type", LOOM_SERVE_TEXT_CHAT_STRING}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      property.value, IREE_ARRAYSIZE(schema), schema,
      LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&schema[0]));
  if (iree_string_view_equal(schema[0].value, IREE_SV("string"))) {
    return loom_serve_text_tool_xml_quote(arguments, value);
  }
  value = iree_string_view_trim(value);
  if (!value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty non-string tool parameter");
  }
  const iree_json_value_type_t type = iree_json_infer_value_type(value.data[0]);
  const bool integer =
      iree_string_view_equal(schema[0].value, IREE_SV("integer"));
  const bool matches =
      (type == IREE_JSON_VALUE_TYPE_NUMBER &&
       (integer ||
        iree_string_view_equal(schema[0].value, IREE_SV("number")))) ||
      ((type == IREE_JSON_VALUE_TYPE_TRUE ||
        type == IREE_JSON_VALUE_TYPE_FALSE) &&
       iree_string_view_equal(schema[0].value, IREE_SV("boolean"))) ||
      (type == IREE_JSON_VALUE_TYPE_OBJECT &&
       iree_string_view_equal(schema[0].value, IREE_SV("object"))) ||
      (type == IREE_JSON_VALUE_TYPE_ARRAY &&
       iree_string_view_equal(schema[0].value, IREE_SV("array")));
  if (!matches) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "generated parameter '%.*s' does not match schema type '%.*s'",
        (int)name.size, name.data, (int)schema[0].value.size,
        schema[0].value.data);
  }
  iree_string_view_t cursor = value;
  iree_string_view_t parsed;
  IREE_RETURN_IF_ERROR(iree_json_consume_value(&cursor, &parsed));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing data in generated tool parameter");
  }
  if (integer) {
    int64_t number = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &number));
  }
  return iree_string_builder_append_string(arguments, value);
}

typedef struct loom_serve_text_tool_xml_parameter_names_t {
  // Host allocator owning the name array.
  iree_allocator_t host_allocator;
  // Number of generated parameters already parsed.
  iree_host_size_t count;
  // Allocated name capacity.
  iree_host_size_t capacity;
  // Inline storage covering ordinary function schemas without allocation.
  iree_string_view_t inline_values[32];
  // Borrowed names in generation order, used for duplicate/required checks.
  iree_string_view_t* values;
} loom_serve_text_tool_xml_parameter_names_t;

static iree_status_t loom_serve_text_tool_xml_required_parameter(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  const loom_serve_text_tool_xml_parameter_names_t* names = user_data;
  if (type != IREE_JSON_VALUE_TYPE_STRING ||
      !loom_serve_text_chat_is_identifier(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required tool parameters must be literal names");
  }
  for (iree_host_size_t i = 0; i < names->count; ++i) {
    if (iree_string_view_equal(names->values[i], value)) {
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "generated tool call omitted required parameter '%.*s'", (int)value.size,
      value.data);
}

static iree_status_t loom_serve_text_tool_xml_parameters_parse(
    iree_string_view_t* cursor, const loom_serve_text_chat_tool_t* tool,
    iree_string_builder_t* arguments) {
  loom_serve_text_tool_xml_parameter_names_t names = {.host_allocator =
                                                          arguments->allocator};
  names.capacity = IREE_ARRAYSIZE(names.inline_values);
  names.values = names.inline_values;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(arguments, "{"));
  iree_status_t status = iree_ok_status();
  *cursor = iree_string_view_trim(*cursor);
  while (iree_string_view_starts_with(*cursor, IREE_SV("<parameter=")) &&
         iree_status_is_ok(status)) {
    iree_string_view_t name;
    status =
        loom_serve_text_tool_xml_name(cursor, IREE_SV("<parameter="), &name);
    for (iree_host_size_t i = 0; i < names.count && iree_status_is_ok(status);
         ++i) {
      if (iree_string_view_equal(names.values[i], name)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "generated tool '%.*s' repeats parameter '%.*s'",
            (int)tool->name.size, tool->name.data, (int)name.size, name.data);
      }
    }
    iree_host_size_t required_capacity = 0;
    if (iree_status_is_ok(status) &&
        !iree_host_size_checked_add(names.count, 1, &required_capacity)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter name capacity overflow");
    }
    if (iree_status_is_ok(status) && required_capacity > names.capacity) {
      iree_string_view_t* values =
          names.values == names.inline_values ? NULL : names.values;
      status = iree_allocator_grow_array(
          names.host_allocator, required_capacity, sizeof(*names.values),
          &names.capacity, (void**)&values);
      if (iree_status_is_ok(status) && names.values == names.inline_values) {
        memcpy(values, names.inline_values,
               names.count * sizeof(*names.values));
      }
      if (iree_status_is_ok(status)) {
        names.values = values;
      }
    }
    const iree_string_view_t end_tag = IREE_SV("</parameter>");
    const iree_host_size_t end = iree_string_view_find(*cursor, end_tag, 0);
    if (iree_status_is_ok(status) && end == IREE_STRING_VIEW_NPOS) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unterminated generated parameter");
    }
    if (iree_status_is_ok(status)) {
      iree_string_view_t value = iree_string_view_substr(*cursor, 0, end);
      // Remove the template's one framing newline, preserving string content.
      if (!iree_string_view_consume_prefix(&value, IREE_SV("\r\n"))) {
        iree_string_view_consume_prefix(&value, IREE_SV("\n"));
      }
      if (!iree_string_view_consume_suffix(&value, IREE_SV("\r\n"))) {
        iree_string_view_consume_suffix(&value, IREE_SV("\n"));
      }
      if (names.count) {
        status = iree_string_builder_append_cstring(arguments, ",");
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_text_tool_xml_quote(arguments, name);
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(arguments, ":");
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_text_tool_xml_parameter_value(tool, name, value,
                                                          arguments);
      }
      if (iree_status_is_ok(status)) {
        names.values[names.count++] = name;
        *cursor = iree_string_view_trim(iree_string_view_substr(
            *cursor, end + end_tag.size, IREE_HOST_SIZE_MAX));
      }
    }
  }
  if (iree_status_is_ok(status) && tool->required.data) {
    status = iree_json_enumerate_array_typed(
        tool->required, loom_serve_text_tool_xml_required_parameter, &names);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(arguments, "}");
  }
  if (names.values != names.inline_values) {
    iree_allocator_free(names.host_allocator, names.values);
  }
  return status;
}

static iree_status_t loom_serve_text_tool_xml_parse(
    const loom_serve_text_tool_schemas_t* schemas, iree_string_view_t text,
    iree_string_builder_t* tool_calls) {
  iree_host_size_t tool_count = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(tool_calls, "["));
  iree_string_view_t cursor = iree_string_view_trim(text);
  iree_string_builder_t arguments;
  iree_string_builder_initialize(tool_calls->allocator, &arguments);
  iree_status_t status = iree_ok_status();
  while (cursor.size && iree_status_is_ok(status)) {
    status = loom_serve_text_tool_xml_consume(&cursor, IREE_SV("<tool_call>"));
    iree_string_view_t name = iree_string_view_empty();
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_text_tool_xml_name(&cursor, IREE_SV("<function="), &name);
    }
    const loom_serve_text_chat_tool_t* tool =
        loom_serve_text_tool_schemas_find(schemas, name);
    if (iree_status_is_ok(status) &&
        tool_count >= LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "too many generated tool calls");
    }
    if (iree_status_is_ok(status) && !tool) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "generated tool '%.*s' is not declared",
                                (int)name.size, name.data);
    }
    iree_string_builder_reset(&arguments);
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_text_tool_xml_parameters_parse(&cursor, tool, &arguments);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_text_tool_xml_consume(&cursor, IREE_SV("</function>"));
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_text_tool_xml_consume(&cursor, IREE_SV("</tool_call>"));
    }
    if (iree_status_is_ok(status)) {
      const iree_host_size_t index = tool_count;
      status = iree_string_builder_append_format(
          tool_calls,
          "%s{\"type\":\"function\",\"function\":{\"name\":", index ? "," : "");
      if (iree_status_is_ok(status)) {
        status = loom_serve_text_tool_xml_quote(tool_calls, name);
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_string_builder_append_cstring(tool_calls, ",\"arguments\":");
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_text_tool_xml_quote(
            tool_calls, iree_string_builder_view(&arguments));
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(tool_calls, "}}");
      }
      ++tool_count;
      cursor = iree_string_view_trim(cursor);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(tool_calls, "]");
  }
  iree_string_builder_deinitialize(&arguments);
  return status;
}

typedef struct loom_serve_text_tool_xml_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Descriptor borrowing the environment's reference types.
  iree_vm_module_descriptor_t descriptor;
  // Host allocation policy.
  iree_allocator_t host_allocator;
  // Canonical buffer reference identity.
  iree_vm_ref_types_t types;
} loom_serve_text_tool_xml_module_t;

static iree_status_t loom_serve_text_tool_xml_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_text_tool_xml_module_t* module =
      (loom_serve_text_tool_xml_module_t*)base;
  iree_string_view_t inputs[2];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_borrow(&params->call, i, &ref);
    if (!ref.object) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tool input is null");
    }
    iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)ref.object;
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
        buffer, 0, iree_vm_buffer_length(buffer), &bytes));
    inputs[i] =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!iree_unicode_utf8_validate(inputs[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tool input must be UTF-8");
    }
  }
  loom_serve_text_tool_schemas_t schemas;
  iree_string_builder_t output;
  iree_string_builder_initialize(module->host_allocator, &output);
  iree_status_t status = loom_serve_text_tool_schemas_initialize(
      inputs[0], module->host_allocator, &schemas);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_tool_xml_parse(&schemas, inputs[1], &output);
  }
  iree_vm_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_string_view_t value = iree_string_builder_view(&output);
    status =
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(value.data, value.size),
                             1, module->host_allocator, &buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t result =
        iree_vm_buffer_ref_from_ptr_move(&module->types, &buffer);
    iree_vm_call_ref_result_store_move(&params->call, 0, &result);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  }
  iree_vm_buffer_release(buffer);
  iree_string_builder_deinitialize(&output);
  loom_serve_text_tool_schemas_deinitialize(&schemas);
  return status;
}

static const iree_vm_module_signature_type_t
    loom_serve_text_tool_xml_arguments[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t
    loom_serve_text_tool_xml_results[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static void loom_serve_text_tool_xml_destroy(iree_vm_module_t* base) {
  loom_serve_text_tool_xml_module_t* module =
      (loom_serve_text_tool_xml_module_t*)base;
  iree_allocator_free(module->host_allocator, module);
}
static void loom_serve_text_tool_xml_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = (iree_vm_module_export_declaration_t){
      .export_name = IREE_SVL("parse_xml"),
      .callable_type_ordinal = 0,
      .function_ordinal = 0};
}
static void loom_serve_text_tool_xml_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = (iree_vm_module_callable_type_declaration_t){
      .signature = {
          .arguments = {loom_serve_text_tool_xml_arguments, 2, 0, 2, 0},
          .results = {loom_serve_text_tool_xml_results, 1, 0, 1, 0}}};
}
static void loom_serve_text_tool_xml_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("tools module has no imports");
}
static void loom_serve_text_tool_xml_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("tools module has no imports");
}
static const iree_vm_module_vtable_t loom_serve_text_tool_xml_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_serve_text_tool_xml_destroy,
    .function_start = loom_serve_text_tool_xml_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_serve_text_tool_xml_query_import_group,
    .query_import = loom_serve_text_tool_xml_query_import,
    .query_export = loom_serve_text_tool_xml_query_export,
    .query_callable_type = loom_serve_text_tool_xml_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_text_tool_xml_module_create(
    iree_vm_environment_t* environment, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  loom_serve_text_tool_xml_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->types = types;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("tools"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->types.buffer, 1},
      .counts = {.function_count = 1,
                 .callable_type_count = 1,
                 .export_count = 1,
                 .callable_fields = {.value_count = 0, .ref_count = 3}}};
  iree_status_t status = iree_vm_module_initialize(
      &loom_serve_text_tool_xml_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
