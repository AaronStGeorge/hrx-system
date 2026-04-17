// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_low_emit(iree_diagnostic_emitter_t emitter,
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

static bool loom_low_descriptor_key_segment_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool loom_low_descriptor_key_segment_continue(char c) {
  return loom_low_descriptor_key_segment_start(c) || (c >= '0' && c <= '9');
}

static bool loom_low_descriptor_key_is_valid(iree_string_view_t key) {
  if (iree_string_view_is_empty(key)) return false;
  bool saw_separator = false;
  bool expect_segment_start = true;
  for (iree_host_size_t i = 0; i < key.size; ++i) {
    char c = key.data[i];
    if (c == '.') {
      if (expect_segment_start) return false;
      saw_separator = true;
      expect_segment_start = true;
      continue;
    }
    if (expect_segment_start) {
      if (!loom_low_descriptor_key_segment_start(c)) return false;
      expect_segment_start = false;
      continue;
    }
    if (!loom_low_descriptor_key_segment_continue(c)) return false;
  }
  return saw_separator && !expect_segment_start;
}

static iree_string_view_t loom_low_string_or_empty(const loom_module_t* module,
                                                   loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_low_emit_descriptor_key_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t key) {
  loom_diagnostic_field_ref_t attr_ref =
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(IREE_SV("opcode")), attr_ref),
      loom_param_string(key),
      loom_param_string(
          IREE_SV("a namespace-qualified descriptor key with non-empty "
                  "identifier segments")),
  };
  return loom_low_emit(emitter, op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 27),
                       params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_verify_descriptor_key(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t opcode_id,
    uint16_t attr_index) {
  iree_string_view_t key = loom_low_string_or_empty(module, opcode_id);
  if (loom_low_descriptor_key_is_valid(key)) return iree_ok_status();
  return loom_low_emit_descriptor_key_error(emitter, op, attr_index, key);
}

iree_status_t loom_low_op_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_descriptor_key(module, op, emitter,
                                        loom_low_op_opcode(op),
                                        loom_low_op_opcode_ATTR_INDEX);
}

iree_status_t loom_low_const_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_descriptor_key(module, op, emitter,
                                        loom_low_const_opcode(op),
                                        loom_low_const_opcode_ATTR_INDEX);
}
