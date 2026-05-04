// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/ops/target.h"

#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/ops/ops.h"

enum {
  LOOM_AMDGPU_ERROR_PROCESSOR_UNKNOWN = 3,
  LOOM_AMDGPU_ERROR_PROCESSOR_NO_DESCRIPTOR_SET = 4,
  LOOM_AMDGPU_ERROR_PROCESSOR_DESCRIPTOR_SET_MISMATCH = 5,
};

static iree_string_view_t loom_amdgpu_target_record_default_processor_name(
    loom_amdgpu_target_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_TARGET_KIND_GFX942:
      return IREE_SV("gfx942");
    case LOOM_AMDGPU_TARGET_KIND_GFX950:
      return IREE_SV("gfx950");
    case LOOM_AMDGPU_TARGET_KIND_GFX1100:
      return IREE_SV("gfx1100");
    case LOOM_AMDGPU_TARGET_KIND_GFX1200:
      return IREE_SV("gfx1200");
    case LOOM_AMDGPU_TARGET_KIND_GFX1250:
      return IREE_SV("gfx1250");
    case LOOM_AMDGPU_TARGET_KIND_COUNT_:
      break;
  }
  return iree_string_view_empty();
}

static iree_string_view_t loom_amdgpu_target_record_symbol_name(
    const loom_module_t* module, const loom_op_t* target_op) {
  loom_symbol_ref_t symbol_ref = loom_amdgpu_target_symbol(target_op);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unknown>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_amdgpu_target_record_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t code,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_AMDGPU, code),
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_amdgpu_target_record_emit_unknown_processor(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor_name),
  };
  return loom_amdgpu_target_record_emit(emitter, op,
                                        LOOM_AMDGPU_ERROR_PROCESSOR_UNKNOWN,
                                        params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_target_record_emit_descriptor_set_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t record_descriptor_set) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor->processor),
      loom_param_string(processor->descriptor_set_key),
      loom_param_string(loom_amdgpu_target_record_symbol_name(module, op)),
      loom_param_string(record_descriptor_set),
  };
  return loom_amdgpu_target_record_emit(
      emitter, op, LOOM_AMDGPU_ERROR_PROCESSOR_DESCRIPTOR_SET_MISMATCH, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_target_record_emit_no_descriptor_set(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_amdgpu_processor_info_t* processor) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor->processor),
  };
  return loom_amdgpu_target_record_emit(
      emitter, op, LOOM_AMDGPU_ERROR_PROCESSOR_NO_DESCRIPTOR_SET, params,
      IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_amdgpu_target_record_string_attr(
    const loom_module_t* module, const loom_op_t* target_op,
    uint8_t attr_index) {
  loom_attribute_t attr = loom_op_attrs(target_op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return iree_string_view_empty();
  }
  loom_string_id_t string_id = loom_attr_as_string_id(attr);
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_string_view_t loom_amdgpu_target_record_processor_name(
    const loom_module_t* module, const loom_op_t* target_op) {
  iree_string_view_t explicit_processor = loom_amdgpu_target_record_string_attr(
      module, target_op, loom_amdgpu_target_processor_ATTR_INDEX);
  if (!iree_string_view_is_empty(explicit_processor)) {
    return explicit_processor;
  }
  return loom_amdgpu_target_record_default_processor_name(
      loom_amdgpu_target_kind(target_op));
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_record_processor(
    const loom_module_t* module, const loom_op_t* target_op) {
  return loom_amdgpu_target_info_find_processor(
      loom_amdgpu_target_record_processor_name(module, target_op));
}

iree_status_t loom_amdgpu_target_record_set_processor(
    loom_module_t* module, loom_op_t* target_op,
    const loom_amdgpu_processor_info_t* processor) {
  loom_string_id_t processor_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, processor->processor, &processor_id));
  loom_op_attrs(target_op)[loom_amdgpu_target_processor_ATTR_INDEX] =
      loom_attr_string(processor_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_target_record_verify(module, op, emitter));

  const iree_string_view_t processor_name =
      loom_amdgpu_target_record_processor_name(module, op);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(processor_name);
  if (processor == NULL) {
    return loom_amdgpu_target_record_emit_unknown_processor(emitter, op,
                                                            processor_name);
  }
  if (processor->descriptor_set_ordinal ==
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE ||
      iree_string_view_is_empty(processor->descriptor_set_key)) {
    return loom_amdgpu_target_record_emit_no_descriptor_set(emitter, op,
                                                            processor);
  }

  const loom_amdgpu_target_kind_t kind = loom_amdgpu_target_kind(op);
  const iree_string_view_t default_processor_name =
      loom_amdgpu_target_record_default_processor_name(kind);
  const loom_amdgpu_processor_info_t* default_processor =
      loom_amdgpu_target_info_find_processor(default_processor_name);
  if (default_processor == NULL) {
    return loom_amdgpu_target_record_emit_unknown_processor(
        emitter, op, default_processor_name);
  }
  if (!iree_string_view_equal(processor->descriptor_set_key,
                              default_processor->descriptor_set_key)) {
    return loom_amdgpu_target_record_emit_descriptor_set_mismatch(
        module, emitter, op, processor, default_processor->descriptor_set_key);
  }
  return iree_ok_status();
}
