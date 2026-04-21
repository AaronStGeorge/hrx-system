// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/types.h"

#include <inttypes.h>

#include "loom/ir/module.h"
#include "loom/target/arch/wasm/descriptors.h"

static iree_string_view_t loom_wasm_module_string(const loom_module_t* module,
                                                  loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_status_t loom_wasm_value_type_from_descriptor_register_class(
    uint16_t descriptor_reg_class_id, loom_wasm_value_type_t* out_value_type) {
  switch (descriptor_reg_class_id) {
    case WASM_CORE_SIMD128_REG_CLASS_ID_WASM_I32:
      *out_value_type = LOOM_WASM_VALUE_TYPE_I32;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_WASM_I64:
      *out_value_type = LOOM_WASM_VALUE_TYPE_I64;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_WASM_F32:
      *out_value_type = LOOM_WASM_VALUE_TYPE_F32;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_WASM_F64:
      *out_value_type = LOOM_WASM_VALUE_TYPE_F64;
      return iree_ok_status();
    case WASM_CORE_SIMD128_REG_CLASS_ID_WASM_V128:
      *out_value_type = LOOM_WASM_VALUE_TYPE_V128;
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "unsupported Wasm descriptor register class ID %" PRIu16,
          descriptor_reg_class_id);
  }
}

iree_status_t loom_wasm_value_type_from_register_type(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    loom_wasm_value_type_t* out_value_type) {
  if (!loom_type_is_register(type)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Wasm emission requires register-typed values");
  }
  if (loom_type_register_unit_count(type) != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm emission requires single-unit register-typed values");
  }
  const iree_string_view_t register_class_name =
      loom_wasm_module_string(module, loom_type_register_class_id(type));
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class = &descriptor_set->reg_classes[i];
    iree_string_view_t descriptor_register_class_name =
        iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset,
        &descriptor_register_class_name));
    if (!iree_string_view_equal(register_class_name,
                                descriptor_register_class_name)) {
      continue;
    }
    return loom_wasm_value_type_from_descriptor_register_class((uint16_t)i,
                                                               out_value_type);
  }
  iree_string_view_t descriptor_set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &descriptor_set_key));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "unsupported Wasm register class '%.*s' for descriptor set '%.*s'",
      (int)register_class_name.size, register_class_name.data,
      (int)descriptor_set_key.size, descriptor_set_key.data);
}
