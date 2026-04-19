// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/types.h"

static iree_string_view_t loom_wasm_module_string(const loom_module_t* module,
                                                  loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_status_t loom_wasm_value_type_from_register_class(
    const loom_module_t* module, loom_string_id_t register_class_id,
    loom_wasm_value_type_t* out_value_type) {
  iree_string_view_t register_class =
      loom_wasm_module_string(module, register_class_id);
  if (iree_string_view_equal(register_class, IREE_SV("wasm.i32"))) {
    *out_value_type = LOOM_WASM_VALUE_TYPE_I32;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class, IREE_SV("wasm.i64"))) {
    *out_value_type = LOOM_WASM_VALUE_TYPE_I64;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class, IREE_SV("wasm.f32"))) {
    *out_value_type = LOOM_WASM_VALUE_TYPE_F32;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class, IREE_SV("wasm.f64"))) {
    *out_value_type = LOOM_WASM_VALUE_TYPE_F64;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class, IREE_SV("wasm.v128"))) {
    *out_value_type = LOOM_WASM_VALUE_TYPE_V128;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported Wasm register class '%.*s'",
                          (int)register_class.size, register_class.data);
}

iree_status_t loom_wasm_value_type_from_register_type(
    const loom_module_t* module, loom_type_t type,
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
  return loom_wasm_value_type_from_register_class(
      module, loom_type_register_class_id(type), out_value_type);
}

iree_status_t loom_wasm_value_type_from_value(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_wasm_value_type_t* out_value_type) {
  return loom_wasm_value_type_from_register_type(
      module, loom_module_value_type(module, value_id), out_value_type);
}
