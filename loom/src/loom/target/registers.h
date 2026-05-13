// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target register payload helpers.
//
// Core IR owns only the by-value type storage and structural equality. The
// target-low layer owns the interpretation of LOOM_TYPE_REGISTER payload
// fields: a register-class identity plus the number of contiguous class units
// carried by the SSA value.

#ifndef LOOM_TARGET_REGISTERS_H_
#define LOOM_TARGET_REGISTERS_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |type| carries a target-low register payload.
static inline bool loom_low_type_is_register(loom_type_t type) {
  return loom_type_is_register(type);
}

// Creates a target-low register type from a descriptor register-class name ID.
// |unit_count| is the number of register-class units carried by the value.
static inline loom_type_t loom_low_register_type(
    loom_string_id_t register_class_name_id, uint32_t unit_count) {
  return loom_type_register(register_class_name_id, unit_count);
}

// Returns the module string ID for the target-low register class name.
static inline loom_string_id_t loom_low_register_type_class_name_id(
    loom_type_t type) {
  return loom_type_register_class_id(type);
}

// Returns the number of register-class units carried by |type|.
static inline uint32_t loom_low_register_type_unit_count(loom_type_t type) {
  return loom_type_register_unit_count(type);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REGISTERS_H_
