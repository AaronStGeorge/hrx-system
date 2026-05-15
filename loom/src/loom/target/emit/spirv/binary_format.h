// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V binary-format constants used by target-local emission.
//
// These are numeric wire values from the SPIR-V specification. They stay in the
// emitter package instead of the core IR so the rest of Loom does not grow a
// dependency on SPIR-V instruction vocabulary.

#ifndef LOOM_TARGET_EMIT_SPIRV_BINARY_FORMAT_H_
#define LOOM_TARGET_EMIT_SPIRV_BINARY_FORMAT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_SPIRV_MAGIC_NUMBER = 0x07230203u,
  LOOM_SPIRV_SCHEMA_RESERVED = 0u,
  LOOM_SPIRV_GENERATOR_LOOM = 0u,
};

typedef enum loom_spirv_op_e {
  LOOM_SPIRV_OP_NAME = 5,
  LOOM_SPIRV_OP_EXTENSION = 10,
  LOOM_SPIRV_OP_MEMORY_MODEL = 14,
  LOOM_SPIRV_OP_ENTRY_POINT = 15,
  LOOM_SPIRV_OP_EXECUTION_MODE = 16,
  LOOM_SPIRV_OP_CAPABILITY = 17,
  LOOM_SPIRV_OP_TYPE_VOID = 19,
  LOOM_SPIRV_OP_TYPE_INT = 21,
  LOOM_SPIRV_OP_TYPE_VECTOR = 23,
  LOOM_SPIRV_OP_TYPE_RUNTIME_ARRAY = 29,
  LOOM_SPIRV_OP_TYPE_STRUCT = 30,
  LOOM_SPIRV_OP_TYPE_POINTER = 32,
  LOOM_SPIRV_OP_TYPE_FUNCTION = 33,
  LOOM_SPIRV_OP_CONSTANT = 43,
  LOOM_SPIRV_OP_FUNCTION = 54,
  LOOM_SPIRV_OP_FUNCTION_END = 56,
  LOOM_SPIRV_OP_VARIABLE = 59,
  LOOM_SPIRV_OP_LOAD = 61,
  LOOM_SPIRV_OP_STORE = 62,
  LOOM_SPIRV_OP_ACCESS_CHAIN = 65,
  LOOM_SPIRV_OP_DECORATE = 71,
  LOOM_SPIRV_OP_MEMBER_DECORATE = 72,
  LOOM_SPIRV_OP_IADD = 128,
  LOOM_SPIRV_OP_LABEL = 248,
  LOOM_SPIRV_OP_RETURN = 253,
} loom_spirv_op_t;

typedef enum loom_spirv_execution_model_e {
  LOOM_SPIRV_EXECUTION_MODEL_GLCOMPUTE = 5,
} loom_spirv_execution_model_t;

typedef enum loom_spirv_execution_mode_e {
  LOOM_SPIRV_EXECUTION_MODE_LOCAL_SIZE = 17,
} loom_spirv_execution_mode_t;

typedef enum loom_spirv_function_control_e {
  LOOM_SPIRV_FUNCTION_CONTROL_NONE = 0,
} loom_spirv_function_control_t;

typedef enum loom_spirv_decoration_core_e {
  LOOM_SPIRV_DECORATION_BLOCK = 2,
  LOOM_SPIRV_DECORATION_ARRAY_STRIDE = 6,
  LOOM_SPIRV_DECORATION_BUILT_IN = 11,
  LOOM_SPIRV_DECORATION_BINDING = 33,
  LOOM_SPIRV_DECORATION_DESCRIPTOR_SET = 34,
  LOOM_SPIRV_DECORATION_OFFSET = 35,
} loom_spirv_decoration_core_t;

typedef enum loom_spirv_builtin_e {
  LOOM_SPIRV_BUILTIN_GLOBAL_INVOCATION_ID = 28,
} loom_spirv_builtin_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_BINARY_FORMAT_H_
