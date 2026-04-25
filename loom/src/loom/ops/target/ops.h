// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables
// clang-format off

#ifndef LOOM_OPS_TARGET_OPS_H_
#define LOOM_OPS_TARGET_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_TARGET_ARTIFACT = LOOM_OP_KIND(LOOM_DIALECT_TARGET, 0),
  LOOM_OP_TARGET_PROFILE = LOOM_OP_KIND(LOOM_DIALECT_TARGET, 1),
  LOOM_OP_TARGET_COUNT_ = 2,
};

// Linkable or loadable artifact format produced for an artifact.
typedef enum loom_target_artifact_artifact_format_e {
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_UNKNOWN = 0,
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_ELF = 1,
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_COFF = 2,
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_MACHO = 3,
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_SPIRV_BINARY = 4,
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_VM_BYTECODE = 5,
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_WASM_BINARY = 6,
  LOOM_TARGET_ARTIFACT_ARTIFACT_FORMAT_COUNT_ = 7,
} loom_target_artifact_artifact_format_t;

// Runtime or linker packaging ABI used by a target artifact.
typedef enum loom_target_artifact_abi_e {
  LOOM_TARGET_ARTIFACT_ABI_UNKNOWN = 0,
  LOOM_TARGET_ARTIFACT_ABI_OBJECT_FILE = 1,
  LOOM_TARGET_ARTIFACT_ABI_HAL_EXECUTABLE = 2,
  LOOM_TARGET_ARTIFACT_ABI_VM_MODULE = 3,
  LOOM_TARGET_ARTIFACT_ABI_WASM_MODULE = 4,
  LOOM_TARGET_ARTIFACT_ABI_SPIRV_MODULE = 5,
  LOOM_TARGET_ARTIFACT_ABI_COUNT_ = 6,
} loom_target_artifact_abi_t;

// LOOM_OP_TARGET_ARTIFACT: Packaging or compile-unit record. Entry functions are derived from function export facts that reference this artifact; the artifact itself never lists functions.
// target.artifact @gfx11_kernels target(@gfx11) {artifact_format = elf, abi = hal_executable}
LOOM_DEFINE_ISA(loom_target_artifact_isa, LOOM_OP_TARGET_ARTIFACT)
LOOM_DEFINE_ATTR_SYMBOL(loom_target_artifact_symbol, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_target_artifact_target, 1)
LOOM_DEFINE_ATTR_ENUM(loom_target_artifact_artifact_format, 2)
LOOM_DEFINE_ATTR_ENUM(loom_target_artifact_abi, 3)
enum loom_target_artifact_build_flag_bits_e {
  LOOM_TARGET_ARTIFACT_BUILD_FLAG_HAS_ARTIFACT_FORMAT = 1u << 0,
  LOOM_TARGET_ARTIFACT_BUILD_FLAG_HAS_ABI = 1u << 1,
};
typedef uint32_t loom_target_artifact_build_flags_t;
iree_status_t loom_target_artifact_build(
    loom_builder_t* builder,
    loom_target_artifact_build_flags_t build_flags,
    loom_symbol_ref_t symbol,
    loom_symbol_ref_t target,
    loom_optional uint8_t artifact_format,
    loom_optional uint8_t abi,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_artifact_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TARGET_PROFILE: Compact reusable target environment profile. Providers own preset tables; the optional override dictionary is resolved once into dense symbol facts so target queries do not walk attr dictionaries.
// target.profile @vm preset("iree-vm")
LOOM_DEFINE_ISA(loom_target_profile_isa, LOOM_OP_TARGET_PROFILE)
LOOM_DEFINE_ATTR_SYMBOL(loom_target_profile_symbol, 0)
LOOM_DEFINE_ATTR_STRING(loom_target_profile_preset, 1)
LOOM_DEFINE_ATTR_DICT(loom_target_profile_overrides, 2)
iree_status_t loom_target_profile_build(
    loom_builder_t* builder,
    loom_symbol_ref_t symbol,
    loom_string_id_t preset,
    loom_optional loom_named_attr_slice_t overrides,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_profile_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the target dialect.
const loom_op_vtable_t* const* loom_target_dialect_vtables(
    iree_host_size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TARGET_OPS_H_
