// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.c_tables.
// clang-format off

#include "loom/target/emit/ireevm/ops/ops.h"

#include <stddef.h>

#include "loom/target/types.h"
#include "loom/error/error_defs.h"

#define _BSTRING(length, value) LOOM_BSTRING_REF(length, value)
#define _OP_NAME(length, namespace_length, value) \
  LOOM_OP_NAME_REF(length, namespace_length, value)

extern const loom_symbol_fact_domain_t loom_target_symbol_fact_domain;

extern const loom_target_bundle_table_t loom_ireevm_target_bundles;

static const loom_format_element_t loom_ireevm_target_format[] = {
    {LOOM_FORMAT_KIND_TEMPLATE_PARAM, 1, 0},
    {LOOM_FORMAT_KIND_SYMBOL_REF, 0, 0},
    {LOOM_FORMAT_KIND_ATTR_DICT, 0, LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS},
};
static const loom_bstring_t loom_ireevm_target_kind_names[] = {
    NULL,
    _BSTRING(4, "core"),
};
static const loom_bstring_t loom_target_codegen_format_names[] = {
    _BSTRING(7, "unknown"),
    _BSTRING(6, "llvmir"),
    _BSTRING(5, "spirv"),
    _BSTRING(2, "vm"),
    _BSTRING(10, "low_native"),
    _BSTRING(4, "wasm"),
};
static const loom_bstring_t loom_target_artifact_format_names[] = {
    _BSTRING(7, "unknown"),
    _BSTRING(3, "elf"),
    _BSTRING(4, "coff"),
    _BSTRING(5, "macho"),
    _BSTRING(12, "spirv_binary"),
    _BSTRING(11, "vm_bytecode"),
    _BSTRING(11, "wasm_binary"),
};
static const loom_bstring_t loom_target_abi_kind_names[] = {
    _BSTRING(7, "unknown"),
    _BSTRING(15, "object_function"),
    _BSTRING(10, "hal_kernel"),
    _BSTRING(18, "vm_module_function"),
    _BSTRING(18, "shader_entry_point"),
    _BSTRING(13, "wasm_function"),
};
static const loom_bstring_t loom_target_linkage_names[] = {
    _BSTRING(7, "default"),
    _BSTRING(9, "dso_local"),
};
static const loom_attr_descriptor_t loom_ireevm_target_attr_desc[] = {
    {_BSTRING(6, "symbol"), LOOM_ATTR_SYMBOL, 0, 0, NULL, NULL},
    {_BSTRING(4, "kind"), LOOM_ATTR_ENUM, 0, IREE_ARRAYSIZE(loom_ireevm_target_kind_names), loom_ireevm_target_kind_names, NULL},
    {_BSTRING(14, "codegen_format"), LOOM_ATTR_ENUM, LOOM_ATTR_OPTIONAL | LOOM_ATTR_OPEN_ENUM, IREE_ARRAYSIZE(loom_target_codegen_format_names), loom_target_codegen_format_names, NULL},
    {_BSTRING(15, "artifact_format"), LOOM_ATTR_ENUM, LOOM_ATTR_OPTIONAL | LOOM_ATTR_OPEN_ENUM, IREE_ARRAYSIZE(loom_target_artifact_format_names), loom_target_artifact_format_names, NULL},
    {_BSTRING(24, "default_pointer_bitwidth"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(14, "index_bitwidth"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(15, "offset_bitwidth"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(20, "max_workgroup_size_x"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(20, "max_workgroup_size_y"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(20, "max_workgroup_size_z"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(23, "max_flat_workgroup_size"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(13, "subgroup_size"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(21, "max_workgroup_count_x"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(21, "max_workgroup_count_y"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(21, "max_workgroup_count_z"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(20, "memory_space_generic"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(19, "memory_space_global"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(22, "memory_space_workgroup"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(21, "memory_space_constant"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(20, "memory_space_private"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(17, "memory_space_host"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(23, "memory_space_descriptor"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(3, "abi"), LOOM_ATTR_ENUM, LOOM_ATTR_OPTIONAL | LOOM_ATTR_OPEN_ENUM, IREE_ARRAYSIZE(loom_target_abi_kind_names), loom_target_abi_kind_names, NULL},
    {_BSTRING(13, "export_symbol"), LOOM_ATTR_STRING, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(7, "linkage"), LOOM_ATTR_ENUM, LOOM_ATTR_OPTIONAL | LOOM_ATTR_OPEN_ENUM, IREE_ARRAYSIZE(loom_target_linkage_names), loom_target_linkage_names, NULL},
    {_BSTRING(21, "hal_binding_alignment"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(25, "hal_buffer_resource_flags"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(16, "contract_set_key"), LOOM_ATTR_STRING, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
    {_BSTRING(21, "contract_feature_bits"), LOOM_ATTR_I64, LOOM_ATTR_OPTIONAL, 0, NULL, NULL},
};
static const loom_target_projection_t loom_ireevm_target_target_projections[] = {
    {offsetof(loom_target_bundle_storage_t, snapshot.codegen_format), 2, LOOM_TARGET_PROJECTION_VALUE_ENUM_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.artifact_format), 3, LOOM_TARGET_PROJECTION_VALUE_ENUM_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.default_pointer_bitwidth), 4, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.index_bitwidth), 5, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.offset_bitwidth), 6, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.max_workgroup_size.x), 7, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.max_workgroup_size.y), 8, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.max_workgroup_size.z), 9, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.max_flat_workgroup_size), 10, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.subgroup_size), 11, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.max_workgroup_count.x), 12, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.max_workgroup_count.y), 13, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.max_workgroup_count.z), 14, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.memory_spaces.generic), 15, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.memory_spaces.global), 16, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.memory_spaces.workgroup), 17, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.memory_spaces.constant), 18, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.memory_spaces.private_memory), 19, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.memory_spaces.host), 20, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, snapshot.memory_spaces.descriptor), 21, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, export_plan.abi_kind), 22, LOOM_TARGET_PROJECTION_VALUE_ENUM_U32},
    {offsetof(loom_target_bundle_storage_t, export_plan.export_symbol), 23, LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW},
    {offsetof(loom_target_bundle_storage_t, export_plan.linkage), 24, LOOM_TARGET_PROJECTION_VALUE_ENUM_U32},
    {offsetof(loom_target_bundle_storage_t, export_plan.hal_kernel.binding_alignment), 25, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, export_plan.hal_kernel.buffer_resource_flags), 26, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32},
    {offsetof(loom_target_bundle_storage_t, config.contract_set_key), 27, LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW},
    {offsetof(loom_target_bundle_storage_t, config.contract_feature_bits), 28, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64},
};
static const loom_target_like_descriptor_t loom_ireevm_target_target_like_descriptor = {
    .bundle_table = &loom_ireevm_target_bundles,
    .projections = loom_ireevm_target_target_projections,
    .projection_count = IREE_ARRAYSIZE(loom_ireevm_target_target_projections),
};

static const loom_target_like_vtable_t loom_ireevm_target_target_like = {
    .symbol_attr_index = 0,
    .selector_attr_index = 1,
    .extension_attrs_attr_index = 255,
    .descriptor = &loom_ireevm_target_target_like_descriptor,
};

static const loom_symbol_definition_descriptor_t loom_ireevm_target_symbol_def = {
    .name = _BSTRING(6, "target"),
    .interfaces = LOOM_SYMBOL_INTERFACE_TARGET | LOOM_SYMBOL_INTERFACE_RECORD,
    .bytecode_kind = LOOM_SYMBOL_RECORD,
    .fact_domain = &loom_target_symbol_fact_domain,
};
static const loom_op_vtable_t loom_ireevm_target_vtable = {
    .traits = LOOM_TRAIT_SYMBOL_DEFINE | LOOM_TRAIT_PURE,
    .symbol_kind = LOOM_SYMBOL_RECORD,
    .verify = loom_target_record_verify,
    .name = _OP_NAME(13, 6, "ireevm.target"),
    .attr_descriptors = loom_ireevm_target_attr_desc,
    .attribute_count = IREE_ARRAYSIZE(loom_ireevm_target_attr_desc),
    .format_elements = loom_ireevm_target_format,
    .format_element_count = IREE_ARRAYSIZE(loom_ireevm_target_format),
    .target_like = &loom_ireevm_target_target_like,
    .symbol_def = &loom_ireevm_target_symbol_def,
};

#undef _OP_NAME
#undef _BSTRING

static const loom_op_vtable_t* const loom_ireevm_vtable_array[] = {
    &loom_ireevm_target_vtable,
};

const loom_op_vtable_t* const* loom_ireevm_dialect_vtables(
    iree_host_size_t* out_count) {
  *out_count = IREE_ARRAYSIZE(loom_ireevm_vtable_array);
  return loom_ireevm_vtable_array;
}

static const loom_op_semantics_t loom_ireevm_semantics_array[] = {
    {
        .phase = LOOM_OP_PHASE_MODULE_METADATA,
    },
};

const loom_op_semantics_t* loom_ireevm_dialect_op_semantics(
    iree_host_size_t* out_count) {
  *out_count = IREE_ARRAYSIZE(loom_ireevm_semantics_array);
  return loom_ireevm_semantics_array;
}

loom_op_semantics_t loom_ireevm_op_semantics(
    loom_op_kind_t kind) {
  if (loom_op_dialect_id(kind) != LOOM_DIALECT_IREEVM) {
    return loom_op_semantics_empty();
  }
  uint8_t op_index = loom_op_dialect_index(kind);
  if (op_index >= IREE_ARRAYSIZE(loom_ireevm_semantics_array)) {
    return loom_op_semantics_empty();
  }
  return loom_ireevm_semantics_array[op_index];
}
