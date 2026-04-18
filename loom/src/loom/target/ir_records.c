// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/ir_records.h"

#include <inttypes.h>

#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"

static iree_status_t loom_target_ir_require_op_kind(
    const loom_op_t* op, loom_op_kind_t expected_kind,
    iree_string_view_t expected_name) {
  if (op == NULL || op->kind != expected_kind) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "expected %.*s op",
                            (int)expected_name.size, expected_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_ir_string(const loom_module_t* module,
                                           loom_string_id_t string_id,
                                           iree_string_view_t field_name,
                                           iree_string_view_t* out_string) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target record field '%.*s' references an invalid "
                            "string id",
                            (int)field_name.size, field_name.data);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_target_ir_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target record field '%.*s' references an invalid "
                            "symbol",
                            (int)field_name.size, field_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return loom_target_ir_string(module, symbol->name_id, field_name, out_string);
}

static iree_status_t loom_target_ir_defined_symbol_from_ref(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t field_name, loom_op_kind_t expected_kind,
    iree_string_view_t expected_name, const loom_symbol_t** out_symbol) {
  *out_symbol = NULL;
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target record field '%.*s' references an "
                            "unresolved symbol",
                            (int)field_name.size, field_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target record field '%.*s' references an "
                            "unresolved symbol",
                            (int)field_name.size, field_name.data);
  }
  if (symbol->defining_op->kind != expected_kind) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_target_ir_string(module, symbol->name_id,
                                               field_name, &symbol_name));
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target record field '%.*s' references @%.*s, expected %.*s",
        (int)field_name.size, field_name.data, (int)symbol_name.size,
        symbol_name.data, (int)expected_name.size, expected_name.data);
  }
  *out_symbol = symbol;
  return iree_ok_status();
}

static iree_status_t loom_target_ir_u32(int64_t value,
                                        iree_string_view_t field_name,
                                        uint32_t* out_value) {
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target record field '%.*s' value %" PRId64
                            " is outside [0, 2^32)",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_target_ir_required_u32(int64_t value,
                                                 iree_string_view_t field_name,
                                                 uint32_t* out_value) {
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(value, field_name, out_value));
  if (*out_value == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target record field '%.*s' must be non-zero",
                            (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

// The IR materializer only rejects the explicit zero/unknown sentinel for enum
// fields that require a selected value. Backend support is a consumer contract,
// so new nonzero target enum ordinals must pass through this layer as data.
static iree_status_t loom_target_ir_require_selected_enum(
    uint8_t value, iree_string_view_t record_name,
    iree_string_view_t field_name) {
  if (value != 0) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s field '%.*s' must use a selected nonzero case",
                          (int)record_name.size, record_name.data,
                          (int)field_name.size, field_name.data);
}

iree_status_t loom_target_ir_snapshot_from_op(
    const loom_module_t* module, const loom_op_t* snapshot_op,
    loom_target_snapshot_t* out_snapshot) {
  if (module == NULL || out_snapshot == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and snapshot output are required");
  }
  *out_snapshot = (loom_target_snapshot_t){0};
  IREE_RETURN_IF_ERROR(loom_target_ir_require_op_kind(
      snapshot_op, LOOM_OP_TARGET_SNAPSHOT, IREE_SV("target.snapshot")));

  IREE_RETURN_IF_ERROR(loom_target_ir_symbol_name(
      module, loom_target_snapshot_symbol(snapshot_op), IREE_SV("symbol"),
      &out_snapshot->name));
  uint8_t codegen_format = loom_target_snapshot_codegen_format(snapshot_op);
  IREE_RETURN_IF_ERROR(loom_target_ir_require_selected_enum(
      codegen_format, IREE_SV("target snapshot"), IREE_SV("codegen_format")));
  out_snapshot->codegen_format = (loom_target_codegen_format_t)codegen_format;
  IREE_RETURN_IF_ERROR(loom_target_ir_string(
      module, loom_target_snapshot_target_triple(snapshot_op),
      IREE_SV("target_triple"), &out_snapshot->target_triple));
  IREE_RETURN_IF_ERROR(loom_target_ir_string(
      module, loom_target_snapshot_data_layout(snapshot_op),
      IREE_SV("data_layout"), &out_snapshot->data_layout));
  uint8_t artifact_format = loom_target_snapshot_artifact_format(snapshot_op);
  IREE_RETURN_IF_ERROR(loom_target_ir_require_selected_enum(
      artifact_format, IREE_SV("target snapshot"), IREE_SV("artifact_format")));
  out_snapshot->artifact_format =
      (loom_target_artifact_format_t)artifact_format;
  IREE_RETURN_IF_ERROR(loom_target_ir_string(
      module, loom_target_snapshot_target_cpu(snapshot_op),
      IREE_SV("target_cpu"), &out_snapshot->target_cpu));
  IREE_RETURN_IF_ERROR(loom_target_ir_string(
      module, loom_target_snapshot_target_features(snapshot_op),
      IREE_SV("target_features"), &out_snapshot->target_features));
  IREE_RETURN_IF_ERROR(loom_target_ir_required_u32(
      loom_target_snapshot_default_pointer_bitwidth(snapshot_op),
      IREE_SV("default_pointer_bitwidth"),
      &out_snapshot->default_pointer_bitwidth));
  IREE_RETURN_IF_ERROR(loom_target_ir_required_u32(
      loom_target_snapshot_index_bitwidth(snapshot_op),
      IREE_SV("index_bitwidth"), &out_snapshot->index_bitwidth));
  IREE_RETURN_IF_ERROR(loom_target_ir_required_u32(
      loom_target_snapshot_offset_bitwidth(snapshot_op),
      IREE_SV("offset_bitwidth"), &out_snapshot->offset_bitwidth));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_snapshot_memory_space_generic(snapshot_op),
      IREE_SV("memory_space_generic"), &out_snapshot->memory_spaces.generic));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_snapshot_memory_space_global(snapshot_op),
      IREE_SV("memory_space_global"), &out_snapshot->memory_spaces.global));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_snapshot_memory_space_workgroup(snapshot_op),
      IREE_SV("memory_space_workgroup"),
      &out_snapshot->memory_spaces.workgroup));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_snapshot_memory_space_constant(snapshot_op),
      IREE_SV("memory_space_constant"), &out_snapshot->memory_spaces.constant));
  IREE_RETURN_IF_ERROR(
      loom_target_ir_u32(loom_target_snapshot_memory_space_private(snapshot_op),
                         IREE_SV("memory_space_private"),
                         &out_snapshot->memory_spaces.private_memory));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_snapshot_memory_space_host(snapshot_op),
      IREE_SV("memory_space_host"), &out_snapshot->memory_spaces.host));
  return loom_target_ir_u32(
      loom_target_snapshot_memory_space_descriptor(snapshot_op),
      IREE_SV("memory_space_descriptor"),
      &out_snapshot->memory_spaces.descriptor);
}

iree_status_t loom_target_ir_export_plan_from_op(
    const loom_module_t* module, const loom_op_t* export_op,
    loom_target_export_plan_t* out_export_plan) {
  if (module == NULL || out_export_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and export-plan output are required");
  }
  *out_export_plan = (loom_target_export_plan_t){0};
  IREE_RETURN_IF_ERROR(loom_target_ir_require_op_kind(
      export_op, LOOM_OP_TARGET_EXPORT, IREE_SV("target.export")));

  IREE_RETURN_IF_ERROR(
      loom_target_ir_symbol_name(module, loom_target_export_symbol(export_op),
                                 IREE_SV("symbol"), &out_export_plan->name));
  const loom_attribute_t source_attr =
      loom_op_attrs(export_op)[loom_target_export_source_ATTR_INDEX];
  if (!loom_attr_is_absent(source_attr)) {
    IREE_RETURN_IF_ERROR(loom_target_ir_symbol_name(
        module, loom_target_export_source(export_op), IREE_SV("source"),
        &out_export_plan->source_symbol));
  }
  IREE_RETURN_IF_ERROR(loom_target_ir_string(
      module, loom_target_export_export_symbol(export_op),
      IREE_SV("export_symbol"), &out_export_plan->export_symbol));
  uint8_t abi_kind = loom_target_export_abi(export_op);
  IREE_RETURN_IF_ERROR(loom_target_ir_require_selected_enum(
      abi_kind, IREE_SV("target export plan"), IREE_SV("abi")));
  out_export_plan->abi_kind = (loom_target_abi_kind_t)abi_kind;
  out_export_plan->linkage =
      (loom_target_linkage_t)loom_target_export_linkage(export_op);
  IREE_RETURN_IF_ERROR(
      loom_target_ir_u32(loom_target_export_hal_binding_alignment(export_op),
                         IREE_SV("hal_binding_alignment"),
                         &out_export_plan->hal_kernel.binding_alignment));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_export_hal_workgroup_size_x(export_op),
      IREE_SV("hal_workgroup_size_x"),
      &out_export_plan->hal_kernel.required_workgroup_size.x));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_export_hal_workgroup_size_y(export_op),
      IREE_SV("hal_workgroup_size_y"),
      &out_export_plan->hal_kernel.required_workgroup_size.y));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_export_hal_workgroup_size_z(export_op),
      IREE_SV("hal_workgroup_size_z"),
      &out_export_plan->hal_kernel.required_workgroup_size.z));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_export_hal_flat_workgroup_size_min(export_op),
      IREE_SV("hal_flat_workgroup_size_min"),
      &out_export_plan->hal_kernel.flat_workgroup_size_min));
  IREE_RETURN_IF_ERROR(loom_target_ir_u32(
      loom_target_export_hal_flat_workgroup_size_max(export_op),
      IREE_SV("hal_flat_workgroup_size_max"),
      &out_export_plan->hal_kernel.flat_workgroup_size_max));
  return loom_target_ir_u32(
      loom_target_export_hal_buffer_resource_flags(export_op),
      IREE_SV("hal_buffer_resource_flags"),
      &out_export_plan->hal_kernel.buffer_resource_flags);
}

iree_status_t loom_target_ir_config_from_op(const loom_module_t* module,
                                            const loom_op_t* config_op,
                                            loom_target_config_t* out_config) {
  if (module == NULL || out_config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and config output are required");
  }
  *out_config = (loom_target_config_t){0};
  IREE_RETURN_IF_ERROR(loom_target_ir_require_op_kind(
      config_op, LOOM_OP_TARGET_CONFIG, IREE_SV("target.config")));

  IREE_RETURN_IF_ERROR(
      loom_target_ir_symbol_name(module, loom_target_config_symbol(config_op),
                                 IREE_SV("symbol"), &out_config->name));
  IREE_RETURN_IF_ERROR(loom_target_ir_string(
      module, loom_target_config_contract_set_key(config_op),
      IREE_SV("contract_set_key"), &out_config->contract_set_key));
  const int64_t feature_bits =
      loom_target_config_contract_feature_bits(config_op);
  if (feature_bits < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target config feature bits must be non-negative");
  }
  out_config->contract_feature_bits = (uint64_t)feature_bits;
  return iree_ok_status();
}

iree_status_t loom_target_ir_bundle_from_ops(
    const loom_module_t* module, const loom_op_t* bundle_op,
    const loom_op_t* snapshot_op, const loom_op_t* export_op,
    const loom_op_t* config_op, loom_target_ir_bundle_storage_t* out_storage) {
  if (module == NULL || out_storage == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and bundle storage output are required");
  }
  *out_storage = (loom_target_ir_bundle_storage_t){0};
  IREE_RETURN_IF_ERROR(loom_target_ir_require_op_kind(
      bundle_op, LOOM_OP_TARGET_BUNDLE, IREE_SV("target.bundle")));
  IREE_RETURN_IF_ERROR(loom_target_ir_snapshot_from_op(module, snapshot_op,
                                                       &out_storage->snapshot));
  IREE_RETURN_IF_ERROR(loom_target_ir_export_plan_from_op(
      module, export_op, &out_storage->export_plan));
  IREE_RETURN_IF_ERROR(
      loom_target_ir_config_from_op(module, config_op, &out_storage->config));

  iree_string_view_t bundle_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_target_ir_symbol_name(module, loom_target_bundle_symbol(bundle_op),
                                 IREE_SV("symbol"), &bundle_name));
  out_storage->bundle = (loom_target_bundle_t){
      .name = bundle_name,
      .snapshot = &out_storage->snapshot,
      .export_plan = &out_storage->export_plan,
      .config = &out_storage->config,
  };
  return iree_ok_status();
}

iree_status_t loom_target_ir_bundle_from_symbol_name(
    const loom_module_t* module, iree_string_view_t bundle_name,
    loom_target_ir_bundle_storage_t* out_storage) {
  if (module == NULL || out_storage == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and bundle storage output are required");
  }
  *out_storage = (loom_target_ir_bundle_storage_t){0};
  bundle_name = iree_string_view_trim(bundle_name);
  if (iree_string_view_is_empty(bundle_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle symbol name is required");
  }
  const loom_string_id_t bundle_name_id =
      loom_module_lookup_string(module, bundle_name);
  if (bundle_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target bundle @%.*s was not found",
                            (int)bundle_name.size, bundle_name.data);
  }
  const uint16_t bundle_symbol_id =
      loom_module_find_symbol(module, bundle_name_id);
  if (bundle_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target bundle @%.*s was not found",
                            (int)bundle_name.size, bundle_name.data);
  }

  const loom_symbol_t* bundle_symbol =
      &module->symbols.entries[bundle_symbol_id];
  if (bundle_symbol->definition == NULL || bundle_symbol->defining_op == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target bundle @%.*s has no definition",
                            (int)bundle_name.size, bundle_name.data);
  }
  const loom_op_t* bundle_op = bundle_symbol->defining_op;
  IREE_RETURN_IF_ERROR(loom_target_ir_require_op_kind(
      bundle_op, LOOM_OP_TARGET_BUNDLE, IREE_SV("target.bundle")));

  const loom_symbol_t* snapshot_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_target_ir_defined_symbol_from_ref(
      module, loom_target_bundle_snapshot(bundle_op), IREE_SV("snapshot"),
      LOOM_OP_TARGET_SNAPSHOT, IREE_SV("target.snapshot"), &snapshot_symbol));
  const loom_symbol_t* export_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_target_ir_defined_symbol_from_ref(
      module, loom_target_bundle_export_plan(bundle_op), IREE_SV("export_plan"),
      LOOM_OP_TARGET_EXPORT, IREE_SV("target.export"), &export_symbol));
  const loom_symbol_t* config_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_target_ir_defined_symbol_from_ref(
      module, loom_target_bundle_config(bundle_op), IREE_SV("config"),
      LOOM_OP_TARGET_CONFIG, IREE_SV("target.config"), &config_symbol));
  return loom_target_ir_bundle_from_ops(
      module, bundle_op, snapshot_symbol->defining_op,
      export_symbol->defining_op, config_symbol->defining_op, out_storage);
}
