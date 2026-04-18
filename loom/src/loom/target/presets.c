// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/presets.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/ops.h"

static iree_status_t loom_target_preset_string(const loom_module_t* module,
                                               loom_string_id_t string_id,
                                               iree_string_view_t field_name,
                                               iree_string_view_t* out_string) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target preset field '%.*s' references an invalid "
                            "string id",
                            (int)field_name.size, field_name.data);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_target_preset_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t field_name, const loom_symbol_t** out_symbol) {
  *out_symbol = NULL;
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target preset field '%.*s' references an "
                            "unresolved symbol",
                            (int)field_name.size, field_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target preset field '%.*s' references an "
                            "unresolved symbol",
                            (int)field_name.size, field_name.data);
  }
  *out_symbol = symbol;
  return iree_ok_status();
}

static iree_status_t loom_target_preset_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol,
    iree_string_view_t field_name, iree_string_view_t* out_name) {
  if (symbol == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target preset field '%.*s' has no symbol",
                            (int)field_name.size, field_name.data);
  }
  return loom_target_preset_string(module, symbol->name_id, field_name,
                                   out_name);
}

static iree_status_t loom_target_preset_intern_string(
    loom_module_t* module, iree_string_view_t string,
    loom_string_id_t* out_string_id) {
  if (string.data == NULL) string = iree_string_view_empty();
  return loom_module_intern_string(module, string, out_string_id);
}

static iree_status_t loom_target_preset_create_symbol_ref(
    loom_module_t* module, iree_string_view_t name,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "target preset expansion would generate duplicate "
                            "symbol @%.*s",
                            (int)name.size, name.data);
  }
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
  *out_symbol_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  return iree_ok_status();
}

static iree_status_t loom_target_preset_build_sibling_name(
    iree_string_builder_t* builder, iree_string_view_t base_name,
    iree_string_view_t suffix) {
  iree_status_t status = iree_string_builder_append_string(builder, base_name);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(builder, suffix);
  }
  return status;
}

static iree_status_t loom_target_preset_require_symbol_available(
    const loom_module_t* module, iree_string_view_t name) {
  loom_string_id_t existing_name_id = loom_module_lookup_string(module, name);
  if (existing_name_id != LOOM_STRING_ID_INVALID &&
      loom_module_find_symbol(module, existing_name_id) !=
          LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "target preset expansion would generate duplicate "
                            "symbol @%.*s",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_preset_create_sibling_symbol_refs(
    loom_module_t* module, iree_string_view_t base_name,
    loom_symbol_ref_t* out_snapshot_ref, loom_symbol_ref_t* out_export_plan_ref,
    loom_symbol_ref_t* out_config_ref) {
  *out_snapshot_ref = loom_symbol_ref_null();
  *out_export_plan_ref = loom_symbol_ref_null();
  *out_config_ref = loom_symbol_ref_null();

  iree_string_builder_t snapshot_name;
  iree_string_builder_initialize(module->allocator, &snapshot_name);
  iree_string_builder_t export_plan_name;
  iree_string_builder_initialize(module->allocator, &export_plan_name);
  iree_string_builder_t config_name;
  iree_string_builder_initialize(module->allocator, &config_name);

  iree_status_t status = loom_target_preset_build_sibling_name(
      &snapshot_name, base_name, IREE_SV("__snapshot"));
  if (iree_status_is_ok(status)) {
    status = loom_target_preset_build_sibling_name(&export_plan_name, base_name,
                                                   IREE_SV("__export"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_preset_build_sibling_name(&config_name, base_name,
                                                   IREE_SV("__config"));
  }

  if (iree_status_is_ok(status)) {
    status = loom_target_preset_require_symbol_available(
        module, iree_string_builder_view(&snapshot_name));
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_preset_require_symbol_available(
        module, iree_string_builder_view(&export_plan_name));
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_preset_require_symbol_available(
        module, iree_string_builder_view(&config_name));
  }

  if (iree_status_is_ok(status)) {
    status = loom_target_preset_create_symbol_ref(
        module, iree_string_builder_view(&snapshot_name), out_snapshot_ref);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_preset_create_symbol_ref(
        module, iree_string_builder_view(&export_plan_name),
        out_export_plan_ref);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_preset_create_symbol_ref(
        module, iree_string_builder_view(&config_name), out_config_ref);
  }

  iree_string_builder_deinitialize(&config_name);
  iree_string_builder_deinitialize(&export_plan_name);
  iree_string_builder_deinitialize(&snapshot_name);
  return status;
}

static iree_status_t loom_target_preset_emit_snapshot(
    loom_builder_t* builder, const loom_target_snapshot_t* snapshot,
    loom_symbol_ref_t symbol_ref, loom_location_id_t location,
    loom_op_t** out_op) {
  loom_string_id_t target_triple_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_target_preset_intern_string(
      builder->module, snapshot->target_triple, &target_triple_id));
  loom_string_id_t data_layout_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_target_preset_intern_string(
      builder->module, snapshot->data_layout, &data_layout_id));
  loom_string_id_t target_cpu_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_target_preset_intern_string(
      builder->module, snapshot->target_cpu, &target_cpu_id));
  loom_string_id_t target_features_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_target_preset_intern_string(
      builder->module, snapshot->target_features, &target_features_id));
  return loom_target_snapshot_build(
      builder, symbol_ref, (uint8_t)snapshot->codegen_format, target_triple_id,
      data_layout_id, (uint8_t)snapshot->artifact_format, target_cpu_id,
      target_features_id, snapshot->default_pointer_bitwidth,
      snapshot->index_bitwidth, snapshot->offset_bitwidth,
      snapshot->memory_spaces.generic, snapshot->memory_spaces.global,
      snapshot->memory_spaces.workgroup, snapshot->memory_spaces.constant,
      snapshot->memory_spaces.private_memory, snapshot->memory_spaces.host,
      snapshot->memory_spaces.descriptor, location, out_op);
}

static iree_status_t loom_target_preset_emit_export(
    loom_builder_t* builder, const loom_target_export_plan_t* export_plan,
    loom_symbol_ref_t symbol_ref, loom_symbol_ref_t source_ref,
    loom_string_id_t export_symbol_id, loom_location_id_t location,
    loom_op_t** out_op) {
  return loom_target_export_build(
      builder, LOOM_TARGET_EXPORT_BUILD_FLAG_HAS_SOURCE, symbol_ref, source_ref,
      export_symbol_id, (uint8_t)export_plan->abi_kind,
      (uint8_t)export_plan->linkage, export_plan->hal_kernel.binding_alignment,
      export_plan->hal_kernel.required_workgroup_size.x,
      export_plan->hal_kernel.required_workgroup_size.y,
      export_plan->hal_kernel.required_workgroup_size.z,
      export_plan->hal_kernel.flat_workgroup_size_min,
      export_plan->hal_kernel.flat_workgroup_size_max,
      export_plan->hal_kernel.buffer_resource_flags, location, out_op);
}

static iree_status_t loom_target_preset_emit_config(
    loom_builder_t* builder, const loom_target_config_t* config,
    loom_symbol_ref_t symbol_ref, loom_location_id_t location,
    loom_op_t** out_op) {
  loom_string_id_t contract_set_key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_target_preset_intern_string(
      builder->module, config->contract_set_key, &contract_set_key_id));
  return loom_target_config_build(builder, symbol_ref, contract_set_key_id,
                                  (int64_t)config->contract_feature_bits,
                                  location, out_op);
}

static iree_status_t loom_target_expand_preset_op(
    loom_module_t* module, const loom_target_preset_registry_t* registry,
    loom_op_t* preset_op) {
  const loom_symbol_t* preset_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_target_preset_defined_symbol(
      module, loom_target_preset_symbol(preset_op), IREE_SV("symbol"),
      &preset_symbol));
  iree_string_view_t preset_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_target_preset_symbol_name(
      module, preset_symbol, IREE_SV("symbol"), &preset_name));

  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_target_preset_string(
      module, loom_target_preset_key(preset_op), IREE_SV("key"), &key));
  const loom_target_bundle_t* bundle = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_preset_registry_lookup_bundle(registry, key, &bundle));
  if (bundle->snapshot == NULL || bundle->export_plan == NULL ||
      bundle->config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target preset '%.*s' has incomplete bundle data",
                            (int)key.size, key.data);
  }

  const loom_symbol_t* source_symbol = NULL;
  const loom_symbol_ref_t source_ref = loom_target_preset_source(preset_op);
  IREE_RETURN_IF_ERROR(loom_target_preset_defined_symbol(
      module, source_ref, IREE_SV("source"), &source_symbol));
  loom_string_id_t export_symbol_id = source_symbol->name_id;

  loom_symbol_ref_t snapshot_ref = loom_symbol_ref_null();
  loom_symbol_ref_t export_plan_ref = loom_symbol_ref_null();
  loom_symbol_ref_t config_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_target_preset_create_sibling_symbol_refs(
      module, preset_name, &snapshot_ref, &export_plan_ref, &config_ref));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, preset_op->parent_block,
                          &builder);
  loom_builder_set_before(&builder, preset_op);

  loom_op_t* snapshot_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_preset_emit_snapshot(&builder, bundle->snapshot, snapshot_ref,
                                       preset_op->location, &snapshot_op));
  loom_op_t* export_op = NULL;
  IREE_RETURN_IF_ERROR(loom_target_preset_emit_export(
      &builder, bundle->export_plan, export_plan_ref, source_ref,
      export_symbol_id, preset_op->location, &export_op));
  loom_op_t* config_op = NULL;
  IREE_RETURN_IF_ERROR(loom_target_preset_emit_config(
      &builder, bundle->config, config_ref, preset_op->location, &config_op));

  const loom_symbol_ref_t preset_ref = loom_target_preset_symbol(preset_op);
  IREE_RETURN_IF_ERROR(loom_op_erase(module, preset_op));
  loom_builder_set_after(&builder, config_op);
  loom_op_t* bundle_op = NULL;
  return loom_target_bundle_build(&builder, preset_ref, snapshot_ref,
                                  export_plan_ref, config_ref,
                                  config_op->location, &bundle_op);
}

iree_status_t loom_target_expand_presets(
    loom_module_t* module, const loom_target_preset_registry_t* registry,
    iree_host_size_t* out_expanded_count) {
  if (out_expanded_count == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expanded count output is required");
  }
  *out_expanded_count = 0;
  if (module == NULL || registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and target preset registry are required");
  }

  loom_block_t* block = loom_module_block(module);
  loom_op_t* op = block->first_op;
  while (op != NULL) {
    loom_op_t* next_op = op->next_op;
    if (op->kind == LOOM_OP_TARGET_PRESET) {
      IREE_RETURN_IF_ERROR(loom_target_expand_preset_op(module, registry, op));
      ++*out_expanded_count;
    }
    op = next_op;
  }
  return iree_ok_status();
}
