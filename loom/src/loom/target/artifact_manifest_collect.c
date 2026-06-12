// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_manifest_collect.h"

#include <stdint.h>
#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

void loom_target_artifact_manifest_collect_options_initialize(
    loom_target_artifact_manifest_collect_options_t* out_options) {
  *out_options = (loom_target_artifact_manifest_collect_options_t){
      .mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE,
      .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
  };
}

static bool loom_target_artifact_manifest_collect_mode_includes_details(
    loom_target_artifact_manifest_mode_t mode) {
  return mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS ||
         mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS;
}

static bool loom_target_artifact_manifest_collect_mode_is_valid(
    loom_target_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return true;
    default:
      return false;
  }
}

static iree_string_view_t loom_target_artifact_manifest_public_format_name(
    loom_target_artifact_format_t format) {
  switch (format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
      return IREE_SV("elf");
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
      return IREE_SV("coff");
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      return IREE_SV("macho");
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
      return IREE_SV("spirv-binary");
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
      return IREE_SV("vm-bytecode");
    case LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY:
      return IREE_SV("wasm-binary");
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_target_artifact_manifest_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_target_artifact_manifest_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_values) {
  *out_values = NULL;
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, element_size, out_values));
  memset(*out_values, 0, count * element_size);
  return iree_ok_status();
}

static iree_string_view_t loom_target_artifact_manifest_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  return loom_target_artifact_manifest_module_string(
      module, module->values.entries[value_id].name_id);
}

static iree_status_t loom_target_artifact_manifest_collect_parameters(
    const loom_module_t* module, const loom_value_id_t* argument_ids,
    uint16_t argument_count, iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_interface_t* out_interface) {
  if (argument_count == 0) return iree_ok_status();
  loom_target_artifact_manifest_parameter_t* parameters = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, argument_count, sizeof(*parameters), (void**)&parameters));
  for (uint16_t i = 0; i < argument_count; ++i) {
    parameters[i] = (loom_target_artifact_manifest_parameter_t){
        .name =
            loom_target_artifact_manifest_value_name(module, argument_ids[i]),
        .kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_VALUE,
        .flags = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX,
        .index = i,
    };
  }
  out_interface->parameters = parameters;
  out_interface->parameter_detail_count = argument_count;
  return iree_ok_status();
}

static iree_host_size_t loom_target_artifact_manifest_find_target(
    const loom_target_artifact_manifest_target_t* targets,
    iree_host_size_t target_count, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    if (iree_string_view_equal(targets[i].name, name)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_string_view_t loom_target_artifact_manifest_entry_target_name(
    const loom_target_entry_t* entry) {
  return entry->bundle_storage.bundle.name;
}

static bool loom_target_artifact_manifest_snapshot_has_workgroup_size_limit(
    const loom_target_snapshot_t* snapshot) {
  return snapshot->max_workgroup_size.x != 0 ||
         snapshot->max_workgroup_size.y != 0 ||
         snapshot->max_workgroup_size.z != 0;
}

static bool loom_target_artifact_manifest_snapshot_has_grid_size_limit(
    const loom_target_snapshot_t* snapshot) {
  return snapshot->max_grid_size.x != 0 || snapshot->max_grid_size.y != 0 ||
         snapshot->max_grid_size.z != 0;
}

static bool loom_target_artifact_manifest_snapshot_has_workgroup_count_limit(
    const loom_target_snapshot_t* snapshot) {
  return snapshot->max_workgroup_count.x != 0 ||
         snapshot->max_workgroup_count.y != 0 ||
         snapshot->max_workgroup_count.z != 0;
}

static bool loom_target_artifact_manifest_memory_space_is_target_specific(
    uint32_t value) {
  return value != 0 && value != UINT32_MAX;
}

static bool loom_target_artifact_manifest_snapshot_has_memory_spaces(
    const loom_target_snapshot_t* snapshot) {
  const loom_target_memory_space_map_t memory_spaces = snapshot->memory_spaces;
  return loom_target_artifact_manifest_memory_space_is_target_specific(
             memory_spaces.generic) ||
         loom_target_artifact_manifest_memory_space_is_target_specific(
             memory_spaces.global) ||
         loom_target_artifact_manifest_memory_space_is_target_specific(
             memory_spaces.workgroup) ||
         loom_target_artifact_manifest_memory_space_is_target_specific(
             memory_spaces.constant) ||
         loom_target_artifact_manifest_memory_space_is_target_specific(
             memory_spaces.private_memory) ||
         loom_target_artifact_manifest_memory_space_is_target_specific(
             memory_spaces.host) ||
         loom_target_artifact_manifest_memory_space_is_target_specific(
             memory_spaces.descriptor);
}

static void loom_target_artifact_manifest_collect_target_details(
    const loom_target_snapshot_t* snapshot,
    loom_target_artifact_manifest_target_t* target) {
  if (snapshot->default_pointer_bitwidth != 0) {
    target->flags |=
        LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_DEFAULT_POINTER_BITWIDTH;
    target->default_pointer_bitwidth = snapshot->default_pointer_bitwidth;
  }
  if (snapshot->index_bitwidth != 0) {
    target->flags |= LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_INDEX_BITWIDTH;
    target->index_bitwidth = snapshot->index_bitwidth;
  }
  if (snapshot->offset_bitwidth != 0) {
    target->flags |= LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_OFFSET_BITWIDTH;
    target->offset_bitwidth = snapshot->offset_bitwidth;
  }
  if (loom_target_artifact_manifest_snapshot_has_workgroup_size_limit(
          snapshot)) {
    target->flags |=
        LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_WORKGROUP_SIZE;
    target->max_workgroup_size = snapshot->max_workgroup_size;
  }
  if (snapshot->max_flat_workgroup_size != 0) {
    target->flags |=
        LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_FLAT_WORKGROUP_SIZE;
    target->max_flat_workgroup_size = snapshot->max_flat_workgroup_size;
  }
  if (snapshot->max_workgroup_storage_bytes != 0) {
    target->flags |=
        LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_WORKGROUP_STORAGE_BYTES;
    target->max_workgroup_storage_bytes = snapshot->max_workgroup_storage_bytes;
  }
  if (snapshot->subgroup_size != 0) {
    target->flags |= LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_SUBGROUP_SIZE;
    target->subgroup_size = snapshot->subgroup_size;
  }
  if (loom_target_artifact_manifest_snapshot_has_grid_size_limit(snapshot)) {
    target->flags |= LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_GRID_SIZE;
    target->max_grid_size = snapshot->max_grid_size;
  }
  if (snapshot->max_flat_grid_size != 0) {
    target->flags |=
        LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_FLAT_GRID_SIZE;
    target->max_flat_grid_size = snapshot->max_flat_grid_size;
  }
  if (loom_target_artifact_manifest_snapshot_has_workgroup_count_limit(
          snapshot)) {
    target->flags |=
        LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_WORKGROUP_COUNT;
    target->max_workgroup_count = snapshot->max_workgroup_count;
  }
  if (loom_target_artifact_manifest_snapshot_has_memory_spaces(snapshot)) {
    target->flags |= LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MEMORY_SPACES;
    target->memory_spaces = snapshot->memory_spaces;
  }
}

static iree_status_t loom_target_artifact_manifest_collect_targets(
    loom_target_entry_list_t entries, loom_target_artifact_manifest_mode_t mode,
    iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_target_t** out_targets,
    iree_host_size_t* out_target_count,
    iree_string_view_t** out_target_name_refs) {
  *out_targets = NULL;
  *out_target_count = 0;
  *out_target_name_refs = NULL;
  if (entries.count == 0) return iree_ok_status();

  loom_target_artifact_manifest_target_t* targets = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, entries.count, sizeof(*targets), (void**)&targets));

  iree_host_size_t target_count = 0;
  for (uint16_t i = 0; i < entries.count; ++i) {
    const iree_string_view_t target_name =
        loom_target_artifact_manifest_entry_target_name(&entries.values[i]);
    if (iree_string_view_is_empty(target_name)) {
      continue;
    }
    if (loom_target_artifact_manifest_find_target(
            targets, target_count, target_name) != IREE_HOST_SIZE_MAX) {
      continue;
    }
    targets[target_count++] = (loom_target_artifact_manifest_target_t){
        .name = target_name,
    };
    if (loom_target_artifact_manifest_collect_mode_includes_details(mode)) {
      loom_target_artifact_manifest_collect_target_details(
          &entries.values[i].bundle_storage.snapshot,
          &targets[target_count - 1]);
    }
  }

  iree_string_view_t* target_name_refs = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, target_count, sizeof(*target_name_refs),
      (void**)&target_name_refs));
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    target_name_refs[i] = targets[i].name;
  }

  *out_targets = targets;
  *out_target_count = target_count;
  *out_target_name_refs = target_name_refs;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_manifest_collect_functions(
    const loom_module_t* module, loom_target_entry_list_t entries,
    loom_target_artifact_manifest_mode_t mode, iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_function_t** out_functions) {
  loom_target_artifact_manifest_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, entries.count, sizeof(*functions), (void**)&functions));

  iree_string_view_t* function_target_name_refs = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, entries.count, sizeof(*function_target_name_refs),
      (void**)&function_target_name_refs));

  for (uint16_t i = 0; i < entries.count; ++i) {
    const loom_target_entry_t* entry = &entries.values[i];
    const loom_target_export_plan_t* export_plan =
        &entry->bundle_storage.export_plan;
    functions[i].name = !iree_string_view_is_empty(export_plan->export_symbol)
                            ? export_plan->export_symbol
                            : entry->func_name;
    functions[i].source_name = entry->func_name;
    function_target_name_refs[i] =
        loom_target_artifact_manifest_entry_target_name(entry);
    if (!iree_string_view_is_empty(function_target_name_refs[i])) {
      functions[i].target_names = &function_target_name_refs[i];
      functions[i].target_name_count = 1;
    }
    uint16_t argument_count = 0;
    const loom_value_id_t* argument_ids =
        loom_func_like_arg_ids(entry->func, &argument_count);
    functions[i].interface.flags =
        LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_PARAMETER_COUNT;
    functions[i].interface.parameter_count = argument_count;
    if (entry->bundle_storage.snapshot.subgroup_size != 0) {
      functions[i].execution.flags =
          LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_SUBGROUP_SIZE;
      functions[i].execution.subgroup_size =
          entry->bundle_storage.snapshot.subgroup_size;
    }
    if (loom_target_artifact_manifest_collect_mode_includes_details(mode)) {
      IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_parameters(
          module, argument_ids, argument_count, arena,
          &functions[i].interface));
    }
  }
  *out_functions = functions;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_manifest_dependency_edge(
    const loom_symbol_dependency_table_t* dependency_table,
    loom_symbol_dependency_edge_id_t edge_id,
    const loom_symbol_dependency_edge_t** out_edge) {
  if (edge_id >= dependency_table->edge_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "dependency edge id %u is out of range for table with %" PRIhsz
        " edges",
        (unsigned)edge_id, dependency_table->edge_count);
  }
  *out_edge = &dependency_table->edges[edge_id];
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_manifest_mark_function_closure(
    loom_target_entry_list_t entries,
    const loom_symbol_dependency_table_t* dependency_table,
    iree_arena_allocator_t* arena, uint8_t* function_marks) {
  if (dependency_table->symbol_count == 0) return iree_ok_status();
  loom_symbol_id_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, dependency_table->symbol_count, sizeof(*stack), (void**)&stack));
  iree_host_size_t stack_count = 0;

  for (uint16_t i = 0; i < entries.count; ++i) {
    const loom_symbol_id_t symbol_id = entries.values[i].func_ref.symbol_id;
    if (symbol_id >= dependency_table->symbol_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "artifact manifest entry symbol id %u is out of range for dependency "
          "table with %" PRIhsz " symbols",
          (unsigned)symbol_id, dependency_table->symbol_count);
    }
    if (!function_marks[symbol_id]) {
      function_marks[symbol_id] = 1;
      stack[stack_count++] = symbol_id;
    }
  }

  while (stack_count > 0) {
    const loom_symbol_id_t symbol_id = stack[--stack_count];
    loom_symbol_dependency_edge_id_t edge_id =
        dependency_table->symbols[symbol_id].first_outgoing_edge_id;
    while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
      const loom_symbol_dependency_edge_t* edge = NULL;
      IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_dependency_edge(
          dependency_table, edge_id, &edge));
      if (edge->kind == LOOM_SYMBOL_DEPENDENCY_EDGE_CALL) {
        if (edge->target_symbol_id >= dependency_table->symbol_count) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "dependency call target symbol id %u is out of range for table "
              "with %" PRIhsz " symbols",
              (unsigned)edge->target_symbol_id, dependency_table->symbol_count);
        }
        if (!function_marks[edge->target_symbol_id]) {
          function_marks[edge->target_symbol_id] = 1;
          stack[stack_count++] = edge->target_symbol_id;
        }
      }
      edge_id = edge->next_outgoing_edge_id;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_manifest_mark_used_globals(
    loom_target_entry_list_t entries,
    const loom_symbol_dependency_table_t* dependency_table,
    iree_arena_allocator_t* arena, uint8_t* global_marks,
    iree_host_size_t* out_global_count) {
  *out_global_count = 0;
  if (dependency_table->symbol_count == 0 || entries.count == 0) {
    return iree_ok_status();
  }

  uint8_t* function_marks = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, dependency_table->symbol_count, sizeof(*function_marks),
      (void**)&function_marks));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_mark_function_closure(
      entries, dependency_table, arena, function_marks));

  iree_host_size_t global_count = 0;
  for (iree_host_size_t i = 0; i < dependency_table->symbol_count; ++i) {
    if (!function_marks[i]) continue;
    loom_symbol_dependency_edge_id_t edge_id =
        dependency_table->symbols[i].first_outgoing_edge_id;
    while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
      const loom_symbol_dependency_edge_t* edge = NULL;
      IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_dependency_edge(
          dependency_table, edge_id, &edge));
      if (edge->kind == LOOM_SYMBOL_DEPENDENCY_EDGE_GLOBAL_ACCESS) {
        if (edge->target_symbol_id >= dependency_table->symbol_count) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "dependency global target symbol id %u is out of range for "
              "table with %" PRIhsz " symbols",
              (unsigned)edge->target_symbol_id, dependency_table->symbol_count);
        }
        if (!global_marks[edge->target_symbol_id]) {
          global_marks[edge->target_symbol_id] = 1;
          ++global_count;
        }
      }
      edge_id = edge->next_outgoing_edge_id;
    }
  }

  *out_global_count = global_count;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_manifest_collect_globals(
    const loom_module_t* module, loom_target_entry_list_t entries,
    const loom_symbol_dependency_table_t* dependency_table,
    const iree_string_view_t* target_name_refs,
    iree_host_size_t target_name_count, iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_global_t** out_globals,
    iree_host_size_t* out_global_count) {
  *out_globals = NULL;
  *out_global_count = 0;
  if (module->symbols.count == 0 || entries.count == 0) {
    return iree_ok_status();
  }

  uint8_t* global_marks = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, module->symbols.count, sizeof(*global_marks),
      (void**)&global_marks));
  iree_host_size_t global_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_mark_used_globals(
      entries, dependency_table, arena, global_marks, &global_count));
  if (global_count == 0) return iree_ok_status();

  loom_target_artifact_manifest_global_t* globals = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, global_count, sizeof(*globals), (void**)&globals));
  iree_host_size_t global_index = 0;
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(module_block, op) {
    loom_symbol_ref_t symbol_ref = loom_symbol_ref_null();
    if (!loom_op_defining_symbol_ref(module, op, &symbol_ref)) {
      continue;
    }
    if (!global_marks[symbol_ref.symbol_id]) continue;
    const iree_string_view_t name = loom_target_artifact_manifest_module_string(
        module, module->symbols.entries[symbol_ref.symbol_id].name_id);
    globals[global_index++] = (loom_target_artifact_manifest_global_t){
        .name = name,
        .source_name = name,
        .target_names = target_name_refs,
        .target_name_count = target_name_count,
    };
  }
  if (global_index != global_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "artifact manifest marked %" PRIhsz
                            " globals but found %" PRIhsz
                            " top-level definitions",
                            global_count, global_index);
  }
  *out_globals = globals;
  *out_global_count = global_count;
  return iree_ok_status();
}

iree_status_t loom_target_artifact_manifest_collect_from_entries(
    const loom_module_t* module, loom_target_entry_list_t entries,
    const loom_symbol_dependency_table_t* dependency_table,
    const loom_target_artifact_manifest_collect_options_t* options,
    iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_t* out_manifest) {
  if (!out_manifest) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest output is NULL");
  }
  *out_manifest = (loom_target_artifact_manifest_t){0};
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest collection options are NULL");
  }
  if (!loom_target_artifact_manifest_collect_mode_is_valid(options->mode)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported artifact manifest mode %d",
                            (int)options->mode);
  }
  if (options->mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_ok_status();
  }
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest module is NULL");
  }
  if (entries.count != 0 && entries.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest entry list is missing values");
  }
  if (!dependency_table) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "artifact manifest symbol dependency table is NULL");
  }
  if (dependency_table->module != module) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact manifest dependency table belongs to another module");
  }
  if (dependency_table->symbol_count != module->symbols.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "artifact manifest dependency table has %" PRIhsz
                            " symbols but module has %" PRIhsz " symbols",
                            dependency_table->symbol_count,
                            module->symbols.count);
  }
  if (!arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest arena is NULL");
  }

  loom_target_artifact_manifest_target_t* targets = NULL;
  iree_host_size_t target_count = 0;
  iree_string_view_t* target_name_refs = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_targets(
      entries, options->mode, arena, &targets, &target_count,
      &target_name_refs));

  loom_target_artifact_manifest_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_functions(
      module, entries, options->mode, arena, &functions));

  loom_target_artifact_manifest_global_t* globals = NULL;
  iree_host_size_t global_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_globals(
      module, entries, dependency_table, target_name_refs, target_count, arena,
      &globals, &global_count));

  loom_target_artifact_format_t artifact_format = options->artifact_format;
  if (artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN &&
      entries.count > 0) {
    artifact_format = entries.values[0].bundle_storage.snapshot.artifact_format;
  }
  out_manifest->artifact.format =
      loom_target_artifact_manifest_public_format_name(artifact_format);
  out_manifest->artifact.name = options->artifact_name;
  if (iree_any_bit_set(
          options->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH)) {
    out_manifest->artifact.flags =
        LOOM_TARGET_ARTIFACT_MANIFEST_ARTIFACT_FLAG_BYTE_LENGTH;
    out_manifest->artifact.byte_length = options->artifact_byte_length;
  }
  out_manifest->targets = targets;
  out_manifest->target_count = target_count;
  out_manifest->functions = functions;
  out_manifest->function_count = entries.count;
  out_manifest->globals = globals;
  out_manifest->global_count = global_count;
  return iree_ok_status();
}

iree_status_t loom_target_artifact_manifest_collect_json_from_entries(
    const loom_module_t* module, loom_target_entry_list_t entries,
    const loom_target_artifact_manifest_collect_options_t* options,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    loom_target_artifact_manifest_json_t* out_json) {
  if (out_json == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest JSON output is NULL");
  }
  *out_json = (loom_target_artifact_manifest_json_t){0};
  if (options == NULL ||
      options->mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_ok_status();
  }
  if (!loom_target_artifact_manifest_collect_mode_is_valid(options->mode)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported artifact manifest mode %d",
                            (int)options->mode);
  }
  if (module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest module is NULL");
  }
  if (arena == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest arena is NULL");
  }

  loom_symbol_dependency_table_t dependency_table = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_dependency_table_build(module, arena, &dependency_table));

  loom_target_artifact_manifest_t manifest = {0};
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_from_entries(
      module, entries, &dependency_table, options, arena, &manifest));

  loom_target_artifact_manifest_format_options_t format_options;
  loom_target_artifact_manifest_format_options_initialize(&format_options);
  format_options.mode = options->mode;
  return loom_target_artifact_manifest_format_json_bytes(
      &manifest, &format_options, allocator, out_json);
}
