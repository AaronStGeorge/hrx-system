// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_manifest_collect.h"

#include <string.h>

#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/target/facts.h"
#include "loom/target/types.h"

void loom_target_artifact_manifest_collect_options_initialize(
    loom_target_artifact_manifest_collect_options_t* out_options) {
  *out_options = (loom_target_artifact_manifest_collect_options_t){
      .mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE,
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

static iree_status_t loom_target_artifact_manifest_lookup_func_facts(
    loom_symbol_fact_table_t* fact_table, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_func_symbol_facts_t** out_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  const loom_func_symbol_facts_t* facts =
      loom_func_symbol_facts_cast(base_facts);
  if (!facts) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact plan symbol id %u does not have function facts",
        (unsigned)symbol_id);
  }
  *out_facts = facts;
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
    const loom_module_t* module, const loom_func_symbol_facts_t* facts,
    iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_interface_t* out_interface) {
  if (facts->argument_count == 0) return iree_ok_status();
  loom_target_artifact_manifest_parameter_t* parameters = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, facts->argument_count, sizeof(*parameters), (void**)&parameters));
  for (uint16_t i = 0; i < facts->argument_count; ++i) {
    parameters[i] = (loom_target_artifact_manifest_parameter_t){
        .name = loom_target_artifact_manifest_value_name(
            module, facts->argument_ids[i]),
        .kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_VALUE,
        .flags = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX,
        .index = i,
    };
  }
  out_interface->parameters = parameters;
  out_interface->parameter_detail_count = facts->argument_count;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_manifest_collect_functions(
    const loom_module_t* module, const loom_target_artifact_plan_t* plan,
    loom_symbol_fact_table_t* fact_table,
    const iree_string_view_t* target_name_refs,
    loom_target_artifact_manifest_mode_t mode, iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_function_t** out_functions) {
  loom_target_artifact_manifest_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, plan->entry_count, sizeof(*functions), (void**)&functions));
  const uint32_t subgroup_size =
      plan->artifact_facts->target->storage.snapshot.subgroup_size;
  for (uint16_t i = 0; i < plan->entry_count; ++i) {
    const loom_func_symbol_facts_t* facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_lookup_func_facts(
        fact_table, module, plan->entry_symbol_ids[i], &facts));
    functions[i].name = !iree_string_view_is_empty(facts->export_symbol)
                            ? facts->export_symbol
                            : facts->name;
    functions[i].source_name = facts->name;
    functions[i].target_names = target_name_refs;
    functions[i].target_name_count = 1;
    functions[i].interface.flags =
        LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_PARAMETER_COUNT;
    functions[i].interface.parameter_count = facts->argument_count;
    if (subgroup_size != 0) {
      functions[i].execution.flags =
          LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_SUBGROUP_SIZE;
      functions[i].execution.subgroup_size = subgroup_size;
    }
    if (loom_target_artifact_manifest_collect_mode_includes_details(mode)) {
      IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_parameters(
          module, facts, arena, &functions[i].interface));
    }
  }
  *out_functions = functions;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_manifest_mark_used_globals(
    const loom_target_artifact_plan_t* plan,
    const loom_symbol_dependency_table_t* dependency_table,
    uint8_t* global_marks, iree_host_size_t* out_global_count) {
  iree_host_size_t global_count = 0;
  for (uint16_t i = 0; i < plan->func_count; ++i) {
    const loom_symbol_id_t symbol_id = plan->func_symbol_ids[i];
    if (symbol_id >= dependency_table->symbol_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "artifact plan function symbol id %u is out of range for dependency "
          "table with %" PRIhsz " symbols",
          (unsigned)symbol_id, dependency_table->symbol_count);
    }
    loom_symbol_dependency_edge_id_t edge_id =
        dependency_table->symbols[symbol_id].first_outgoing_edge_id;
    while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
      if (edge_id >= dependency_table->edge_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "dependency edge id %u is out of range for table with %" PRIhsz
            " edges",
            (unsigned)edge_id, dependency_table->edge_count);
      }
      const loom_symbol_dependency_edge_t* edge =
          &dependency_table->edges[edge_id];
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
    const loom_module_t* module, const loom_target_artifact_plan_t* plan,
    const loom_symbol_dependency_table_t* dependency_table,
    const iree_string_view_t* target_name_refs, iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_global_t** out_globals,
    iree_host_size_t* out_global_count) {
  *out_globals = NULL;
  *out_global_count = 0;
  if (module->symbols.count == 0 || plan->func_count == 0) {
    return iree_ok_status();
  }

  uint8_t* global_marks = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, module->symbols.count, sizeof(*global_marks),
      (void**)&global_marks));
  iree_host_size_t global_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_mark_used_globals(
      plan, dependency_table, global_marks, &global_count));
  if (global_count == 0) return iree_ok_status();

  loom_target_artifact_manifest_global_t* globals = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, global_count, sizeof(*globals), (void**)&globals));
  iree_host_size_t global_index = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (!global_marks[i]) continue;
    const iree_string_view_t name = loom_target_artifact_manifest_module_string(
        module, module->symbols.entries[i].name_id);
    globals[global_index++] = (loom_target_artifact_manifest_global_t){
        .name = name,
        .source_name = name,
        .target_names = target_name_refs,
        .target_name_count = 1,
    };
  }
  *out_globals = globals;
  *out_global_count = global_count;
  return iree_ok_status();
}

iree_status_t loom_target_artifact_manifest_collect_from_plan(
    const loom_module_t* module, const loom_target_artifact_plan_t* plan,
    loom_symbol_fact_table_t* fact_table,
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
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest artifact plan is NULL");
  }
  if (!plan->artifact_facts || !plan->artifact_facts->target) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact manifest requires a resolved artifact plan");
  }
  if (!fact_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest symbol fact table is NULL");
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

  const iree_string_view_t target_name = plan->artifact_facts->target->name;
  iree_string_view_t* target_name_refs = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, 1, sizeof(*target_name_refs), (void**)&target_name_refs));
  target_name_refs[0] = target_name;

  loom_target_artifact_manifest_target_t* targets = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_allocate_array(
      arena, 1, sizeof(*targets), (void**)&targets));
  targets[0].name = target_name;

  loom_target_artifact_manifest_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_functions(
      module, plan, fact_table, target_name_refs, options->mode, arena,
      &functions));

  loom_target_artifact_manifest_global_t* globals = NULL;
  iree_host_size_t global_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_collect_globals(
      module, plan, dependency_table, target_name_refs, arena, &globals,
      &global_count));

  out_manifest->artifact.format =
      loom_target_artifact_manifest_public_format_name(
          plan->artifact_facts->format);
  out_manifest->artifact.name = plan->artifact_facts->name;
  if (iree_any_bit_set(
          options->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH)) {
    out_manifest->artifact.flags =
        LOOM_TARGET_ARTIFACT_MANIFEST_ARTIFACT_FLAG_BYTE_LENGTH;
    out_manifest->artifact.byte_length = options->artifact_byte_length;
  }
  out_manifest->targets = targets;
  out_manifest->target_count = 1;
  out_manifest->functions = functions;
  out_manifest->function_count = plan->entry_count;
  out_manifest->globals = globals;
  out_manifest->global_count = global_count;
  return iree_ok_status();
}
