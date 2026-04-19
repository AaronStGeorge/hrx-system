// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/target/emit/native/amdgpu/metadata.h"

#define LOOM_AMDGPU_KERNEL_ASSEMBLY_CODE_OBJECT_VERSION 5u

typedef struct loom_amdgpu_kernel_assembly_register_usage_t {
  // Highest SGPR index used by the function body plus one.
  uint32_t next_free_sgpr;
  // Highest VGPR index used by the function body plus one.
  uint32_t next_free_vgpr;
} loom_amdgpu_kernel_assembly_register_usage_t;

static bool loom_amdgpu_kernel_assembly_symbol_start_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '$';
}

static bool loom_amdgpu_kernel_assembly_symbol_continue_char(char c) {
  return loom_amdgpu_kernel_assembly_symbol_start_char(c) ||
         (c >= '0' && c <= '9') || c == '.';
}

static iree_status_t loom_amdgpu_kernel_assembly_validate_symbol(
    iree_string_view_t symbol) {
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel assembly symbol is required");
  }
  if (!loom_amdgpu_kernel_assembly_symbol_start_char(symbol.data[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel assembly symbol '%.*s' has an invalid first character",
        (int)symbol.size, symbol.data);
  }
  for (iree_host_size_t i = 1; i < symbol.size; ++i) {
    if (!loom_amdgpu_kernel_assembly_symbol_continue_char(symbol.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel assembly symbol '%.*s' contains an invalid character",
          (int)symbol.size, symbol.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_assembly_validate_target_id_component(
    iree_string_view_t value, iree_string_view_t field_name) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel assembly target-id field '%.*s' is required",
        (int)field_name.size, field_name.data);
  }
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const unsigned char c = (unsigned char)value.data[i];
    if (c <= ' ' || c == '"' || c == '\\') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel assembly target-id field '%.*s' contains an "
          "unsupported character",
          (int)field_name.size, field_name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_assembly_symbol_name(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_string_view_t* out_symbol) {
  IREE_ASSERT_ARGUMENT(out_symbol);
  *out_symbol = iree_string_view_empty();
  loom_symbol_ref_t symbol_ref = loom_low_func_def_callee(function_op);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel assembly function symbol is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel assembly function symbol has no "
                            "module string");
  }
  *out_symbol = module->strings.entries[symbol->name_id];
  return loom_amdgpu_kernel_assembly_validate_symbol(*out_symbol);
}

static iree_status_t loom_amdgpu_kernel_assembly_export_symbol(
    const loom_low_resolved_target_t* target, const loom_module_t* module,
    const loom_op_t* function_op, iree_string_view_t* out_symbol) {
  IREE_ASSERT_ARGUMENT(out_symbol);
  if (!iree_string_view_is_empty(
          target->bundle_storage.export_plan.export_symbol)) {
    *out_symbol = target->bundle_storage.export_plan.export_symbol;
    return loom_amdgpu_kernel_assembly_validate_symbol(*out_symbol);
  }
  return loom_amdgpu_kernel_assembly_symbol_name(module, function_op,
                                                 out_symbol);
}

static iree_status_t loom_amdgpu_kernel_assembly_validate_target(
    const loom_low_resolved_target_t* target) {
  const loom_target_snapshot_t* snapshot = &target->bundle_storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &target->bundle_storage.export_plan;
  if (target->descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel assembly target bundle is required");
  }
  if (snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel assembly requires low_native codegen snapshots");
  }
  if (!iree_string_view_equal(snapshot->target_triple,
                              IREE_SV("amdgcn-amd-amdhsa"))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel assembly requires amdgcn-amd-amdhsa, got '%.*s'",
        (int)snapshot->target_triple.size, snapshot->target_triple.data);
  }
  if (!iree_string_view_is_empty(snapshot->target_features)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU kernel assembly target-id feature strings are not encoded yet");
  }
  if (snapshot->artifact_format != LOOM_TARGET_ARTIFACT_FORMAT_ELF) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel assembly requires ELF artifacts");
  }
  if (export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU kernel assembly requires a HAL kernel ABI");
  }
  if (export_plan->linkage != LOOM_TARGET_LINKAGE_DEFAULT) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU kernel assembly currently requires default linkage");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_validate_target_id_component(
      snapshot->target_triple, IREE_SV("target_triple")));
  return loom_amdgpu_kernel_assembly_validate_target_id_component(
      snapshot->target_cpu, IREE_SV("target_cpu"));
}

static iree_status_t loom_amdgpu_kernel_assembly_validate_export_source(
    const loom_low_resolved_target_t* target, const loom_module_t* module,
    const loom_op_t* function_op) {
  iree_string_view_t source_symbol =
      target->bundle_storage.export_plan.source_symbol;
  if (iree_string_view_is_empty(source_symbol)) {
    return iree_ok_status();
  }
  iree_string_view_t function_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_symbol_name(
      module, function_op, &function_symbol));
  if (!iree_string_view_equal(source_symbol, function_symbol)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel assembly export plan source '%.*s' does not match low "
        "function '%.*s'",
        (int)source_symbol.size, source_symbol.data, (int)function_symbol.size,
        function_symbol.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_assembly_validate_function_shape(
    const loom_op_t* function_op) {
  if (!loom_low_func_def_isa(function_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel assembly requires low.func.def");
  }
  loom_region_t* body = loom_low_func_def_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel assembly function body is required");
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  if (entry_block->arg_count != 0 || function_op->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU kernel assembly currently requires ABI-lowered kernels with no "
        "low function arguments or results");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_assembly_update_high_water(
    uint32_t location_base, uint32_t location_count, uint32_t* inout_value) {
  IREE_ASSERT_ARGUMENT(inout_value);
  uint64_t next_free = (uint64_t)location_base + location_count;
  if (next_free > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel assembly register high-water mark overflows");
  }
  if ((uint32_t)next_free > *inout_value) {
    *inout_value = (uint32_t)next_free;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_assembly_collect_register_usage(
    const loom_low_allocation_sidecar_t* allocation,
    loom_amdgpu_kernel_assembly_register_usage_t* out_usage) {
  IREE_ASSERT_ARGUMENT(out_usage);
  *out_usage = (loom_amdgpu_kernel_assembly_register_usage_t){0};
  if (allocation->spill_plan_count != 0 || allocation->spill_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU kernel assembly requires materialized spill lowering before "
        "spilled allocations can be emitted");
  }
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU kernel assembly value %" PRIu32
                              " is not physically allocated",
                              assignment->value_id);
    }
    if (assignment->location_count == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU kernel assembly value %" PRIu32
                              " has an empty physical register range",
                              assignment->value_id);
    }
    iree_string_view_t register_class = iree_string_view_empty();
    if (assignment->value_class.register_class_id == LOOM_STRING_ID_INVALID ||
        assignment->value_class.register_class_id >=
            allocation->module->strings.count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU kernel assembly value %" PRIu32
                              " has no register-class string",
                              assignment->value_id);
    }
    register_class = allocation->module->strings
                         .entries[assignment->value_class.register_class_id];
    if (iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"))) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_update_high_water(
          assignment->location_base, assignment->location_count,
          &out_usage->next_free_sgpr));
      continue;
    }
    if (iree_string_view_equal(register_class, IREE_SV("amdgpu.vgpr"))) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_update_high_water(
          assignment->location_base, assignment->location_count,
          &out_usage->next_free_vgpr));
      continue;
    }
    if (iree_string_view_starts_with(register_class, IREE_SV("amdgpu."))) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU kernel assembly register class '%.*s' requires additional "
          "kernel descriptor metadata",
          (int)register_class.size, register_class.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_assembly_wavefront_size(
    iree_string_view_t target_cpu, uint32_t* out_wavefront_size) {
  IREE_ASSERT_ARGUMENT(out_wavefront_size);
  *out_wavefront_size = 0;
  if (iree_string_view_starts_with(target_cpu, IREE_SV("gfx9"))) {
    *out_wavefront_size = 64;
    return iree_ok_status();
  }
  if (iree_string_view_starts_with(target_cpu, IREE_SV("gfx11")) ||
      iree_string_view_starts_with(target_cpu, IREE_SV("gfx12"))) {
    *out_wavefront_size = 32;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU kernel assembly has no wavefront-size rule for target CPU '%.*s'",
      (int)target_cpu.size, target_cpu.data);
}

static iree_status_t loom_amdgpu_kernel_assembly_append_metadata(
    const loom_low_resolved_target_t* target, iree_string_view_t symbol,
    const loom_amdgpu_kernel_assembly_register_usage_t* register_usage,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n.rodata\n.p2align 6\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ".amdhsa_kernel %.*s\n", (int)symbol.size, symbol.data));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "  .amdhsa_group_segment_fixed_size 0\n"
      "  .amdhsa_private_segment_fixed_size 0\n"
      "  .amdhsa_kernarg_size 0\n"
      "  .amdhsa_user_sgpr_count 0\n"
      "  .amdhsa_system_sgpr_workgroup_id_x 0\n"
      "  .amdhsa_system_sgpr_workgroup_id_y 0\n"
      "  .amdhsa_system_sgpr_workgroup_id_z 0\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_next_free_vgpr %" PRIu32 "\n",
      register_usage->next_free_vgpr));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  .amdhsa_next_free_sgpr %" PRIu32 "\n",
      register_usage->next_free_sgpr));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ".end_amdhsa_kernel\n"));

  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(iree_allocator_system(), &target_id_builder);
  iree_string_builder_t descriptor_symbol_builder;
  iree_string_builder_initialize(iree_allocator_system(),
                                 &descriptor_symbol_builder);

  const loom_target_snapshot_t* snapshot = &target->bundle_storage.snapshot;
  const loom_target_hal_kernel_abi_t* hal_kernel =
      &target->bundle_storage.export_plan.hal_kernel;
  uint32_t wavefront_size = 0;
  iree_status_t status = iree_string_builder_append_format(
      &target_id_builder, "%.*s--%.*s", (int)snapshot->target_triple.size,
      snapshot->target_triple.data, (int)snapshot->target_cpu.size,
      snapshot->target_cpu.data);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_kernel_assembly_wavefront_size(snapshot->target_cpu,
                                                        &wavefront_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &descriptor_symbol_builder, "%.*s.kd", (int)symbol.size, symbol.data);
  }
  if (iree_status_is_ok(status)) {
    loom_amdgpu_metadata_kernel_t kernel = {
        .name = symbol,
        .descriptor_symbol =
            iree_string_builder_view(&descriptor_symbol_builder),
        .kernarg_segment_size = 0,
        .kernarg_segment_alignment = 8,
        .wavefront_size = wavefront_size,
        .group_segment_fixed_size = 0,
        .private_segment_fixed_size = 0,
        .sgpr_count = register_usage->next_free_sgpr,
        .vgpr_count = register_usage->next_free_vgpr,
        .max_flat_workgroup_size = hal_kernel->flat_workgroup_size_max,
        .required_workgroup_size = hal_kernel->required_workgroup_size,
        .has_required_workgroup_size = true,
        .arguments = NULL,
        .argument_count = 0,
    };
    loom_amdgpu_code_object_metadata_t metadata = {
        .target = iree_string_builder_view(&target_id_builder),
        .kernels = &kernel,
        .kernel_count = 1,
    };
    status = loom_amdgpu_metadata_append_assembly(&metadata, builder);
  }
  iree_string_builder_deinitialize(&descriptor_symbol_builder);
  iree_string_builder_deinitialize(&target_id_builder);
  return status;
}

static iree_status_t loom_amdgpu_kernel_assembly_emit(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_string_builder_t* builder) {
  if (builder == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel assembly output builder is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_sidecars(schedule, allocation));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_assembly_validate_target(&schedule->target));
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_validate_function_shape(
      schedule->function_op));
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_validate_export_source(
      &schedule->target, schedule->module, schedule->function_op));

  iree_string_view_t symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_export_symbol(
      &schedule->target, schedule->module, schedule->function_op, &symbol));

  loom_amdgpu_kernel_assembly_register_usage_t register_usage = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_assembly_collect_register_usage(
      allocation, &register_usage));

  const loom_target_snapshot_t* snapshot =
      &schedule->target.bundle_storage.snapshot;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ".text\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ".amdgcn_target \"%.*s--%.*s\"\n",
      (int)snapshot->target_triple.size, snapshot->target_triple.data,
      (int)snapshot->target_cpu.size, snapshot->target_cpu.data));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ".amdhsa_code_object_version %u\n\n",
      LOOM_AMDGPU_KERNEL_ASSEMBLY_CODE_OBJECT_VERSION));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ".protected %.*s\n"
      ".globl %.*s\n"
      ".p2align 8\n"
      ".type %.*s,@function\n"
      "%.*s:\n",
      (int)symbol.size, symbol.data, (int)symbol.size, symbol.data,
      (int)symbol.size, symbol.data, (int)symbol.size, symbol.data));
  if (wait_packets != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_assembly_fragment_with_wait_packets(
        schedule, allocation, wait_packets, builder));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_assembly_fragment(schedule, allocation, builder));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ".Lfunc_end0:\n"
      ".size %.*s, .Lfunc_end0-%.*s\n",
      (int)symbol.size, symbol.data, (int)symbol.size, symbol.data));
  return loom_amdgpu_kernel_assembly_append_metadata(&schedule->target, symbol,
                                                     &register_usage, builder);
}

iree_status_t loom_amdgpu_emit_kernel_assembly(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_builder_t* builder) {
  return loom_amdgpu_kernel_assembly_emit(schedule, allocation, NULL, builder);
}

iree_status_t loom_amdgpu_emit_kernel_assembly_with_wait_packets(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_string_builder_t* builder) {
  if (wait_packets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel assembly wait packets are required");
  }
  return loom_amdgpu_kernel_assembly_emit(schedule, allocation, wait_packets,
                                          builder);
}
