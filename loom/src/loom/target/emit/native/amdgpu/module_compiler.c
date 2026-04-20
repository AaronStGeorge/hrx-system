// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/module_compiler.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/hal_resource_materialization.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_plan.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/module_compiler.h"

enum {
  LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS = 20,
};

static bool loom_amdgpu_module_compile_bundle_is_compatible(
    void* user_data, const loom_target_bundle_t* bundle) {
  (void)user_data;
  return bundle && bundle->snapshot && bundle->export_plan &&
         bundle->snapshot->codegen_format ==
             LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF &&
         iree_string_view_equal(bundle->snapshot->target_triple,
                                IREE_SV("amdgcn-amd-amdhsa")) &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

static iree_status_t loom_amdgpu_module_compile_append_target_id(
    const loom_target_snapshot_t* snapshot, iree_string_builder_t* builder) {
  if (!snapshot || iree_string_view_is_empty(snapshot->target_triple) ||
      iree_string_view_is_empty(snapshot->target_cpu)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target snapshot requires triple and CPU");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, snapshot->target_triple));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("--")));
  return iree_string_builder_append_string(builder, snapshot->target_cpu);
}

static iree_status_t loom_amdgpu_module_compile_append_descriptor_symbol(
    const loom_target_bundle_t* bundle, iree_string_builder_t* builder) {
  iree_string_view_t symbol = iree_string_view_empty();
  if (bundle && bundle->export_plan &&
      !iree_string_view_is_empty(bundle->export_plan->export_symbol)) {
    symbol = bundle->export_plan->export_symbol;
  } else if (bundle && bundle->export_plan) {
    symbol = bundle->export_plan->source_symbol;
  }
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL executable export symbol is required");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, symbol));
  return iree_string_builder_append_string(builder, IREE_SV(".kd"));
}

static iree_status_t loom_amdgpu_module_compile_apply_target_cpu(
    loom_module_t* module, loom_target_module_compile_target_t* target,
    iree_string_view_t target_cpu) {
  if (iree_string_view_is_empty(target_cpu)) {
    return iree_ok_status();
  }
  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(target_cpu, &processor));
  if (iree_string_view_is_empty(processor->low_preset_key) ||
      !iree_string_view_equal(processor->descriptor_set_key,
                              target->bundle_storage.config.contract_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU target CPU '%.*s' is not compatible with descriptor set '%.*s'",
        (int)target_cpu.size, target_cpu.data,
        (int)target->bundle_storage.config.contract_set_key.size,
        target->bundle_storage.config.contract_set_key.data);
  }

  if (!loom_symbol_ref_is_valid(target->target_ref) ||
      target->target_ref.module_id != 0 ||
      target->target_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected AMDGPU target symbol is invalid");
  }
  loom_op_t* bundle_op =
      module->symbols.entries[target->target_ref.symbol_id].defining_op;
  if (!loom_target_bundle_isa(bundle_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected AMDGPU target is not a target.bundle");
  }
  const loom_symbol_ref_t snapshot_ref = loom_target_bundle_snapshot(bundle_op);
  if (!loom_symbol_ref_is_valid(snapshot_ref) || snapshot_ref.module_id != 0 ||
      snapshot_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected AMDGPU target snapshot is invalid");
  }
  loom_op_t* snapshot_op =
      module->symbols.entries[snapshot_ref.symbol_id].defining_op;
  if (!loom_target_snapshot_isa(snapshot_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected AMDGPU target snapshot is not a "
                            "target.snapshot");
  }
  loom_string_id_t target_cpu_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, processor->target_cpu, &target_cpu_id));
  loom_op_attrs(snapshot_op)[loom_target_snapshot_target_cpu_ATTR_INDEX] =
      loom_attr_string(target_cpu_id);

  target->bundle_storage.snapshot.target_cpu = processor->target_cpu;
  loom_target_ir_bundle_storage_rebind(&target->bundle_storage);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_read_stream_contents(
    iree_io_stream_t* stream, iree_allocator_t allocator,
    iree_const_byte_span_t* out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = iree_const_byte_span_empty();
  const iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
  if (stream_length < 0 || (uint64_t)stream_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO stream length is out of range");
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, (iree_host_size_t)stream_length, (void**)&data));
  iree_status_t status =
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_read(stream, (iree_host_size_t)stream_length, data,
                                 /*out_buffer_length=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_contents =
        iree_make_const_byte_span(data, (iree_host_size_t)stream_length);
  } else {
    iree_allocator_free(allocator, data);
  }
  return status;
}

static iree_status_t loom_amdgpu_module_compile_emit_hsaco(
    const loom_low_packetization_t* packetization,
    iree_arena_allocator_t* sidecar_arena, iree_allocator_t allocator,
    iree_const_byte_span_t* out_hsaco) {
  IREE_ASSERT_ARGUMENT(out_hsaco);
  *out_hsaco = iree_const_byte_span_empty();

  loom_amdgpu_wait_plan_t wait_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build(&packetization->schedule,
                                                   sidecar_arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_plan_build(
      &wait_plan, sidecar_arena, &wait_packets));

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      32 * 1024, allocator, &stream));
  iree_status_t status = loom_amdgpu_emit_kernel_hsaco_with_wait_packets(
      &packetization->schedule, &packetization->allocation, &wait_packets,
      stream, sidecar_arena);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_read_stream_contents(stream, allocator,
                                                             out_hsaco);
  }
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_amdgpu_module_compile_low_function(
    loom_module_t* module,
    const loom_amdgpu_module_compile_options_t* amdgpu_options,
    const loom_target_module_compile_options_t* target_options,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_module_compile_target_t* target,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* sidecar_arena,
    loom_amdgpu_hal_executable_t* out_executable, iree_allocator_t allocator) {
  (void)amdgpu_options;
  loom_func_like_t source_function = {0};
  IREE_RETURN_IF_ERROR(loom_target_module_compile_find_source_function(
      module, &target->bundle_storage.bundle, &source_function));
  if (!loom_low_func_def_isa(source_function.op)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU HAL executable compilation currently requires the export "
        "source to be a low.func.def");
  }

  loom_amdgpu_hal_resource_materialization_result_t materialization = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_materialize(
      module, source_function.op, &target->bundle_storage.bundle,
      &materialization, sidecar_arena));

  const loom_low_allocation_fixed_value_t* fixed_values = NULL;
  iree_host_size_t fixed_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module, source_function.op, &fixed_values, &fixed_value_count,
      sidecar_arena));

  const uint32_t max_errors = loom_target_module_compile_max_errors(
      target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  IREE_RETURN_IF_ERROR(loom_target_module_compile_verify_module(
      module, target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS));
  IREE_RETURN_IF_ERROR(loom_target_module_compile_verify_low_module(
      module, low_registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION, diagnostic_emitter,
      max_errors));

  loom_low_packetization_t packetization = {0};
  const loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = &low_registry->registry,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
  };
  IREE_RETURN_IF_ERROR(loom_low_packetize_function(
      module, source_function.op, &packetization_options, sidecar_arena,
      &packetization));

  iree_const_byte_span_t hsaco = iree_const_byte_span_empty();
  iree_status_t status = loom_amdgpu_module_compile_emit_hsaco(
      &packetization, sidecar_arena, allocator, &hsaco);

  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(allocator, &target_id_builder);
  iree_string_builder_t descriptor_symbol_builder;
  iree_string_builder_initialize(allocator, &descriptor_symbol_builder);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_append_target_id(
        target->bundle_storage.bundle.snapshot, &target_id_builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_append_descriptor_symbol(
        &target->bundle_storage.bundle, &descriptor_symbol_builder);
  }
  if (iree_status_is_ok(status)) {
    loom_amdgpu_hal_kernel_abi_layout_t abi_layout = {0};
    status = loom_amdgpu_hal_kernel_abi_layout_from_low(
        module, source_function.op, &target->bundle_storage.bundle, &abi_layout,
        sidecar_arena);
    if (iree_status_is_ok(status)) {
      loom_amdgpu_hal_executable_binding_flags_t* binding_flags = NULL;
      if (abi_layout.resource_count != 0) {
        status = iree_arena_allocate_array(
            sidecar_arena, abi_layout.resource_count, sizeof(*binding_flags),
            (void**)&binding_flags);
        if (iree_status_is_ok(status)) {
          memset(binding_flags, 0,
                 abi_layout.resource_count * sizeof(*binding_flags));
        }
      }
      if (iree_status_is_ok(status)) {
        const loom_target_hal_kernel_abi_t* hal_kernel =
            &target->bundle_storage.bundle.export_plan->hal_kernel;
        const loom_amdgpu_hal_executable_export_t export_def = {
            .symbol_name = iree_string_builder_view(&descriptor_symbol_builder),
            .workgroup_size = hal_kernel->required_workgroup_size,
            .constant_count = 0,
            .binding_flags = binding_flags,
            .binding_count = abi_layout.resource_count,
        };
        status = loom_amdgpu_emit_hal_executable(
            iree_string_builder_view(&target_id_builder), hsaco, &export_def, 1,
            allocator, out_executable);
      }
    }
  }

  iree_string_builder_deinitialize(&descriptor_symbol_builder);
  iree_string_builder_deinitialize(&target_id_builder);
  iree_allocator_free(allocator, (void*)hsaco.data);
  return status;
}

iree_status_t loom_amdgpu_compile_hal_executable(
    loom_module_t* module, const loom_amdgpu_module_compile_options_t* options,
    iree_allocator_t allocator, loom_amdgpu_hal_executable_t* out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = (loom_amdgpu_hal_executable_t){0};
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "module is required");
  }

  const loom_target_module_compile_options_t target_options = {
      .target_symbol =
          options ? options->target_symbol : iree_string_view_empty(),
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .max_errors = options ? options->max_errors : 0,
  };
  loom_target_low_descriptor_registry_t low_registry = {0};
  loom_amdgpu_low_descriptor_registry_initialize(&low_registry);
  loom_target_module_compile_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_module_compile_diagnostic_emitter_initialize(
      module, &target_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool, &sidecar_arena);

  loom_target_module_compile_target_t target = {0};
  iree_status_t status =
      loom_target_module_compile_expand_presets(module, &low_registry);
  if (iree_status_is_ok(status)) {
    status = loom_target_module_compile_verify_module(
        module, &target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_module_compile_select_target(
        module, &target_options,
        loom_amdgpu_module_compile_bundle_is_compatible, NULL,
        IREE_SV("AMDGPU HAL-native"), &target);
  }
  if (iree_status_is_ok(status) && options != NULL) {
    status = loom_amdgpu_module_compile_apply_target_cpu(module, &target,
                                                         options->target_cpu);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_low_function(
        module, options, &target_options, &low_registry, &target,
        &diagnostic_emitter, &sidecar_arena, out_executable, allocator);
  }

  if (!iree_status_is_ok(status)) {
    loom_amdgpu_hal_executable_deinitialize(out_executable, allocator);
  }
  iree_arena_deinitialize(&sidecar_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}
