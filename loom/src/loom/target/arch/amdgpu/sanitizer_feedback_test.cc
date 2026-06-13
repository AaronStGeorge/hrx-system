// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/sanitizer/options.h"
#include "loom/sanitizer/site_table.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/target/amdgpu/artifact_provider.h"

namespace loom {
namespace {

using ::loom::testing::ModulePtr;

constexpr uint64_t kReportFaultAddress = UINT64_C(0x0123456789ABCDEF);
constexpr uint64_t kReportAccessLength = 16;
constexpr loom_sanitizer_site_id_t kReportSiteId = 0;
constexpr uint64_t kReportShadowAddress = UINT64_C(0x0000056789ABCDEF);
constexpr uint64_t kReportShadowValue = UINT64_C(0xF0);

uint32_t LoadLeU32(const uint8_t* data, iree_host_size_t offset) {
  return ((uint32_t)data[offset]) | ((uint32_t)data[offset + 1] << 8) |
         ((uint32_t)data[offset + 2] << 16) |
         ((uint32_t)data[offset + 3] << 24);
}

std::string StatusToStringAndFree(iree_status_t status) {
  iree_host_size_t length = 0;
  iree_status_format(status, /*buffer_capacity=*/0, /*buffer=*/nullptr,
                     &length);
  std::vector<char> buffer(length + 1);
  if (length != 0) {
    iree_status_format(status, buffer.size(), buffer.data(), &length);
  }
  std::string message(buffer.data(), length);
  iree_status_free(status);
  return message;
}

iree_status_t InitializeAmdgpuContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_ops_register_dialect(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

iree_status_t ReadDeviceBufferBytes(iree_hal_device_t* device,
                                    iree_hal_allocator_t* allocator,
                                    iree_hal_buffer_t* source_buffer,
                                    iree_device_size_t source_offset,
                                    iree_device_size_t length,
                                    std::vector<uint8_t>* out_data) {
  out_data->clear();
  if (source_offset > iree_hal_buffer_byte_length(source_buffer)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "readback offset %" PRIu64 " exceeds buffer length %" PRIu64,
        (uint64_t)source_offset,
        (uint64_t)iree_hal_buffer_byte_length(source_buffer));
  }
  if (length > iree_hal_buffer_byte_length(source_buffer) - source_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "readback length %" PRIu64 " at offset %" PRIu64
        " exceeds buffer length %" PRIu64,
        (uint64_t)length, (uint64_t)source_offset,
        (uint64_t)iree_hal_buffer_byte_length(source_buffer));
  }
  if (length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "readback length %" PRIu64
                            " exceeds host vector capacity",
                            (uint64_t)length);
  }

  std::vector<uint8_t> data((iree_host_size_t)length);
  iree_hal_buffer_t* staging_buffer = nullptr;
  iree_hal_semaphore_t* semaphore = nullptr;

  iree_hal_buffer_params_t staging_params = {};
  staging_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  staging_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;

  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, staging_params, length, &staging_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
  }
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&semaphore,
      /*.payload_values=*/&signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_list, source_buffer, source_offset, staging_buffer,
        /*target_offset=*/0, length, IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(signal_list, iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_read(staging_buffer, /*offset=*/0, data.data(),
                                      data.size());
  }
  if (iree_status_is_ok(status)) {
    *out_data = std::move(data);
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(staging_buffer);
  return status;
}

template <typename T>
iree_status_t ReadDeviceBufferData(iree_hal_device_t* device,
                                   iree_hal_allocator_t* allocator,
                                   iree_hal_buffer_t* source_buffer,
                                   std::vector<T>* out_data) {
  out_data->clear();
  const iree_device_size_t byte_length =
      iree_hal_buffer_byte_length(source_buffer);
  if (byte_length % sizeof(T) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "readback length %" PRIu64
                            " is not divisible by element size %" PRIhsz,
                            (uint64_t)byte_length, sizeof(T));
  }

  std::vector<uint8_t> bytes;
  IREE_RETURN_IF_ERROR(ReadDeviceBufferBytes(device, allocator, source_buffer,
                                             /*source_offset=*/0, byte_length,
                                             &bytes));
  std::vector<T> data((iree_host_size_t)(byte_length / sizeof(T)));
  std::memcpy(data.data(), bytes.data(), (iree_host_size_t)byte_length);
  *out_data = std::move(data);
  return iree_ok_status();
}

class AmdgpuAsanDeviceEventRecorder {
 public:
  AmdgpuAsanDeviceEventRecorder() = default;

  AmdgpuAsanDeviceEventRecorder(const AmdgpuAsanDeviceEventRecorder&) = delete;
  AmdgpuAsanDeviceEventRecorder& operator=(
      const AmdgpuAsanDeviceEventRecorder&) = delete;

  iree_hal_device_event_sink_t sink() {
    iree_hal_device_event_sink_t sink;
    sink.fn = AmdgpuAsanDeviceEventRecorder::Capture;
    sink.user_data = this;
    return sink;
  }

  void WaitForAsanReportCount(iree_host_size_t expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return asan_report_count_ >= expected_count; });
  }

  iree_host_size_t asan_report_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asan_report_count_;
  }

  iree_hal_device_event_source_t last_source() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_source_;
  }

  iree_hal_device_asan_report_t last_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_report_;
  }

 private:
  static void Capture(void* user_data, const iree_hal_device_event_t* event) {
    auto* recorder = static_cast<AmdgpuAsanDeviceEventRecorder*>(user_data);
    if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT ||
        !event->payload.data ||
        event->payload.data_length < sizeof(iree_hal_device_asan_report_t)) {
      return;
    }
    std::lock_guard<std::mutex> lock(recorder->mutex_);
    recorder->last_source_ = event->source;
    std::memcpy(&recorder->last_report_, event->payload.data,
                sizeof(recorder->last_report_));
    ++recorder->asan_report_count_;
    recorder->condition_.notify_all();
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  iree_host_size_t asan_report_count_ = 0;
  iree_hal_device_event_source_t last_source_ =
      iree_hal_device_event_source_default();
  iree_hal_device_asan_report_t last_report_ = {};
};

class AmdgpuAsanHalRuntime {
 public:
  AmdgpuAsanHalRuntime() = default;
  ~AmdgpuAsanHalRuntime() { Deinitialize(); }

  AmdgpuAsanHalRuntime(const AmdgpuAsanHalRuntime&) = delete;
  AmdgpuAsanHalRuntime& operator=(const AmdgpuAsanHalRuntime&) = delete;

  iree_status_t Initialize() {
    return Initialize(iree_hal_device_event_sink_discard());
  }

  iree_status_t Initialize(iree_hal_device_event_sink_t event_sink) {
    iree_async_proactor_pool_t* proactor_pool = nullptr;
    iree_async_frontier_tracker_t* frontier_tracker = nullptr;

    iree_status_t status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
    if (iree_status_is_ok(status)) {
      status = iree_async_frontier_tracker_create(
          iree_async_frontier_tracker_options_default(),
          iree_allocator_system(), &frontier_tracker);
    }

    iree_hal_driver_registry_t* driver_registry = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_registry_allocate(iree_allocator_system(),
                                                 &driver_registry);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_driver_module_register(driver_registry);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_registry_try_create(
          driver_registry, IREE_SV("amdgpu"), iree_allocator_system(),
          &driver_);
    }
    iree_hal_driver_registry_free(driver_registry);

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.event_sink = event_sink;
    create_params.proactor_pool = proactor_pool;
    iree_string_pair_t device_parameters[1] = {
        iree_make_cstring_pair("hal.sanitizer", "asan"),
    };
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_create_device_by_ordinal(
          driver_, /*device_ordinal=*/0, IREE_ARRAYSIZE(device_parameters),
          device_parameters, &create_params, iree_allocator_system(),
          &runtime_.device);
    }
    iree_async_proactor_pool_release(proactor_pool);

    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          runtime_.device, frontier_tracker, iree_allocator_system(),
          &runtime_.device_group);
    }
    iree_async_frontier_tracker_release(frontier_tracker);

    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_cache_create(
          runtime_.device, IREE_SV("loom"), &runtime_.executable_cache);
    }
    if (!iree_status_is_ok(status)) {
      Deinitialize();
    }
    return status;
  }

  const loom_run_hal_runtime_t* runtime() const { return &runtime_; }

  iree_hal_allocator_t* allocator() const {
    return iree_hal_device_allocator(runtime_.device);
  }

 private:
  void Deinitialize() {
    loom_run_hal_runtime_deinitialize(&runtime_);
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
  }

  loom_run_hal_runtime_t runtime_ = {};
  iree_hal_driver_t* driver_ = nullptr;
};

iree_status_t GetOrCreateModuleSymbol(loom_module_t* module,
                                      iree_string_view_t name,
                                      loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
  }
  *out_symbol_ref = (loom_symbol_ref_t){
      /*.module_id=*/0,
      /*.symbol_id=*/symbol_id,
  };
  return iree_ok_status();
}

iree_status_t BuildU32Attr(loom_builder_t* builder, iree_string_view_t name,
                           uint32_t value, loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_attr = {};
  out_attr->name_id = name_id;
  out_attr->value = loom_attr_i64(value);
  return iree_ok_status();
}

iree_status_t AppendSingleSanitizerSiteTable(loom_module_t* module) {
  loom_sanitizer_site_row_t row = {};
  row.site_id = kReportSiteId;
  row.op_kind = LOOM_OP_SANITIZER_ASSERT_ACCESS;
  row.location = LOOM_LOCATION_UNKNOWN;
  row.payload_location = LOOM_LOCATION_UNKNOWN;
  row.source_location = LOOM_LOCATION_UNKNOWN;

  loom_sanitizer_site_collection_t collection = {};
  collection.rows = &row;
  collection.row_count = 1;

  iree_const_byte_span_t site_table = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_encode(
      module, &collection, &module->arena, &site_table));

  loom_symbol_ref_t symbol_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(GetOrCreateModuleSymbol(
      module, IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME), &symbol_ref));
  loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->defining_op != nullptr) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "test sanitizer site table symbol is already defined");
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* rodata_op = nullptr;
  return loom_global_rodata_build(
      &builder, LOOM_GLOBAL_RODATA_BUILD_FLAG_HAS_ALIGNMENT, symbol_ref,
      site_table, 8, LOOM_LOCATION_UNKNOWN, &rodata_op);
}

void ExpectSingleSanitizerSiteTable(loom_module_t* module) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      module, IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME), &name_id));
  const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  ASSERT_NE(symbol->defining_op, nullptr);
  ASSERT_TRUE(loom_global_rodata_isa(symbol->defining_op));

  const iree_const_byte_span_t contents =
      loom_global_rodata_contents(symbol->defining_op);
  ASSERT_GE(contents.data_length, LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH +
                                      LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  EXPECT_EQ(
      LoadLeU32(contents.data, LOOM_SANITIZER_SITE_TABLE_HEADER_MAGIC_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_MAGIC);
  EXPECT_EQ(contents.data[LOOM_SANITIZER_SITE_TABLE_HEADER_VERSION_OFFSET],
            LOOM_SANITIZER_SITE_TABLE_VERSION);
  EXPECT_EQ(LoadLeU32(contents.data,
                      LOOM_SANITIZER_SITE_TABLE_HEADER_ROW_COUNT_OFFSET),
            1u);

  const uint8_t* record =
      contents.data + LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH;
  EXPECT_EQ(LoadLeU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
            kReportSiteId);
  EXPECT_EQ(LoadLeU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET),
            LOOM_OP_SANITIZER_ASSERT_ACCESS);
  EXPECT_EQ(LoadLeU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET),
            0u);
}

iree_status_t BuildConstU32(loom_builder_t* builder,
                            const loom_low_descriptor_set_t* descriptor_set,
                            loom_amdgpu_descriptor_ref_t descriptor_ref,
                            uint32_t value, loom_type_t result_type,
                            loom_location_id_t location,
                            loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = nullptr;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));

  loom_named_attr_t imm32_attr = {};
  IREE_RETURN_IF_ERROR(
      BuildU32Attr(builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

iree_status_t BuildRegisterU32Constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t register_class, uint32_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_build_register_type(descriptor_set, register_class,
                                   /*unit_count=*/1, &register_type));
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR
          ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
  return BuildConstU32(builder, descriptor_set, descriptor_ref, value,
                       register_type, location, out_value);
}

iree_status_t BuildRegisterU64Constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t register_class, uint64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(builder, descriptor_set,
                                                register_class, (uint32_t)value,
                                                location, &low_value));
  loom_value_id_t high_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      BuildRegisterU32Constant(builder, descriptor_set, register_class,
                               (uint32_t)(value >> 32), location, &high_value));

  loom_type_t register_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_build_register_type(descriptor_set, register_class,
                                   /*unit_count=*/2, &register_x2_type));
  const loom_value_id_t parts[] = {low_value, high_value};
  loom_op_t* concat_op = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts),
                            register_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t FindKernelOpByName(loom_module_t* module, iree_string_view_t name,
                                 loom_op_t** out_op) {
  *out_op = nullptr;
  loom_block_t* module_block = loom_module_block(module);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(module_block, op) {
    if (!loom_low_kernel_def_isa(op)) continue;
    const loom_symbol_ref_t callee_ref = loom_low_kernel_def_callee(op);
    if (callee_ref.module_id != 0 ||
        callee_ref.symbol_id >= module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low.kernel.def has an invalid callee symbol");
    }
    const loom_symbol_t* symbol =
        &module->symbols.entries[callee_ref.symbol_id];
    if (symbol->name_id == LOOM_STRING_ID_INVALID ||
        symbol->name_id >= module->strings.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low.kernel.def callee symbol has an invalid name");
    }
    if (!iree_string_view_equal(module->strings.entries[symbol->name_id],
                                name)) {
      continue;
    }
    if (*out_op != nullptr) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "expected one low.kernel.def named '%.*s' but found more than one",
          (int)name.size, name.data);
    }
    *out_op = op;
  }
  if (*out_op == nullptr) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "low.kernel.def named '%.*s' was not found",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

iree_status_t InsertBlockAfter(loom_module_t* module, loom_block_t* block,
                               loom_block_t** out_block) {
  *out_block = nullptr;
  if (block->parent_region == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected block to belong to a region");
  }
  return loom_region_insert_block(module, block->parent_region,
                                  (uint16_t)(block->region_index + 1),
                                  out_block);
}

iree_status_t BuildAsanReportPublishAndReturn(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location) {
  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* feedback_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, entry_block, &feedback_block));
  loom_block_t* return_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, feedback_block, &return_block));

  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
      builder, descriptor_set, feedback_config_symbol, location,
      &config_values));
  loom_value_id_t feedback_enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_enabled_scc(
      builder, descriptor_set, config_values.flags, location,
      &feedback_enabled_scc));
  loom_op_t* enabled_branch = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, feedback_enabled_scc,
                                              feedback_block, return_block,
                                              location, &enabled_branch));

  loom_builder_set_block(builder, feedback_block);
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
      builder, descriptor_set, config_values.channel_base, location,
      &channel_values));
  const uint32_t packet_length = (uint32_t)loom_amdgpu_feedback_packet_length(
      LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_BYTE_LENGTH);
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation(
      builder, descriptor_set, channel_values.address, channel_values.ring_base,
      channel_values.ring_capacity, packet_length, location, &reservation));

  loom_block_t* reservation_block = builder->ip.block;
  loom_block_t* report_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, reservation_block, &report_block));
  loom_value_id_t reservation_succeeded_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_succeeded_scc(
      builder, descriptor_set, reservation.reserved_mask, location,
      &reservation_succeeded_scc));
  loom_op_t* reserved_branch = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, reservation_succeeded_scc, report_block,
                             return_block, location, &reserved_branch));

  loom_builder_set_block(builder, report_block);
  const loom_amdgpu_feedback_packet_header_t header = {
      /*.record_length=*/packet_length,
      /*.kind=*/LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
      /*.flags=*/LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_NONE,
      /*.sequence=*/reservation.sequence,
      /*.source_dispatch_ptr=*/source->dispatch_ptr,
      /*.source_workgroup_id_x=*/source->workgroup_id_x,
      /*.source_workitem_id_x=*/source->workitem_id_x,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_header(
      builder, descriptor_set, &reservation.packet_address, &header, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_payload(
      builder, descriptor_set, &reservation.packet_address, report, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet(
      builder, descriptor_set, &reservation.packet_address,
      config_values.notify_signal, location));
  loom_op_t* publish_branch = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, return_block,
                                         /*args=*/nullptr, /*args_count=*/0,
                                         location, &publish_branch));

  loom_builder_set_block(builder, return_block);
  loom_op_t* return_op = nullptr;
  return loom_low_return_build(builder, /*values=*/nullptr, /*values_count=*/0,
                               location, &return_op);
}

iree_status_t AppendAsanReportProducer(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t kernel_name) {
  loom_op_t* kernel_op = nullptr;
  IREE_RETURN_IF_ERROR(FindKernelOpByName(module, kernel_name, &kernel_op));
  loom_region_t* body = loom_low_kernel_def_body(kernel_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  if (entry_block->op_count != 1 ||
      !loom_low_return_isa(entry_block->last_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected report kernel skeleton to contain only "
                            "one low.return terminator");
  }
  IREE_RETURN_IF_ERROR(loom_op_erase(module, entry_block->last_op));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  builder.ip.parent_op = kernel_op;
  constexpr loom_location_id_t location = LOOM_LOCATION_UNKNOWN;

  loom_symbol_ref_t feedback_config_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(GetOrCreateModuleSymbol(
      module, IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME),
      &feedback_config_symbol));

  loom_amdgpu_sanitizer_report_source_t source = {};
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*value=*/0, location, &source.dispatch_ptr));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*value=*/0, location, &source.workgroup_id_x));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      /*value=*/0, location, &source.workitem_id_x));

  loom_amdgpu_sanitizer_access_report_t report = {
      /*.access_kind=*/LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE,
      /*.flags=*/LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE,
  };
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportFaultAddress, location, &report.fault_address));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportAccessLength, location, &report.access_size));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      (uint64_t)kReportSiteId, location, &report.site_id));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportShadowAddress, location, &report.shadow_address));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportShadowValue, location, &report.shadow_value));

  return BuildAsanReportPublishAndReturn(&builder, descriptor_set,
                                         feedback_config_symbol, &source,
                                         &report, location);
}

class AmdgpuHalSanitizerFeedbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(InitializeAmdgpuContext(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_text_parse_options_t ParseOptions() const {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    return options;
  }

  iree_status_t ParseModuleSource(iree_string_view_t source,
                                  iree_string_view_t source_name,
                                  ModulePtr* out_module) {
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_text_parse(source, source_name, &context_,
                                         &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_status_t ParseArithmeticModule(iree_string_view_t target_key,
                                      ModulePtr* out_module) {
    std::string source;
    source += "amdgpu.target<";
    source.append(target_key.data, target_key.size);
    source += "> @gfx_target\n";
    source +=
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
        "reg<amdgpu.vgpr>\n"
        "  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : "
        "reg<amdgpu.vgpr>\n"
        "  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.return\n"
        "}\n";

    IREE_RETURN_IF_ERROR(ParseModuleSource(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom"), out_module));
    return iree_ok_status();
  }

  iree_status_t ParseReportModule(const loom_run_hal_device_target_t* target,
                                  ModulePtr* out_module) {
    if (target->target_bundle == nullptr ||
        target->target_bundle->config == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "selected AMDGPU target has no bundle config");
    }
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_lookup(
            &low_registry_.registry,
            target->target_bundle->config->contract_set_key);
    if (descriptor_set == nullptr) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE, "AMDGPU descriptor set is not linked: %.*s",
          (int)target->target_bundle->config->contract_set_key.size,
          target->target_bundle->config->contract_set_key.data);
    }

    std::string source;
    source += "amdgpu.target<";
    source.append(target->target_key.data, target->target_key.size);
    source += "> @gfx_target\n";
    source +=
        "low.kernel.def target(@gfx_target) workgroup_size(1, 1, 1) "
        "@loom_idle_kernel() {\n"
        "  low.return\n"
        "}\n"
        "low.kernel.def target(@gfx_target) workgroup_size(1, 1, 1) "
        "@loom_report_kernel() {\n"
        "  low.return\n"
        "}\n";

    ModulePtr module;
    IREE_RETURN_IF_ERROR(ParseModuleSource(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom"), &module));
    IREE_RETURN_IF_ERROR(AppendSingleSanitizerSiteTable(module.get()));
    IREE_RETURN_IF_ERROR(AppendAsanReportProducer(
        module.get(), descriptor_set, IREE_SV("loom_report_kernel")));
    *out_module = std::move(module);
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

class AmdgpuHalArtifact {
 public:
  AmdgpuHalArtifact() = default;
  ~AmdgpuHalArtifact() { Deinitialize(); }

  AmdgpuHalArtifact(const AmdgpuHalArtifact&) = delete;
  AmdgpuHalArtifact& operator=(const AmdgpuHalArtifact&) = delete;

  loom_run_hal_artifact_t* value() { return &artifact_; }

 private:
  void Deinitialize() {
    loom_amdgpu_hal_artifact_provider.deinitialize_artifact(
        &loom_amdgpu_hal_artifact_provider, &artifact_,
        iree_allocator_system());
    artifact_ = {};
  }

  loom_run_hal_artifact_t artifact_ = {};
};

class AmdgpuHalPreparedCandidate {
 public:
  AmdgpuHalPreparedCandidate() = default;
  ~AmdgpuHalPreparedCandidate() {
    loom_run_hal_prepared_candidate_deinitialize(&candidate_);
  }

  AmdgpuHalPreparedCandidate(const AmdgpuHalPreparedCandidate&) = delete;
  AmdgpuHalPreparedCandidate& operator=(const AmdgpuHalPreparedCandidate&) =
      delete;

  loom_run_hal_prepared_candidate_t* value() { return &candidate_; }

 private:
  loom_run_hal_prepared_candidate_t candidate_ = {};
};

TEST_F(AmdgpuHalSanitizerFeedbackTest,
       PublishesRuntimeGlobalsFromLoomNativeHsaco) {
  AmdgpuAsanHalRuntime runtime;
  iree_status_t status = runtime.Initialize();
  if (iree_status_is_unavailable(status) || iree_status_is_not_found(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  loom_run_hal_device_target_t target = {};
  status = loom_amdgpu_hal_artifact_provider.select_device_target(
      &loom_amdgpu_hal_artifact_provider, runtime.runtime(),
      iree_allocator_system(), &target);
  if (iree_status_is_unavailable(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  ModulePtr module;
  IREE_ASSERT_OK(ParseArithmeticModule(target.target_key, &module));

  const loom_target_pipeline_options_t target_pipeline_options = {
      /*.source_to_low_max_errors=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_ACCESS,
      },
  };
  AmdgpuHalArtifact artifact;
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/(loom_diagnostic_sink_t){},
      /*source_resolver=*/(loom_source_resolver_t){}, /*max_errors=*/20,
      &target_pipeline_options,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, artifact.value()));
  ASSERT_TRUE(emitted);

  AmdgpuHalPreparedCandidate candidate;
  IREE_ASSERT_OK(loom_run_hal_prepared_candidate_prepare(
      runtime.runtime(), artifact.value(), candidate.value()));

  bool found = false;
  iree_hal_executable_global_t asan_global =
      iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      candidate.value()->executable,
      IREE_SV(IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME), &found, &asan_global));
  ASSERT_TRUE(found);
  iree_hal_executable_global_info_t asan_info = {};
  IREE_ASSERT_OK(iree_hal_executable_global_info(candidate.value()->executable,
                                                 asan_global, &asan_info));
  EXPECT_TRUE(iree_string_view_equal(
      asan_info.name, IREE_SV(IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME)));
  ASSERT_EQ(asan_info.byte_length, sizeof(iree_hal_amdgpu_asan_config_t));

  iree_hal_buffer_t* asan_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      candidate.value()->executable, asan_global, IREE_HAL_QUEUE_AFFINITY_ANY,
      &asan_buffer));
  ASSERT_NE(asan_buffer, nullptr);
  std::vector<iree_hal_amdgpu_asan_config_t> asan_configs;
  IREE_ASSERT_OK(ReadDeviceBufferData(runtime.runtime()->device,
                                      runtime.allocator(), asan_buffer,
                                      &asan_configs));
  ASSERT_EQ(asan_configs.size(), 1u);
  EXPECT_EQ(asan_configs[0].record_length, sizeof(asan_configs[0]));
  EXPECT_EQ(asan_configs[0].abi_version,
            IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0);
  EXPECT_NE(asan_configs[0].flags & IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED,
            0u);
  EXPECT_NE(asan_configs[0].shadow_base, 0u);

  iree_hal_executable_global_t feedback_global =
      iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      candidate.value()->executable,
      IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME), &found,
      &feedback_global));
  ASSERT_TRUE(found);
  iree_hal_executable_global_info_t feedback_info = {};
  IREE_ASSERT_OK(iree_hal_executable_global_info(
      candidate.value()->executable, feedback_global, &feedback_info));
  EXPECT_TRUE(iree_string_view_equal(
      feedback_info.name,
      IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME)));
  ASSERT_EQ(feedback_info.byte_length,
            sizeof(iree_hal_amdgpu_feedback_config_t));

  iree_hal_buffer_t* feedback_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      candidate.value()->executable, feedback_global,
      IREE_HAL_QUEUE_AFFINITY_ANY, &feedback_buffer));
  ASSERT_NE(feedback_buffer, nullptr);
  std::vector<iree_hal_amdgpu_feedback_config_t> feedback_configs;
  IREE_ASSERT_OK(ReadDeviceBufferData(runtime.runtime()->device,
                                      runtime.allocator(), feedback_buffer,
                                      &feedback_configs));
  ASSERT_EQ(feedback_configs.size(), 1u);
  EXPECT_EQ(feedback_configs[0].record_length, sizeof(feedback_configs[0]));
  EXPECT_EQ(feedback_configs[0].abi_version,
            IREE_HAL_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_0);
  EXPECT_NE(
      feedback_configs[0].flags & IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED,
      0u);
  EXPECT_NE(feedback_configs[0].channel_base, 0u);
  EXPECT_NE(feedback_configs[0].notify_signal.handle, 0u);
}

TEST_F(AmdgpuHalSanitizerFeedbackTest,
       PublishesAsanFeedbackPacketFromLoomNativeHsaco) {
  AmdgpuAsanDeviceEventRecorder recorder;
  AmdgpuAsanHalRuntime runtime;
  iree_status_t status = runtime.Initialize(recorder.sink());
  if (iree_status_is_unavailable(status) || iree_status_is_not_found(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  loom_run_hal_device_target_t target = {};
  status = loom_amdgpu_hal_artifact_provider.select_device_target(
      &loom_amdgpu_hal_artifact_provider, runtime.runtime(),
      iree_allocator_system(), &target);
  if (iree_status_is_unavailable(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  ModulePtr module;
  status = ParseReportModule(&target, &module);
  if (iree_status_is_unavailable(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);
  ASSERT_NO_FATAL_FAILURE(ExpectSingleSanitizerSiteTable(module.get()));

  const loom_target_pipeline_options_t target_pipeline_options = {
      /*.source_to_low_max_errors=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_ACCESS,
      },
  };
  AmdgpuHalArtifact artifact;
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/(loom_diagnostic_sink_t){},
      /*source_resolver=*/(loom_source_resolver_t){}, /*max_errors=*/20,
      &target_pipeline_options,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, artifact.value()));
  ASSERT_TRUE(emitted);

  AmdgpuHalPreparedCandidate candidate;
  IREE_ASSERT_OK(loom_run_hal_prepared_candidate_prepare(
      runtime.runtime(), artifact.value(), candidate.value()));
  ASSERT_EQ(iree_hal_executable_function_count(candidate.value()->executable),
            2u);

  iree_vm_list_t* bindings = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                     /*capacity=*/0, iree_allocator_system(),
                                     &bindings));
  loom_run_hal_invocation_options_t options;
  loom_run_hal_invocation_options_initialize(&options);
  options.function_name = IREE_SV("loom_report_kernel");
  status =
      loom_run_hal_dispatch(runtime.runtime()->device,
                            candidate.value()->executable, bindings, &options);
  iree_vm_list_release(bindings);

  const iree_status_code_t dispatch_code = iree_status_code(status);
  if (dispatch_code == IREE_STATUS_ABORTED) {
    iree_status_free(status);
  } else {
    IREE_ASSERT_OK(status);
  }

  IREE_ASSERT_OK(iree_hal_device_queue_flush(runtime.runtime()->device,
                                             IREE_HAL_QUEUE_AFFINITY_ANY));
  recorder.WaitForAsanReportCount(1);
  EXPECT_EQ(recorder.asan_report_count(), 1u);
  iree_hal_device_asan_report_t report = recorder.last_report();
  EXPECT_EQ(report.record_length, sizeof(report));
  EXPECT_EQ(report.abi_version, IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(report.access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(report.fault_address, kReportFaultAddress);
  EXPECT_EQ(report.access_length, kReportAccessLength);
  EXPECT_EQ(report.site_id, (uint64_t)kReportSiteId);
  EXPECT_EQ(report.shadow_address, kReportShadowAddress);
  EXPECT_EQ(report.shadow_value, kReportShadowValue);
  EXPECT_EQ(report.workgroup_id[0], 0u);
  EXPECT_EQ(report.workitem_id[0], 0u);
  EXPECT_EQ(report.source_dispatch_ptr, 0u);

  iree_hal_device_event_source_t source = recorder.last_source();
  EXPECT_TRUE(iree_string_view_equal(source.driver_id, IREE_SV("amdgpu")));
  EXPECT_NE(source.physical_device_ordinal, UINT32_MAX);
  IREE_EXPECT_OK(iree_hal_device_queue_flush(runtime.runtime()->device,
                                             IREE_HAL_QUEUE_AFFINITY_ANY));
}

}  // namespace
}  // namespace loom
