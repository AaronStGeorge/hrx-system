// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/module_binary.h"

#include <cstdint>
#include <initializer_list>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/wasm/descriptors.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

constexpr uint8_t kWasmSectionType = 1;
constexpr uint8_t kWasmSectionFunction = 3;
constexpr uint8_t kWasmSectionMemory = 5;
constexpr uint8_t kWasmSectionExport = 7;
constexpr uint8_t kWasmSectionCode = 10;
constexpr uint8_t kWasmTypeI32 = 0x7F;
constexpr uint8_t kWasmTypeV128 = 0x7B;
constexpr uint8_t kWasmFunctionType = 0x60;
constexpr uint8_t kWasmExportFunction = 0;
constexpr uint8_t kWasmLimitsMinOnly = 0;

struct ModuleOwner {
  ~ModuleOwner() {
    loom_wasm_module_binary_deinitialize(&module, iree_allocator_system());
  }

  loom_wasm_module_binary_t module = {};
};

class WasmReader {
 public:
  WasmReader() = default;

  WasmReader(const uint8_t* data, iree_host_size_t length)
      : data_(data), length_(length) {}

  uint8_t ReadU8() {
    EXPECT_LT(position_, length_);
    if (position_ >= length_) {
      return 0;
    }
    return data_[position_++];
  }

  uint32_t ReadU32Leb() {
    uint32_t value = 0;
    uint32_t shift = 0;
    while (true) {
      const uint8_t byte = ReadU8();
      value |= static_cast<uint32_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) {
        return value;
      }
      shift += 7;
      EXPECT_LT(shift, 35u);
    }
  }

  std::string ReadName() {
    const uint32_t length = ReadU32Leb();
    const uint8_t* bytes = ReadBytes(length);
    return std::string(reinterpret_cast<const char*>(bytes), length);
  }

  void ReadSection(uint8_t expected_id, WasmReader* out_payload) {
    EXPECT_EQ(ReadU8(), expected_id);
    const uint32_t payload_length = ReadU32Leb();
    const uint8_t* payload = ReadBytes(payload_length);
    *out_payload = WasmReader(payload, payload_length);
  }

  void ExpectU8(uint8_t expected) { EXPECT_EQ(ReadU8(), expected); }

  void ExpectConsumed() { EXPECT_EQ(position_, length_); }

  void Skip(uint32_t byte_count) { (void)ReadBytes(byte_count); }

 private:
  const uint8_t* ReadBytes(iree_host_size_t byte_count) {
    EXPECT_LE(byte_count, length_ - position_);
    if (byte_count > length_ - position_) {
      position_ = length_;
      return data_;
    }
    const uint8_t* bytes = data_ + position_;
    position_ += byte_count;
    return bytes;
  }

  const uint8_t* data_ = nullptr;
  iree_host_size_t length_ = 0;
  iree_host_size_t position_ = 0;
};

static void ExpectValueTypeList(WasmReader* reader,
                                std::initializer_list<uint8_t> types) {
  EXPECT_EQ(reader->ReadU32Leb(), static_cast<uint32_t>(types.size()));
  for (uint8_t type : types) {
    reader->ExpectU8(type);
  }
}

static void ExpectSingleFunctionType(WasmReader* reader,
                                     std::initializer_list<uint8_t> parameters,
                                     std::initializer_list<uint8_t> results) {
  EXPECT_EQ(reader->ReadU32Leb(), 1u);
  reader->ExpectU8(kWasmFunctionType);
  ExpectValueTypeList(reader, parameters);
  ExpectValueTypeList(reader, results);
  reader->ExpectConsumed();
}

static void ExpectSingleFunctionSection(WasmReader* reader) {
  EXPECT_EQ(reader->ReadU32Leb(), 1u);
  EXPECT_EQ(reader->ReadU32Leb(), 0u);
  reader->ExpectConsumed();
}

static void ExpectSingleMemorySection(WasmReader* reader) {
  EXPECT_EQ(reader->ReadU32Leb(), 1u);
  reader->ExpectU8(kWasmLimitsMinOnly);
  EXPECT_EQ(reader->ReadU32Leb(), 1u);
  reader->ExpectConsumed();
}

static void ExpectSingleFunctionExport(WasmReader* reader,
                                       const char* expected_name) {
  EXPECT_EQ(reader->ReadU32Leb(), 1u);
  EXPECT_EQ(reader->ReadName(), expected_name);
  reader->ExpectU8(kWasmExportFunction);
  EXPECT_EQ(reader->ReadU32Leb(), 0u);
  reader->ExpectConsumed();
}

static void ExpectSingleCodeBody(WasmReader* reader) {
  EXPECT_EQ(reader->ReadU32Leb(), 1u);
  const uint32_t body_length = reader->ReadU32Leb();
  EXPECT_GT(body_length, 0u);
  reader->Skip(body_length);
  reader->ExpectConsumed();
}

class WasmModuleBinaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    ReleaseSidecars();
    if (module_) {
      loom_module_free(module_);
      module_ = nullptr;
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("wasm_module_binary_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_op_t* FindFirstLowFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  void BuildSidecars(const char* body,
                     loom_low_schedule_sidecar_t* out_schedule,
                     loom_low_allocation_sidecar_t* out_allocation) {
    std::string source =
        "target.snapshot @wasm32 {codegen_format = wasm, target_triple = "
        "\"wasm32-unknown-unknown\", data_layout = \"\", artifact_format = "
        "wasm_binary, target_cpu = \"generic\", target_features = "
        "\"+simd128\", default_pointer_bitwidth = 32, index_bitwidth = 32, "
        "offset_bitwidth = 32, memory_space_generic = 0, memory_space_global = "
        "0, memory_space_workgroup = 4294967295, memory_space_constant = 0, "
        "memory_space_private = 4294967295, memory_space_host = 4294967295, "
        "memory_space_descriptor = 4294967295}\n"
        "target.export @wasm_export {export_symbol = \"wasm_test\", abi = "
        "wasm_function, linkage = default, hal_binding_alignment = 0, "
        "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
        "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
        "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
        "target.config @wasm_config {contract_set_key = "
        "\"wasm.core.simd128\", contract_feature_bits = 7}\n"
        "target.bundle @wasm_target {snapshot = @wasm32, export_plan = "
        "@wasm_export, config = @wasm_config}\n";
    source += body;
    module_ = ParseSource(source.c_str());
    ASSERT_NE(module_, nullptr);

    loom_op_t* low_func = FindFirstLowFunction(module_);
    ASSERT_NE(low_func, nullptr);
    static const loom_low_descriptor_set_t* descriptor_sets[] = {
        loom_wasm_core_simd128_descriptor_set(),
    };
    registry_ = loom_low_descriptor_registry_t{
        .descriptor_sets = descriptor_sets,
        .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
    };

    loom_low_verify_options_t verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &registry_,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t verify_result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);

    iree_arena_initialize(&block_pool_, &sidecar_arena_);
    sidecar_arena_initialized_ = true;

    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &registry_,
    };
    loom_low_packetization_t packetization = {};
    IREE_ASSERT_OK(
        loom_low_packetize_function(module_, low_func, &packetization_options,
                                    &sidecar_arena_, &packetization));
    *out_schedule = packetization.schedule;
    *out_allocation = packetization.allocation;
  }

  void ReleaseSidecars() {
    if (sidecar_arena_initialized_) {
      iree_arena_deinitialize(&sidecar_arena_);
      sidecar_arena_initialized_ = false;
    }
  }

  void ExpectWasmHeader(WasmReader* reader) {
    reader->ExpectU8(0x00);
    reader->ExpectU8(0x61);
    reader->ExpectU8(0x73);
    reader->ExpectU8(0x6D);
    reader->ExpectU8(0x01);
    reader->ExpectU8(0x00);
    reader->ExpectU8(0x00);
    reader->ExpectU8(0x00);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_low_descriptor_registry_t registry_ = {};
  iree_arena_allocator_t sidecar_arena_ = {};
  bool sidecar_arena_initialized_ = false;
};

TEST_F(WasmModuleBinaryTest, EmitsStraightLineI32Module) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@wasm_target) @wasm_test(%input : reg<wasm.i32>) "
      "-> (reg<wasm.i32>) {\n"
      "  %one = low.const<wasm.i32.const> {i32_value = 1} : reg<wasm.i32>\n"
      "  %sum = low.op<wasm.i32.add>(%input, %one) : "
      "(reg<wasm.i32>, reg<wasm.i32>) -> reg<wasm.i32>\n"
      "  low.return %sum : reg<wasm.i32>\n"
      "}\n",
      &schedule, &allocation);

  ModuleOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_single_function_module(
      &schedule, &allocation, IREE_SV("add_one"), iree_allocator_system(),
      &owner.module));

  EXPECT_GT(owner.module.data_length, 8u);
  EXPECT_FALSE(owner.module.has_memory);

  WasmReader reader(owner.module.data, owner.module.data_length);
  ExpectWasmHeader(&reader);

  WasmReader type_section;
  reader.ReadSection(kWasmSectionType, &type_section);
  ExpectSingleFunctionType(&type_section, {kWasmTypeI32}, {kWasmTypeI32});

  WasmReader function_section;
  reader.ReadSection(kWasmSectionFunction, &function_section);
  ExpectSingleFunctionSection(&function_section);

  WasmReader export_section;
  reader.ReadSection(kWasmSectionExport, &export_section);
  ExpectSingleFunctionExport(&export_section, "add_one");

  WasmReader code_section;
  reader.ReadSection(kWasmSectionCode, &code_section);
  ExpectSingleCodeBody(&code_section);
  reader.ExpectConsumed();
}

TEST_F(WasmModuleBinaryTest, EmitsSimdMemoryModule) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@wasm_target) @wasm_test(%addr : reg<wasm.i32>, "
      "%lhs : reg<wasm.v128>, %rhs : reg<wasm.v128>) -> "
      "(reg<wasm.v128>) {\n"
      "  %loaded = low.op<wasm.v128.load>(%addr) : (reg<wasm.i32>) -> "
      "reg<wasm.v128>\n"
      "  %sum = low.op<wasm.i32x4.add>(%loaded, %lhs) : "
      "(reg<wasm.v128>, reg<wasm.v128>) -> reg<wasm.v128>\n"
      "  low.op<wasm.v128.store>(%addr, %sum) : "
      "(reg<wasm.i32>, reg<wasm.v128>)\n"
      "  %out = low.op<wasm.i32x4.mul>(%sum, %rhs) : "
      "(reg<wasm.v128>, reg<wasm.v128>) -> reg<wasm.v128>\n"
      "  low.return %out : reg<wasm.v128>\n"
      "}\n",
      &schedule, &allocation);

  ModuleOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_single_function_module(
      &schedule, &allocation, IREE_SV("simd_mem"), iree_allocator_system(),
      &owner.module));

  EXPECT_GT(owner.module.data_length, 8u);
  EXPECT_TRUE(owner.module.has_memory);

  WasmReader reader(owner.module.data, owner.module.data_length);
  ExpectWasmHeader(&reader);

  WasmReader type_section;
  reader.ReadSection(kWasmSectionType, &type_section);
  ExpectSingleFunctionType(&type_section,
                           {kWasmTypeI32, kWasmTypeV128, kWasmTypeV128},
                           {kWasmTypeV128});

  WasmReader function_section;
  reader.ReadSection(kWasmSectionFunction, &function_section);
  ExpectSingleFunctionSection(&function_section);

  WasmReader memory_section;
  reader.ReadSection(kWasmSectionMemory, &memory_section);
  ExpectSingleMemorySection(&memory_section);

  WasmReader export_section;
  reader.ReadSection(kWasmSectionExport, &export_section);
  ExpectSingleFunctionExport(&export_section, "simd_mem");

  WasmReader code_section;
  reader.ReadSection(kWasmSectionCode, &code_section);
  ExpectSingleCodeBody(&code_section);
  reader.ExpectConsumed();
}

TEST_F(WasmModuleBinaryTest, RejectsEmptyExportName) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@wasm_target) @wasm_test(%input : reg<wasm.i32>) "
      "-> (reg<wasm.i32>) {\n"
      "  low.return %input : reg<wasm.i32>\n"
      "}\n",
      &schedule, &allocation);

  ModuleOwner owner;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_wasm_emit_single_function_module(
                            &schedule, &allocation, iree_string_view_empty(),
                            iree_allocator_system(), &owner.module));
}

}  // namespace
}  // namespace loom
