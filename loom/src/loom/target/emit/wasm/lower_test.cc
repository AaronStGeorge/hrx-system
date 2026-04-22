// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/wasm/low_registry.h"
#include "loom/target/emit/wasm/module_binary.h"
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
constexpr uint8_t kWasmOpcodeReturn = 0x0F;
constexpr uint8_t kWasmOpcodeI32Const = 0x41;
constexpr uint8_t kWasmOpcodeI32Add = 0x6A;
constexpr uint8_t kWasmOpcodeI32Sub = 0x6B;
constexpr uint8_t kWasmOpcodeI32Mul = 0x6C;
constexpr uint8_t kWasmOpcodeEnd = 0x0B;
constexpr uint8_t kWasmOpcodeSimdPrefix = 0xFD;
constexpr uint8_t kWasmSimdV128Load = 0x00;
constexpr uint8_t kWasmSimdV128Store = 0x0B;
constexpr uint8_t kWasmSimdI32x4AddLeb0 = 0xAE;
constexpr uint8_t kWasmSimdI32x4MulLeb0 = 0xB5;
constexpr uint8_t kWasmSimdI32x4Leb1 = 0x01;

struct CollectedEmission {
  const loom_error_def_t* error = nullptr;
  const loom_op_t* op = nullptr;
  std::vector<std::string> string_params;
};

struct EmissionCollector {
  std::vector<CollectedEmission> emissions;

  iree_diagnostic_emitter_t emitter() {
    return iree_diagnostic_emitter_t{
        .fn = Collect,
        .user_data = this,
    };
  }

 private:
  static std::string CopyString(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }

  static iree_status_t Collect(void* user_data,
                               const loom_diagnostic_emission_t* emission) {
    auto* collector = static_cast<EmissionCollector*>(user_data);
    CollectedEmission entry;
    entry.error = emission->error;
    entry.op = emission->op;
    for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
      const loom_diagnostic_param_t* param = &emission->params[i];
      if (param->kind == LOOM_PARAM_STRING) {
        entry.string_params.push_back(CopyString(param->string));
      }
    }
    collector->emissions.push_back(std::move(entry));
    return iree_ok_status();
  }
};

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

struct WasmModuleOwner {
  ~WasmModuleOwner() {
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

  std::vector<uint8_t> ReadCodeBodyBytes() {
    EXPECT_EQ(ReadU32Leb(), 1u);
    const uint32_t body_length = ReadU32Leb();
    const uint8_t* bytes = ReadBytes(body_length);
    return std::vector<uint8_t>(bytes, bytes + body_length);
  }

  void ExpectU8(uint8_t expected) { EXPECT_EQ(ReadU8(), expected); }

  void ExpectConsumed() { EXPECT_EQ(position_, length_); }

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

static bool ByteVectorContains(const std::vector<uint8_t>& bytes,
                               uint8_t value) {
  return std::find(bytes.begin(), bytes.end(), value) != bytes.end();
}

static bool ByteVectorContainsSubsequence(const std::vector<uint8_t>& bytes,
                                          const std::vector<uint8_t>& needle) {
  return std::search(bytes.begin(), bytes.end(), needle.begin(),
                     needle.end()) != bytes.end();
}

class WasmLowerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_wasm_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    ReleaseSidecars();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("wasm_lower_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  ModulePtr ParseAndLowerTargetedSource(const char* source,
                                        EmissionCollector* collector,
                                        loom_low_lower_result_t* out_result) {
    ModulePtr module = ParseSource(source);

    loom_low_lower_policy_registry_t policy_registry = {};
    loom_wasm_low_lower_policy_registry_initialize(&policy_registry);
    iree_arena_allocator_t selection_arena;
    iree_arena_initialize(module->arena.block_pool, &selection_arena);
    const loom_low_source_selection_options_t selection_options = {
        .descriptor_registry = &target_registry_.registry,
        .policy_registry = &policy_registry,
        .lowering_kind = IREE_SVL("WASM source-to-low"),
    };
    loom_low_source_selection_t selection = {};
    IREE_CHECK_OK(loom_low_select_source_func(module.get(), &selection_options,
                                              &selection_arena, &selection));
    const loom_low_lower_options_t options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = selection.policy,
        .emitter = collector->emitter(),
        .max_errors = 20,
    };
    IREE_CHECK_OK(loom_low_lower_function(module.get(), selection.func,
                                          &options, out_result));
    iree_arena_deinitialize(&selection_arena);
    return module;
  }

  void VerifyLowModule(loom_module_t* module) {
    EmissionCollector collector;
    const loom_low_verify_options_t options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .emitter = collector.emitter(),
        .max_errors = 20,
    };
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(loom_low_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
    EXPECT_TRUE(collector.emissions.empty());
  }

  void Packetize(loom_module_t* module, loom_op_t* low_function,
                 loom_low_packetization_t* out_packetization) {
    ReleaseSidecars();
    iree_arena_initialize(&block_pool_, &sidecar_arena_);
    sidecar_arena_initialized_ = true;

    loom_low_packetization_options_t options = {
        .descriptor_registry = &target_registry_.registry,
    };
    IREE_ASSERT_OK(loom_low_packetize_function(
        module, low_function, &options, &sidecar_arena_, out_packetization));
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
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t target_registry_ = {};
  iree_arena_allocator_t sidecar_arena_ = {};
  bool sidecar_arena_initialized_ = false;
};

TEST_F(WasmLowerTest, LowersSemanticFunctionAndEmitsWasmModule) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @wasm_target preset(\"wasm-simd128\")\n"
      "func.def target(@wasm_target) @arith(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %seven = scalar.constant 7 : i32\n"
      "  %sum = scalar.addi %lhs, %seven : i32\n"
      "  %diff = scalar.subi %sum, %rhs : i32\n"
      "  func.return %diff : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  ASSERT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());
  VerifyLowModule(module.get());

  loom_low_packetization_t packetization = {};
  Packetize(module.get(), lower_result.low_func_op, &packetization);

  WasmModuleOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_single_function_module(
      &packetization.schedule, &packetization.allocation, IREE_SV("arith"),
      iree_allocator_system(), &owner.module));

  EXPECT_GT(owner.module.data_length, 8u);
  EXPECT_FALSE(owner.module.has_memory);

  WasmReader reader(owner.module.data, owner.module.data_length);
  ExpectWasmHeader(&reader);

  WasmReader type_section;
  reader.ReadSection(kWasmSectionType, &type_section);
  EXPECT_EQ(type_section.ReadU32Leb(), 1u);
  type_section.ExpectU8(kWasmFunctionType);
  EXPECT_EQ(type_section.ReadU32Leb(), 2u);
  type_section.ExpectU8(kWasmTypeI32);
  type_section.ExpectU8(kWasmTypeI32);
  EXPECT_EQ(type_section.ReadU32Leb(), 1u);
  type_section.ExpectU8(kWasmTypeI32);
  type_section.ExpectConsumed();

  WasmReader function_section;
  reader.ReadSection(kWasmSectionFunction, &function_section);
  EXPECT_EQ(function_section.ReadU32Leb(), 1u);
  EXPECT_EQ(function_section.ReadU32Leb(), 0u);
  function_section.ExpectConsumed();

  WasmReader export_section;
  reader.ReadSection(kWasmSectionExport, &export_section);
  EXPECT_EQ(export_section.ReadU32Leb(), 1u);
  EXPECT_EQ(export_section.ReadName(), "arith");
  export_section.ExpectU8(kWasmExportFunction);
  EXPECT_EQ(export_section.ReadU32Leb(), 0u);
  export_section.ExpectConsumed();

  WasmReader code_section;
  reader.ReadSection(kWasmSectionCode, &code_section);
  const std::vector<uint8_t> body = code_section.ReadCodeBodyBytes();
  code_section.ExpectConsumed();
  reader.ExpectConsumed();

  ASSERT_FALSE(body.empty());
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeI32Const));
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeI32Add));
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeI32Sub));
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeReturn));
  EXPECT_EQ(body.back(), kWasmOpcodeEnd);
}

TEST_F(WasmLowerTest, LowersSemanticSimdFunctionAndEmitsWasmModule) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @wasm_target preset(\"wasm-simd128\")\n"
      "func.def target(@wasm_target) @simd(%lhs: vector<4xi32>, %rhs: "
      "vector<4xi32>) -> "
      "(vector<4xi32>) {\n"
      "  %sum = vector.addi %lhs, %rhs : vector<4xi32>\n"
      "  %product = vector.muli %sum, %rhs : vector<4xi32>\n"
      "  func.return %product : vector<4xi32>\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  ASSERT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());
  VerifyLowModule(module.get());

  loom_low_packetization_t packetization = {};
  Packetize(module.get(), lower_result.low_func_op, &packetization);

  WasmModuleOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_single_function_module(
      &packetization.schedule, &packetization.allocation, IREE_SV("simd"),
      iree_allocator_system(), &owner.module));

  EXPECT_GT(owner.module.data_length, 8u);
  EXPECT_FALSE(owner.module.has_memory);

  WasmReader reader(owner.module.data, owner.module.data_length);
  ExpectWasmHeader(&reader);

  WasmReader type_section;
  reader.ReadSection(kWasmSectionType, &type_section);
  EXPECT_EQ(type_section.ReadU32Leb(), 1u);
  type_section.ExpectU8(kWasmFunctionType);
  EXPECT_EQ(type_section.ReadU32Leb(), 2u);
  type_section.ExpectU8(kWasmTypeV128);
  type_section.ExpectU8(kWasmTypeV128);
  EXPECT_EQ(type_section.ReadU32Leb(), 1u);
  type_section.ExpectU8(kWasmTypeV128);
  type_section.ExpectConsumed();

  WasmReader function_section;
  reader.ReadSection(kWasmSectionFunction, &function_section);
  EXPECT_EQ(function_section.ReadU32Leb(), 1u);
  EXPECT_EQ(function_section.ReadU32Leb(), 0u);
  function_section.ExpectConsumed();

  WasmReader export_section;
  reader.ReadSection(kWasmSectionExport, &export_section);
  EXPECT_EQ(export_section.ReadU32Leb(), 1u);
  EXPECT_EQ(export_section.ReadName(), "simd");
  export_section.ExpectU8(kWasmExportFunction);
  EXPECT_EQ(export_section.ReadU32Leb(), 0u);
  export_section.ExpectConsumed();

  WasmReader code_section;
  reader.ReadSection(kWasmSectionCode, &code_section);
  const std::vector<uint8_t> body = code_section.ReadCodeBodyBytes();
  code_section.ExpectConsumed();
  reader.ExpectConsumed();

  ASSERT_FALSE(body.empty());
  EXPECT_TRUE(ByteVectorContainsSubsequence(
      body,
      {kWasmOpcodeSimdPrefix, kWasmSimdI32x4AddLeb0, kWasmSimdI32x4Leb1}));
  EXPECT_TRUE(ByteVectorContainsSubsequence(
      body,
      {kWasmOpcodeSimdPrefix, kWasmSimdI32x4MulLeb0, kWasmSimdI32x4Leb1}));
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeReturn));
  EXPECT_EQ(body.back(), kWasmOpcodeEnd);
}

TEST_F(WasmLowerTest, LowersSemanticMemoryFunctionAndEmitsWasmMemoryModule) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @wasm_target preset(\"wasm-simd128\")\n"
      "func.def target(@wasm_target) @copy_indexed(%input: buffer, %output: "
      "buffer, %i: index) {\n"
      "  %zero = index.constant 0 : offset\n"
      "  %input_view = buffer.view %input[%zero] : buffer -> view<16xi32, "
      "#dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> view<16xi32, "
      "#dense>\n"
      "  %loaded = vector.load %input_view[%i] : view<16xi32, #dense> -> "
      "vector<4xi32>\n"
      "  vector.store %loaded, %output_view[%i] : vector<4xi32>, "
      "view<16xi32, #dense>\n"
      "  func.return\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  ASSERT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());
  VerifyLowModule(module.get());

  loom_low_packetization_t packetization = {};
  Packetize(module.get(), lower_result.low_func_op, &packetization);

  WasmModuleOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_single_function_module(
      &packetization.schedule, &packetization.allocation,
      IREE_SV("copy_indexed"), iree_allocator_system(), &owner.module));

  EXPECT_GT(owner.module.data_length, 8u);
  EXPECT_TRUE(owner.module.has_memory);

  WasmReader reader(owner.module.data, owner.module.data_length);
  ExpectWasmHeader(&reader);

  WasmReader type_section;
  reader.ReadSection(kWasmSectionType, &type_section);
  EXPECT_EQ(type_section.ReadU32Leb(), 1u);
  type_section.ExpectU8(kWasmFunctionType);
  EXPECT_EQ(type_section.ReadU32Leb(), 3u);
  type_section.ExpectU8(kWasmTypeI32);
  type_section.ExpectU8(kWasmTypeI32);
  type_section.ExpectU8(kWasmTypeI32);
  EXPECT_EQ(type_section.ReadU32Leb(), 0u);
  type_section.ExpectConsumed();

  WasmReader function_section;
  reader.ReadSection(kWasmSectionFunction, &function_section);
  EXPECT_EQ(function_section.ReadU32Leb(), 1u);
  EXPECT_EQ(function_section.ReadU32Leb(), 0u);
  function_section.ExpectConsumed();

  WasmReader memory_section;
  reader.ReadSection(kWasmSectionMemory, &memory_section);
  EXPECT_EQ(memory_section.ReadU32Leb(), 1u);
  memory_section.ExpectU8(kWasmLimitsMinOnly);
  EXPECT_EQ(memory_section.ReadU32Leb(), 1u);
  memory_section.ExpectConsumed();

  WasmReader export_section;
  reader.ReadSection(kWasmSectionExport, &export_section);
  EXPECT_EQ(export_section.ReadU32Leb(), 1u);
  EXPECT_EQ(export_section.ReadName(), "copy_indexed");
  export_section.ExpectU8(kWasmExportFunction);
  EXPECT_EQ(export_section.ReadU32Leb(), 0u);
  export_section.ExpectConsumed();

  WasmReader code_section;
  reader.ReadSection(kWasmSectionCode, &code_section);
  const std::vector<uint8_t> body = code_section.ReadCodeBodyBytes();
  code_section.ExpectConsumed();
  reader.ExpectConsumed();

  ASSERT_FALSE(body.empty());
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeI32Const));
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeI32Mul));
  EXPECT_TRUE(ByteVectorContains(body, kWasmOpcodeI32Add));
  EXPECT_TRUE(ByteVectorContainsSubsequence(
      body, {kWasmOpcodeSimdPrefix, kWasmSimdV128Load, 4, 0}));
  EXPECT_TRUE(ByteVectorContainsSubsequence(
      body, {kWasmOpcodeSimdPrefix, kWasmSimdV128Store, 4, 0}));
  EXPECT_EQ(body.back(), kWasmOpcodeEnd);
}

TEST_F(WasmLowerTest, UnsupportedVectorShapeEmitsDiagnosticAndNoLowFunction) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @wasm_target preset(\"wasm-simd128\")\n"
      "func.def target(@wasm_target) @wide(%lhs: vector<8xi32>, %rhs: "
      "vector<8xi32>) -> "
      "(vector<8xi32>) {\n"
      "  %sum = vector.addi %lhs, %rhs : vector<8xi32>\n"
      "  func.return %sum : vector<8xi32>\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_GT(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_EQ(lower_result.low_func_op, nullptr);
  EXPECT_FALSE(loom_symbol_ref_is_valid(lower_result.low_func_ref));
  ASSERT_FALSE(lower_collector.emissions.empty());
  bool saw_type_rejection = false;
  for (const CollectedEmission& emission : lower_collector.emissions) {
    EXPECT_EQ(emission.error,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1));
    ASSERT_EQ(emission.string_params.size(), 7u);
    EXPECT_EQ(emission.string_params[0], "wasm_target");
    EXPECT_EQ(emission.string_params[3], "wide");
    if (emission.string_params[4] == "type" &&
        emission.string_params[6].find("vector<4xi32>") != std::string::npos) {
      saw_type_rejection = true;
    }
  }
  EXPECT_TRUE(saw_type_rejection);

  loom_string_id_t low_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("wide__low"),
                                           &low_name_id));
  EXPECT_EQ(loom_module_find_symbol(module.get(), low_name_id),
            LOOM_SYMBOL_ID_INVALID);
}

TEST_F(WasmLowerTest, UnsupportedSourceOpEmitsDiagnosticAndNoLowFunction) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @wasm_target preset(\"wasm-simd128\")\n"
      "func.def target(@wasm_target) @mul(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %product = scalar.muli %lhs, %rhs : i32\n"
      "  func.return %product : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 1u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_EQ(lower_result.low_func_op, nullptr);
  EXPECT_FALSE(loom_symbol_ref_is_valid(lower_result.low_func_ref));
  ASSERT_EQ(lower_collector.emissions.size(), 1u);
  const CollectedEmission& emission = lower_collector.emissions[0];
  EXPECT_EQ(emission.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1));
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[0], "wasm_target");
  EXPECT_EQ(emission.string_params[3], "mul");
  EXPECT_EQ(emission.string_params[4], "op");
  EXPECT_EQ(emission.string_params[5], "scalar.muli");
  EXPECT_NE(emission.string_params[6].find("descriptor mapping"),
            std::string::npos);

  loom_string_id_t low_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("mul__low"),
                                           &low_name_id));
  EXPECT_EQ(loom_module_find_symbol(module.get(), low_name_id),
            LOOM_SYMBOL_ID_INVALID);
}

}  // namespace
}  // namespace loom
