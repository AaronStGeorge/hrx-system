// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/function_body.h"

#include <cstdint>
#include <string>
#include <vector>

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
#include "loom/target/arch/wasm/low_registry.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

constexpr uint8_t kWasmOpcodeEnd = 0x0B;
constexpr uint8_t kWasmOpcodeReturn = 0x0F;
constexpr uint8_t kWasmOpcodeLocalGet = 0x20;
constexpr uint8_t kWasmOpcodeLocalSet = 0x21;
constexpr uint8_t kWasmOpcodeI32Const = 0x41;
constexpr uint8_t kWasmOpcodeI32Add = 0x6A;
constexpr uint8_t kWasmOpcodeSimdPrefix = 0xFD;
constexpr uint8_t kWasmTypeI32 = 0x7F;
constexpr uint8_t kWasmTypeV128 = 0x7B;

constexpr uint32_t kWasmSimdV128Load = 0x00;
constexpr uint32_t kWasmSimdV128Store = 0x0B;
constexpr uint32_t kWasmSimdV128Const = 0x0C;
constexpr uint32_t kWasmSimdI8x16Shuffle = 0x0D;
constexpr uint32_t kWasmSimdI32x4ExtractLane = 0x1B;
constexpr uint32_t kWasmSimdI32x4ReplaceLane = 0x1C;
constexpr uint32_t kWasmSimdI32x4Add = 0xAE;
constexpr uint32_t kWasmSimdI32x4Mul = 0xB5;

struct FunctionBodyOwner {
  ~FunctionBodyOwner() {
    loom_wasm_function_body_deinitialize(&body, iree_allocator_system());
  }

  loom_wasm_function_body_t body = {};
};

class WasmReader {
 public:
  explicit WasmReader(const loom_wasm_function_body_t& body)
      : data_(body.data), length_(body.data_length) {}

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

  int32_t ReadI32Leb() {
    int32_t value = 0;
    uint32_t shift = 0;
    uint8_t byte = 0;
    do {
      byte = ReadU8();
      value |= static_cast<int32_t>(byte & 0x7F) << shift;
      shift += 7;
    } while ((byte & 0x80) != 0);
    if (shift < 32 && (byte & 0x40) != 0) {
      value |= static_cast<int32_t>(~0u << shift);
    }
    return value;
  }

  uint64_t ReadU64Le() {
    uint64_t value = 0;
    for (uint32_t i = 0; i < 8; ++i) {
      value |= static_cast<uint64_t>(ReadU8()) << (i * 8);
    }
    return value;
  }

  void ExpectU8(uint8_t expected) { EXPECT_EQ(ReadU8(), expected); }

  uint32_t ExpectLocalGet() {
    ExpectU8(kWasmOpcodeLocalGet);
    return ReadU32Leb();
  }

  uint32_t ExpectLocalSet() {
    ExpectU8(kWasmOpcodeLocalSet);
    return ReadU32Leb();
  }

  void ExpectSimd(uint32_t expected_subopcode) {
    ExpectU8(kWasmOpcodeSimdPrefix);
    EXPECT_EQ(ReadU32Leb(), expected_subopcode);
  }

  void ExpectConsumed() { EXPECT_EQ(position_, length_); }

 private:
  const uint8_t* data_ = nullptr;
  iree_host_size_t length_ = 0;
  iree_host_size_t position_ = 0;
};

static void ReassignFirstValue(
    std::vector<loom_low_allocation_assignment_t>* assignments,
    loom_low_allocation_location_kind_t location_kind) {
  ASSERT_FALSE(assignments->empty());
  assignments->front().location_kind = location_kind;
}

class WasmFunctionBodyTest : public ::testing::Test {
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
        iree_make_cstring_view(source), IREE_SV("wasm_function_body_test.loom"),
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
        "target.profile @wasm_target preset(\"wasm-simd128\")\n";
    source += body;
    module_ = ParseSource(source.c_str());
    ASSERT_NE(module_, nullptr);

    loom_op_t* low_func = FindFirstLowFunction(module_);
    ASSERT_NE(low_func, nullptr);
    static const loom_low_descriptor_set_t* descriptor_sets[] = {
        loom_wasm_core_simd128_descriptor_set(),
    };
    static const loom_target_bundle_t* target_bundles[] = {
        &loom_wasm_low_target_bundle_core_simd128,
    };
    registry_ = loom_low_descriptor_registry_t{
        .descriptor_sets = descriptor_sets,
        .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
        .target_bundles = target_bundles,
        .target_bundle_count = IREE_ARRAYSIZE(target_bundles),
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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_low_descriptor_registry_t registry_ = {};
  iree_arena_allocator_t sidecar_arena_ = {};
  bool sidecar_arena_initialized_ = false;
};

TEST_F(WasmFunctionBodyTest, EmitsStraightLineI32FunctionBody) {
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

  FunctionBodyOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_function_body(
      &schedule, &allocation, iree_allocator_system(), &owner.body));

  EXPECT_EQ(owner.body.parameter_count, 1u);
  EXPECT_GE(owner.body.local_count, 1u);
  EXPECT_GT(owner.body.data_length, owner.body.body_length);

  WasmReader reader(owner.body);
  EXPECT_EQ(reader.ReadU32Leb(), owner.body.body_length);
  const uint32_t local_declaration_count = reader.ReadU32Leb();
  for (uint32_t i = 0; i < local_declaration_count; ++i) {
    EXPECT_GT(reader.ReadU32Leb(), 0u);
    reader.ExpectU8(kWasmTypeI32);
  }
  reader.ExpectU8(kWasmOpcodeI32Const);
  EXPECT_EQ(reader.ReadI32Leb(), 1);
  const uint32_t one = reader.ExpectLocalSet();
  EXPECT_EQ(reader.ExpectLocalGet(), 0u);
  EXPECT_EQ(reader.ExpectLocalGet(), one);
  reader.ExpectU8(kWasmOpcodeI32Add);
  const uint32_t sum = reader.ExpectLocalSet();
  EXPECT_EQ(reader.ExpectLocalGet(), sum);
  reader.ExpectU8(kWasmOpcodeReturn);
  reader.ExpectU8(kWasmOpcodeEnd);
  reader.ExpectConsumed();
}

TEST_F(WasmFunctionBodyTest, EmitsV128ConstFunctionBody) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@wasm_target) @wasm_test() -> (reg<wasm.v128>) {\n"
      "  %bits = low.const<wasm.v128.const> {lo64 = 72623859790382856, "
      "hi64 = 1230066625199609624} : reg<wasm.v128>\n"
      "  low.return %bits : reg<wasm.v128>\n"
      "}\n",
      &schedule, &allocation);

  FunctionBodyOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_function_body(
      &schedule, &allocation, iree_allocator_system(), &owner.body));

  EXPECT_EQ(owner.body.parameter_count, 0u);
  EXPECT_GE(owner.body.local_count, 1u);

  WasmReader reader(owner.body);
  EXPECT_EQ(reader.ReadU32Leb(), owner.body.body_length);
  EXPECT_EQ(reader.ReadU32Leb(), 1u);
  EXPECT_EQ(reader.ReadU32Leb(), 1u);
  reader.ExpectU8(kWasmTypeV128);

  reader.ExpectSimd(kWasmSimdV128Const);
  EXPECT_EQ(reader.ReadU64Le(), UINT64_C(0x0102030405060708));
  EXPECT_EQ(reader.ReadU64Le(), UINT64_C(0x1112131415161718));
  const uint32_t bits = reader.ExpectLocalSet();
  EXPECT_EQ(reader.ExpectLocalGet(), bits);
  reader.ExpectU8(kWasmOpcodeReturn);
  reader.ExpectU8(kWasmOpcodeEnd);
  reader.ExpectConsumed();
}

TEST_F(WasmFunctionBodyTest, EmitsStraightLineSimdFunctionBody) {
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

  FunctionBodyOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_function_body(
      &schedule, &allocation, iree_allocator_system(), &owner.body));

  EXPECT_EQ(owner.body.parameter_count, 3u);
  EXPECT_GE(owner.body.local_count, 3u);

  WasmReader reader(owner.body);
  EXPECT_EQ(reader.ReadU32Leb(), owner.body.body_length);
  const uint32_t local_declaration_count = reader.ReadU32Leb();
  for (uint32_t i = 0; i < local_declaration_count; ++i) {
    EXPECT_GT(reader.ReadU32Leb(), 0u);
    reader.ExpectU8(kWasmTypeV128);
  }

  EXPECT_EQ(reader.ExpectLocalGet(), 0u);
  reader.ExpectSimd(kWasmSimdV128Load);
  EXPECT_EQ(reader.ReadU32Leb(), 4u);
  EXPECT_EQ(reader.ReadU32Leb(), 0u);
  const uint32_t loaded = reader.ExpectLocalSet();

  EXPECT_EQ(reader.ExpectLocalGet(), loaded);
  EXPECT_EQ(reader.ExpectLocalGet(), 1u);
  reader.ExpectSimd(kWasmSimdI32x4Add);
  const uint32_t sum = reader.ExpectLocalSet();

  EXPECT_EQ(reader.ExpectLocalGet(), 0u);
  EXPECT_EQ(reader.ExpectLocalGet(), sum);
  reader.ExpectSimd(kWasmSimdV128Store);
  EXPECT_EQ(reader.ReadU32Leb(), 4u);
  EXPECT_EQ(reader.ReadU32Leb(), 0u);

  EXPECT_EQ(reader.ExpectLocalGet(), sum);
  EXPECT_EQ(reader.ExpectLocalGet(), 2u);
  reader.ExpectSimd(kWasmSimdI32x4Mul);
  const uint32_t out = reader.ExpectLocalSet();

  EXPECT_EQ(reader.ExpectLocalGet(), out);
  reader.ExpectU8(kWasmOpcodeReturn);
  reader.ExpectU8(kWasmOpcodeEnd);
  reader.ExpectConsumed();
}

TEST_F(WasmFunctionBodyTest, EmitsSimdLaneFunctionBody) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@wasm_target) @wasm_test(%v : reg<wasm.v128>, "
      "%x : reg<wasm.i32>) -> (reg<wasm.v128>) {\n"
      "  %lane = low.op<wasm.i32x4.extract_lane>(%v) {lane = 1} : "
      "(reg<wasm.v128>) -> reg<wasm.i32>\n"
      "  %sum = low.op<wasm.i32.add>(%lane, %x) : "
      "(reg<wasm.i32>, reg<wasm.i32>) -> reg<wasm.i32>\n"
      "  %updated = low.op<wasm.i32x4.replace_lane>(%v, %sum) {lane = 2} : "
      "(reg<wasm.v128>, reg<wasm.i32>) -> reg<wasm.v128>\n"
      "  %shuffled = low.op<wasm.i8x16.shuffle>(%updated, %updated) "
      "{lane0 = 12, lane1 = 13, lane2 = 14, lane3 = 15, lane4 = 8, "
      "lane5 = 9, lane6 = 10, lane7 = 11, lane8 = 4, lane9 = 5, "
      "lane10 = 6, lane11 = 7, lane12 = 0, lane13 = 1, lane14 = 2, "
      "lane15 = 31} : (reg<wasm.v128>, reg<wasm.v128>) -> reg<wasm.v128>\n"
      "  low.return %shuffled : reg<wasm.v128>\n"
      "}\n",
      &schedule, &allocation);

  FunctionBodyOwner owner;
  IREE_ASSERT_OK(loom_wasm_emit_function_body(
      &schedule, &allocation, iree_allocator_system(), &owner.body));

  EXPECT_EQ(owner.body.parameter_count, 2u);
  EXPECT_GE(owner.body.local_count, 2u);

  WasmReader reader(owner.body);
  EXPECT_EQ(reader.ReadU32Leb(), owner.body.body_length);
  const uint32_t local_declaration_count = reader.ReadU32Leb();
  for (uint32_t i = 0; i < local_declaration_count; ++i) {
    EXPECT_GT(reader.ReadU32Leb(), 0u);
    const uint8_t type = reader.ReadU8();
    EXPECT_TRUE(type == kWasmTypeI32 || type == kWasmTypeV128);
  }

  EXPECT_EQ(reader.ExpectLocalGet(), 0u);
  reader.ExpectSimd(kWasmSimdI32x4ExtractLane);
  EXPECT_EQ(reader.ReadU8(), 1u);
  const uint32_t lane = reader.ExpectLocalSet();

  EXPECT_EQ(reader.ExpectLocalGet(), lane);
  EXPECT_EQ(reader.ExpectLocalGet(), 1u);
  reader.ExpectU8(kWasmOpcodeI32Add);
  const uint32_t sum = reader.ExpectLocalSet();

  EXPECT_EQ(reader.ExpectLocalGet(), 0u);
  EXPECT_EQ(reader.ExpectLocalGet(), sum);
  reader.ExpectSimd(kWasmSimdI32x4ReplaceLane);
  EXPECT_EQ(reader.ReadU8(), 2u);
  const uint32_t updated = reader.ExpectLocalSet();

  EXPECT_EQ(reader.ExpectLocalGet(), updated);
  EXPECT_EQ(reader.ExpectLocalGet(), updated);
  reader.ExpectSimd(kWasmSimdI8x16Shuffle);
  const uint8_t expected_lanes[16] = {
      12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 31,
  };
  for (uint8_t expected_lane : expected_lanes) {
    EXPECT_EQ(reader.ReadU8(), expected_lane);
  }
  const uint32_t shuffled = reader.ExpectLocalSet();

  EXPECT_EQ(reader.ExpectLocalGet(), shuffled);
  reader.ExpectU8(kWasmOpcodeReturn);
  reader.ExpectU8(kWasmOpcodeEnd);
  reader.ExpectConsumed();
}

TEST_F(WasmFunctionBodyTest, RejectsSpilledAllocation) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@wasm_target) @wasm_test(%input : reg<wasm.i32>) "
      "-> (reg<wasm.i32>) {\n"
      "  low.return %input : reg<wasm.i32>\n"
      "}\n",
      &schedule, &allocation);

  std::vector<loom_low_allocation_assignment_t> assignments(
      allocation.assignments,
      allocation.assignments + allocation.assignment_count);
  ReassignFirstValue(&assignments, LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT);
  loom_low_allocation_sidecar_t spilled_allocation = allocation;
  spilled_allocation.assignments = assignments.data();

  FunctionBodyOwner owner;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_wasm_emit_function_body(&schedule, &spilled_allocation,
                                   iree_allocator_system(), &owner.body));
}

}  // namespace
}  // namespace loom
