// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::iree::StatusCode;
using ModulePtr = ::loom::testing::ModulePtr;

static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

struct CapturedAllocationDiagnostic {
  // Number of capacity diagnostics observed by the capture callback.
  uint32_t count = 0;
  // Error domain of the most recently captured diagnostic.
  loom_error_domain_t domain = LOOM_ERROR_DOMAIN_COUNT_;
  // Error code of the most recently captured diagnostic.
  uint16_t code = 0;
  // Kind of allocator input that exceeded capacity.
  std::string subject_kind;
  // Register class reported by the capacity diagnostic.
  std::string register_class;
  // First requested location in the reported range.
  uint32_t location_base = UINT32_MAX;
  // Number of requested units in the reported range.
  uint32_t location_count = UINT32_MAX;
  // Exclusive end location in the reported range.
  uint64_t location_end = UINT64_MAX;
  // Effective allocation capacity for the reported register class.
  uint32_t allocation_capacity = UINT32_MAX;
};

iree_status_t CaptureAllocationDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  CapturedAllocationDiagnostic* captured =
      static_cast<CapturedAllocationDiagnostic*>(user_data);
  ++captured->count;
  captured->domain = emission->error->domain;
  captured->code = emission->error->code;
  if (emission->param_count < 10u) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation diagnostic missing capacity params");
  }
  captured->subject_kind = ToString(emission->params[4].string);
  captured->register_class = ToString(emission->params[5].string);
  captured->location_base = emission->params[6].u32;
  captured->location_count = emission->params[7].u32;
  captured->location_end = emission->params[8].u64;
  captured->allocation_capacity = emission->params[9].u32;
  return iree_ok_status();
}

class LowAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    descriptor_registry_ = {
        .descriptor_set_providers = kDescriptorSetProviders,
        .descriptor_set_provider_count =
            IREE_ARRAYSIZE(kDescriptorSetProviders),
    };
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry_, &options.low_asm_environment);
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("allocation_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_op_t* FindLowFunction(loom_module_t* module, iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(loom_low_func_def_isa(op));
    return op;
  }

  iree_status_t Allocate(
      loom_module_t* module, loom_op_t* function_op,
      const loom_low_allocation_reserved_range_t* reserved_ranges,
      iree_host_size_t reserved_range_count,
      CapturedAllocationDiagnostic* captured_diagnostic) {
    const iree_diagnostic_emitter_t emitter = {
        .fn = CaptureAllocationDiagnostic,
        .user_data = captured_diagnostic,
    };
    loom_low_allocation_options_t options = {
        .descriptor_registry = &descriptor_registry_,
        .reserved_ranges = reserved_ranges,
        .reserved_range_count = reserved_range_count,
        .emitter = emitter,
    };
    loom_low_allocation_table_t table = {};
    return loom_low_allocate_function(module, function_op, &options,
                                      &analysis_arena_, &table);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_low_descriptor_registry_t descriptor_registry_;
};

static const char kSinglePhysFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @single(%value: reg<test.phys>) -> (reg<test.phys>) {
  low.return %value : reg<test.phys>
}
)";

TEST_F(LowAllocationTest, EmitsDiagnosticForReservedRangeBeyondCapacity) {
  ModulePtr module = ParseModule(kSinglePhysFunction);
  loom_op_t* function_op = FindLowFunction(module.get(), IREE_SV("single"));
  const loom_low_allocation_reserved_range_t reserved_range = {
      .register_class = IREE_SV("test.phys"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 32,
      .location_count = 1,
  };

  CapturedAllocationDiagnostic captured;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      Allocate(module.get(), function_op, &reserved_range, 1, &captured));

  EXPECT_EQ(captured.count, 1u);
  EXPECT_EQ(captured.domain, LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(captured.code, 22u);
  EXPECT_EQ(captured.subject_kind, "reserved range");
  EXPECT_EQ(captured.register_class, "test.phys");
  EXPECT_EQ(captured.location_base, 32u);
  EXPECT_EQ(captured.location_count, 1u);
  EXPECT_EQ(captured.location_end, 33u);
  EXPECT_EQ(captured.allocation_capacity, 32u);
}

}  // namespace
}  // namespace loom
