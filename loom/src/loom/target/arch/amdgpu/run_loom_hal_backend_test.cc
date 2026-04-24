// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

void RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                     DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  IREE_ASSERT_OK(loom_context_register_dialect(context, dialect_id, vtables,
                                               (uint16_t)count));
}

iree_status_t ParseCurrentTargetB128AddModule(loom_context_t* context,
                                              iree_arena_block_pool_t* pool,
                                              loom_module_t** out_module) {
  static const char kSource[] =
      "target.profile @gfx_target preset(\"amdgpu-current\")\n"
      "func.def target(@gfx_target) @loom_kernel(%lhs: buffer, %rhs: buffer, "
      "%output: buffer) {\n"
      "  %tid = kernel.workitem.id<x> : index\n"
      "  %zero = index.constant 0 : offset\n"
      "  %lhs_view = buffer.view %lhs[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %rhs_view = buffer.view %rhs[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %lhs_loaded = vector.load %lhs_view[%tid, 0] : "
      "view<64x4xi32, #dense> -> vector<4xi32>\n"
      "  %rhs_loaded = vector.load %rhs_view[%tid, 0] : "
      "view<64x4xi32, #dense> -> vector<4xi32>\n"
      "  %sum = vector.addi %lhs_loaded, %rhs_loaded : vector<4xi32>\n"
      "  vector.store %sum, %output_view[%tid, 0] : vector<4xi32>, "
      "view<64x4xi32, #dense>\n"
      "  func.return\n"
      "}\n";
  loom_text_parse_options_t options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_amdgpu_hal_backend.loom"), context,
                         pool, &options, out_module);
}

TEST(AmdgpuRunLoomHalBackendTest, PreservesDetailedReportRows) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  loom_context_t context = {};
  loom_context_initialize(iree_allocator_system(), &context);
  RegisterDialect(&context, LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_ENCODING,
                  loom_encoding_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
  RegisterDialect(&context, LOOM_DIALECT_LOW, loom_low_dialect_vtables);
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      ParseCurrentTargetB128AddModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1100"), &processor));
  ASSERT_NE(processor, nullptr);

  loom_target_compile_report_pressure_row_t pressure_rows[4] = {};
  loom_target_compile_report_spill_row_t spill_rows[4] = {};
  loom_target_compile_report_source_low_row_t source_low_rows[8] = {};
  const loom_target_compile_report_row_storage_t row_storage = {
      .pressure_rows = pressure_rows,
      .pressure_row_capacity = IREE_ARRAYSIZE(pressure_rows),
      .spill_rows = spill_rows,
      .spill_row_capacity = IREE_ARRAYSIZE(spill_rows),
      .source_low_rows = source_low_rows,
      .source_low_row_capacity = IREE_ARRAYSIZE(source_low_rows),
  };

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report);
  loom_target_compile_report_set_row_storage(&report, &row_storage);

  loom_run_hal_selected_target_t target = {
      .data = processor,
      .preset_key = processor->low_preset_key,
  };
  loom_run_hal_executable_t executable = {};
  IREE_ASSERT_OK(iree_run_loom_amdgpu_hal_backend.compile(
      &iree_run_loom_amdgpu_hal_backend, module, &target,
      /*entry_symbol=*/iree_string_view_empty(),
      /*diagnostic_sink=*/(loom_diagnostic_sink_t){0},
      /*source_resolver=*/(loom_source_resolver_t){0}, /*max_errors=*/20,
      &report, iree_allocator_system(), &executable));

  EXPECT_EQ(report.source_low_rows, source_low_rows);
  EXPECT_GT(report.source_low_row_total_count, 0u);
  EXPECT_GT(report.source_low_row_count, 0u);
  bool found_emitting_source_row = false;
  for (iree_host_size_t i = 0; i < report.source_low_row_count; ++i) {
    if (report.source_low_rows[i].emitted_low_op_count != 0) {
      found_emitting_source_row = true;
      break;
    }
  }
  EXPECT_TRUE(found_emitting_source_row);
  EXPECT_EQ(report.pressure_rows, pressure_rows);
  EXPECT_GT(report.pressure_row_total_count, 0u);
  EXPECT_GT(report.pressure_row_count, 0u);
  EXPECT_EQ(report.spill_rows, spill_rows);

  iree_run_loom_amdgpu_hal_backend.deinitialize_executable(
      &iree_run_loom_amdgpu_hal_backend, &executable, iree_allocator_system());
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
