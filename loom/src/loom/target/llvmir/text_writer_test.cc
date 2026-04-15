// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/text_writer.h"

#include <cctype>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/llvmir/test_modules.h"
#include "loom/target/llvmir/verify.h"

namespace loom {
namespace {

iree_status_t EmitText(loom_llvmir_module_t* module, std::string* out_text) {
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_module(module));
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_llvmir_text_write_module(module, &stream);
  if (iree_status_is_ok(status)) {
    iree_string_view_t view = iree_string_builder_view(&builder);
    out_text->assign(view.data, view.size);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ScenarioTestName(
    const testing::TestParamInfo<loom_llvmir_test_module_scenario_t>& info) {
  std::string name =
      ToString(loom_llvmir_test_module_scenario_name(info.param));
  for (char& character : name) {
    if (!std::isalnum(static_cast<unsigned char>(character))) {
      character = '_';
    }
  }
  return name;
}

class LlvmIrTextWriterTest
    : public testing::TestWithParam<loom_llvmir_test_module_scenario_t> {};

TEST_P(LlvmIrTextWriterTest, EmitsExpectedText) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_test_module_build(
      GetParam(), iree_allocator_system(), &module));

  std::string text;
  iree_status_t status = EmitText(module, &text);
  loom_llvmir_module_free(module);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(text, ToString(loom_llvmir_test_module_expected_text(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(
    All, LlvmIrTextWriterTest,
    testing::Values(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4,
                    LOOM_LLVMIR_TEST_MODULE_CFG_PHI,
                    LOOM_LLVMIR_TEST_MODULE_INLINE_ASM,
                    LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS),
    ScenarioTestName);

}  // namespace
}  // namespace loom
