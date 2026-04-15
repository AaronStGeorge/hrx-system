// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/bitcode_writer.h"

#include <cctype>
#include <memory>
#include <string>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/llvmir/bitcode_format.h"
#include "loom/target/llvmir/target_env.h"
#include "loom/target/llvmir/test_modules.h"
#include "loom/target/llvmir/verify.h"

namespace loom {
namespace {

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

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

StreamPtr CreateStream() {
  iree_io_stream_t* stream = NULL;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      1024, iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

std::string StreamBytes(iree_io_stream_t* stream) {
  iree_io_stream_pos_t length = iree_io_stream_length(stream);
  IREE_ASSERT_GE(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(iree_io_stream_read(stream, bytes.size(), bytes.data(), NULL));
  return bytes;
}

TEST(LlvmIrBitcodeWriterTest, WritesModuleHeaderAndTypeBlock) {
  const loom_llvmir_target_env_t* target_env =
      loom_llvmir_target_env_x86_64_unknown_linux_gnu();
  loom_llvmir_target_config_t target_config = {};
  IREE_ASSERT_OK(loom_llvmir_target_env_module_config(
      target_env, IREE_SV("loom-empty"), &target_config));

  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_allocate(&target_config,
                                             iree_allocator_system(), &module));
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v4f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_module_get_void_type(module, &void_type));
  IREE_ASSERT_OK(loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_ASSERT_OK(loom_llvmir_module_get_pointer_type(
      module, target_env->address_spaces.generic, &ptr_type));
  IREE_ASSERT_OK(loom_llvmir_module_get_float_type(
      module, LOOM_LLVMIR_FLOAT_F32, &f32_type));
  IREE_ASSERT_OK(
      loom_llvmir_module_get_vector_type(module, 4, f32_type, &v4f32_type));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module));

  StreamPtr stream = CreateStream();
  IREE_EXPECT_OK(loom_llvmir_bitcode_write_module(module, stream.get()));
  std::string bytes = StreamBytes(stream.get());
  ASSERT_GE(bytes.size(), 8u);
  EXPECT_EQ((uint8_t)bytes[0], LOOM_LLVMIR_BITCODE_MAGIC_0);
  EXPECT_EQ((uint8_t)bytes[1], LOOM_LLVMIR_BITCODE_MAGIC_1);
  EXPECT_EQ((uint8_t)bytes[2], LOOM_LLVMIR_BITCODE_MAGIC_2);
  EXPECT_EQ((uint8_t)bytes[3], LOOM_LLVMIR_BITCODE_MAGIC_3);
  loom_llvmir_module_free(module);
}

class LlvmIrBitcodeWriterTest
    : public testing::TestWithParam<loom_llvmir_test_module_scenario_t> {};

TEST_P(LlvmIrBitcodeWriterTest, RejectsFunctionModulesBeforeWriting) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_test_module_build(
      GetParam(), iree_allocator_system(), &module));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module));

  StreamPtr stream = CreateStream();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_llvmir_bitcode_write_module(module, stream.get()));
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);
  loom_llvmir_module_free(module);
}

INSTANTIATE_TEST_SUITE_P(
    All, LlvmIrBitcodeWriterTest,
    testing::Values(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4,
                    LOOM_LLVMIR_TEST_MODULE_CFG_PHI,
                    LOOM_LLVMIR_TEST_MODULE_INLINE_ASM,
                    LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS),
    ScenarioTestName);

}  // namespace
}  // namespace loom
