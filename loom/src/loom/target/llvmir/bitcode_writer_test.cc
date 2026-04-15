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
#include "loom/target/llvmir/builder.h"
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
using ModulePtr =
    std::unique_ptr<loom_llvmir_module_t, void (*)(loom_llvmir_module_t*)>;

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

void ExpectBitcodeMagic(const std::string& bytes) {
  ASSERT_GE(bytes.size(), 8u);
  EXPECT_EQ((uint8_t)bytes[0], LOOM_LLVMIR_BITCODE_MAGIC_0);
  EXPECT_EQ((uint8_t)bytes[1], LOOM_LLVMIR_BITCODE_MAGIC_1);
  EXPECT_EQ((uint8_t)bytes[2], LOOM_LLVMIR_BITCODE_MAGIC_2);
  EXPECT_EQ((uint8_t)bytes[3], LOOM_LLVMIR_BITCODE_MAGIC_3);
}

void ExpectWritesBitcode(loom_llvmir_module_t* module) {
  StreamPtr stream = CreateStream();
  IREE_EXPECT_OK(loom_llvmir_bitcode_write_module(module, stream.get()));
  ExpectBitcodeMagic(StreamBytes(stream.get()));
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
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t v4f32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  IREE_ASSERT_OK(
      loom_llvmir_module_get_integer_type(module_ptr.get(), 32, &i32_type));
  IREE_ASSERT_OK(loom_llvmir_module_get_pointer_type(
      module_ptr.get(), target_env->address_spaces.generic, &ptr_type));
  IREE_ASSERT_OK(loom_llvmir_module_get_float_type(
      module_ptr.get(), LOOM_LLVMIR_FLOAT_F32, &f32_type));
  IREE_ASSERT_OK(loom_llvmir_module_get_vector_type(module_ptr.get(), 4,
                                                    f32_type, &v4f32_type));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  ExpectWritesBitcode(module_ptr.get());
}

TEST(LlvmIrBitcodeWriterTest, WritesFunctionDeclarationsAndRetVoidBodies) {
  const loom_llvmir_target_env_t* target_env =
      loom_llvmir_target_env_x86_64_unknown_linux_gnu();
  loom_llvmir_target_config_t target_config = {};
  IREE_ASSERT_OK(loom_llvmir_target_env_module_config(
      target_env, IREE_SV("loom-functions"), &target_config));

  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_allocate(&target_config,
                                             iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  IREE_ASSERT_OK(loom_llvmir_module_get_pointer_type(
      module_ptr.get(), target_env->address_spaces.generic, &ptr_type));

  loom_llvmir_function_desc_t import_desc = {};
  import_desc.kind = LOOM_LLVMIR_FUNCTION_DECLARATION;
  import_desc.name = IREE_SV("imported");
  import_desc.return_type = void_type;
  import_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_function_t* imported = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(), &import_desc,
                                                 &imported));
  loom_llvmir_parameter_desc_t parameter_desc = {};
  parameter_desc.type_id = ptr_type;
  parameter_desc.name = IREE_SV("buffer");
  loom_llvmir_value_id_t parameter = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_function_add_parameter(imported, &parameter_desc,
                                                    &parameter));

  loom_llvmir_function_desc_t definition_desc = {};
  definition_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  definition_desc.name = IREE_SV("entry");
  definition_desc.return_type = void_type;
  definition_desc.linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL;
  definition_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_function_t* entry = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &definition_desc, &entry));
  loom_llvmir_block_t* block = NULL;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_block(entry, IREE_SV("entry"), &block));
  IREE_ASSERT_OK(loom_llvmir_build_ret_void(block));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  ExpectWritesBitcode(module_ptr.get());
}

TEST(LlvmIrBitcodeWriterTest, WritesCfgPhiFunctionBodies) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_test_module_build(
      LOOM_LLVMIR_TEST_MODULE_CFG_PHI, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  ExpectWritesBitcode(module_ptr.get());
}

TEST(LlvmIrBitcodeWriterTest, WritesScalarBinopFunctionBodies) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_test_module_build(
      LOOM_LLVMIR_TEST_MODULE_SCALAR_BINOP, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  ExpectWritesBitcode(module_ptr.get());
}

TEST(LlvmIrBitcodeWriterTest, WritesObjectVadd4MemoryFunctionBody) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_test_module_build(
      LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  ExpectWritesBitcode(module_ptr.get());
}

TEST(LlvmIrBitcodeWriterTest, WritesCallConstantsFunctionBodies) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(
      loom_llvmir_test_module_build(LOOM_LLVMIR_TEST_MODULE_CALL_CONSTANTS,
                                    iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  ExpectWritesBitcode(module_ptr.get());
}

TEST(LlvmIrBitcodeWriterTest,
     RejectsDeclarationMetadataAttachmentsBeforeWriting) {
  const loom_llvmir_target_env_t* target_env =
      loom_llvmir_target_env_x86_64_unknown_linux_gnu();
  loom_llvmir_target_config_t target_config = {};
  IREE_ASSERT_OK(loom_llvmir_target_env_module_config(
      target_env, IREE_SV("loom-unsupported-declaration-metadata"),
      &target_config));

  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_allocate(&target_config,
                                             iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  int32_t tuple_values[] = {1, 1, 1};
  loom_llvmir_metadata_i32_tuple_t metadata = {};
  metadata.values = tuple_values;
  metadata.value_count = IREE_ARRAYSIZE(tuple_values);
  loom_llvmir_metadata_id_t metadata_id = LOOM_LLVMIR_METADATA_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_module_add_metadata_i32_tuple(
      module_ptr.get(), &metadata, &metadata_id));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DECLARATION;
  function_desc.name = IREE_SV("decl_with_metadata");
  function_desc.return_type = void_type;
  function_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_function_t* function = NULL;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));
  loom_llvmir_metadata_attachment_t attachment = {};
  attachment.name = IREE_SV("reqd_work_group_size");
  attachment.metadata_id = metadata_id;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_metadata_attachment(function, &attachment));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  StreamPtr stream = CreateStream();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_llvmir_bitcode_write_module(module_ptr.get(), stream.get()));
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);
}

class LlvmIrBitcodeWriterScenarioTest
    : public testing::TestWithParam<loom_llvmir_test_module_scenario_t> {};

TEST_P(LlvmIrBitcodeWriterScenarioTest, WritesFixtureScenario) {
  loom_llvmir_module_t* module = NULL;
  IREE_ASSERT_OK(loom_llvmir_test_module_build(
      GetParam(), iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  ExpectWritesBitcode(module_ptr.get());
}

INSTANTIATE_TEST_SUITE_P(
    All, LlvmIrBitcodeWriterScenarioTest,
    testing::Values(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4,
                    LOOM_LLVMIR_TEST_MODULE_CALL_CONSTANTS,
                    LOOM_LLVMIR_TEST_MODULE_BUILTIN_INTRINSICS,
                    LOOM_LLVMIR_TEST_MODULE_CFG_PHI,
                    LOOM_LLVMIR_TEST_MODULE_SCALAR_BINOP,
                    LOOM_LLVMIR_TEST_MODULE_INLINE_ASM,
                    LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS),
    ScenarioTestName);

}  // namespace
}  // namespace loom
