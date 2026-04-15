// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/lower.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/encoding.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/llvmir/text_writer.h"
#include "loom/target/llvmir/tool.h"
#include "loom/target/llvmir/verify.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using LlvmModulePtr =
    std::unique_ptr<loom_llvmir_module_t, void (*)(loom_llvmir_module_t*)>;

std::string StatusToString(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = NULL;
  iree_host_size_t length = 0;
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    std::string result(buffer, length);
    iree_allocator_free(allocator, buffer);
    return result;
  }
  return std::string("status code ") +
         std::to_string(static_cast<int>(iree_status_code(status)));
}

iree_string_view_t StringView(const std::string& value) {
  return iree_make_string_view(value.data(), value.size());
}

bool IsToolUnavailable(iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);
  return code == IREE_STATUS_NOT_FOUND || code == IREE_STATUS_UNAVAILABLE ||
         code == IREE_STATUS_UNIMPLEMENTED;
}

const char* TempDirectory() {
  const char* temp_directory = getenv("TEST_TMPDIR");
  if (temp_directory != NULL && temp_directory[0] != '\0') {
    return temp_directory;
  }
  temp_directory = getenv("TMPDIR");
  if (temp_directory != NULL && temp_directory[0] != '\0') {
    return temp_directory;
  }
#if defined(IREE_PLATFORM_WINDOWS)
  temp_directory = getenv("TEMP");
  if (temp_directory != NULL && temp_directory[0] != '\0') {
    return temp_directory;
  }
  return "C:/Temp";
#else
  return "/tmp";
#endif
}

std::string TempPath(const char* suffix) {
  static uint32_t counter = 0;
  return std::string(TempDirectory()) + "/loom_llvmir_lower_test_" +
         std::to_string(counter++) + suffix;
}

class TempFile {
 public:
  explicit TempFile(std::string path) : path_(std::move(path)) {}
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  ~TempFile() { std::remove(path_.c_str()); }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

iree_status_t WriteTempFile(const std::string& path,
                            const std::string& contents) {
  return iree_io_file_contents_write(
      StringView(path),
      iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system());
}

class LlvmIrLowerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("lower_test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  typedef const loom_op_vtable_t* const* (*DialectVtablesFn)(
      iree_host_size_t* out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_symbol_ref_t MakeSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t ref;
    ref.module_id = 0;
    ref.symbol_id = symbol_id;
    return ref;
  }

  void SetValueName(loom_value_id_t value_id, iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    module_->values.entries[value_id].name_id = name_id;
  }

  uint16_t AddDenseEncoding() {
    loom_string_id_t dense_name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("dense"), &dense_name_id));
    loom_encoding_t encoding;
    memset(&encoding, 0, sizeof(encoding));
    encoding.name_id = dense_name_id;
    uint16_t encoding_id = 0;
    IREE_CHECK_OK(loom_module_add_encoding(module_, &encoding, &encoding_id));
    return encoding_id;
  }

  loom_builder_t BodyBuilder(loom_op_t* func_op) {
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    loom_region_t* body = loom_func_like_body(func);
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body), &builder);
    return builder;
  }

  std::string LowerToText() {
    loom_llvmir_lowering_options_t options;
    options.target_profile = loom_llvmir_target_profile_x86_64_object();
    options.source_name = IREE_SV("lower_test");
    loom_llvmir_module_t* lowered = NULL;
    IREE_CHECK_OK(loom_llvmir_lower_module(module_, &options,
                                           iree_allocator_system(), &lowered));
    LlvmModulePtr lowered_ptr(lowered, loom_llvmir_module_free);
    IREE_CHECK_OK(loom_llvmir_verify_module(lowered_ptr.get()));

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_CHECK_OK(loom_llvmir_text_write_module(lowered_ptr.get(), &stream));
    std::string text(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return text;
  }

  void VerifyTextWithLlvmTools(const std::string& text) {
    TempFile input_file(TempPath(".ll"));
    TempFile bitcode_file(TempPath(".bc"));
    IREE_ASSERT_OK(WriteTempFile(input_file.path(), text));

    loom_llvmir_toolchain_t toolchain;
    loom_llvmir_toolchain_initialize_from_environment(&toolchain);
    iree_status_t status = loom_llvmir_tool_assemble_text_file(
        &toolchain, StringView(input_file.path()),
        StringView(bitcode_file.path()), iree_allocator_system());
    if (IsToolUnavailable(status)) {
      std::string message = StatusToString(status);
      iree_status_ignore(status);
      GTEST_SKIP() << message;
    }
    IREE_ASSERT_OK(status);

    status = loom_llvmir_tool_verify_bitcode_file(
        &toolchain, StringView(bitcode_file.path()), iree_allocator_system());
    if (IsToolUnavailable(status)) {
      std::string message = StatusToString(status);
      iree_status_ignore(status);
      GTEST_SKIP() << message;
    }
    IREE_ASSERT_OK(status);
  }

  void BuildAddI32Function() {
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_type_t arg_types[2] = {i32, i32};
    loom_type_t result_types[1] = {i32};
    loom_symbol_ref_t callee = MakeSymbol(IREE_SV("add_i32"));
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types, 2, result_types,
        1, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 2);
    SetValueName(args[0], IREE_SV("lhs"));
    SetValueName(args[1], IREE_SV("rhs"));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* add_op = NULL;
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &body_builder, LOOM_SCALAR_INTOVERFLOWFLAGS_NSW, args[0], args[1], i32,
        LOOM_LOCATION_UNKNOWN, &add_op));
    loom_value_id_t sum = loom_scalar_addi_result(add_op);
    SetValueName(sum, IREE_SV("sum"));
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &sum, 1,
                                          LOOM_LOCATION_UNKNOWN, &add_op));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = NULL;
  loom_builder_t module_builder_;
};

TEST_F(LlvmIrLowerTest, LowersIntegerArithmeticAndReturn) {
  BuildAddI32Function();
  std::string text = LowerToText();
  EXPECT_NE(text.find("define dso_local i32 @add_i32(i32 %lhs, i32 %rhs) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %sum = add nsw i32 %lhs, %rhs\n"), std::string::npos)
      << text;
  EXPECT_NE(text.find("  ret i32 %sum\n"), std::string::npos) << text;
}

TEST_F(LlvmIrLowerTest, LoweredTextAssemblesWithLlvmTools) {
  BuildAddI32Function();
  VerifyTextWithLlvmTools(LowerToText());
}

TEST_F(LlvmIrLowerTest, LowersCallsComparisonsSelectAndCasts) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t i16 = loom_type_scalar(LOOM_SCALAR_TYPE_I16);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("callee_i32"));
  loom_type_t callee_arg_types[1] = {i32};
  loom_type_t callee_result_types[1] = {i32};
  loom_op_t* callee_op = NULL;
  IREE_ASSERT_OK(loom_func_decl_build(&module_builder_, 0, 0, 0, 0, 0, 0,
                                      callee, callee_arg_types, 1,
                                      callee_result_types, 1, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &callee_op));
  loom_func_like_t callee_func = loom_func_like_cast(module_, callee_op);
  uint16_t callee_arg_count = 0;
  const loom_value_id_t* callee_args =
      loom_func_like_arg_ids(callee_func, &callee_arg_count);
  ASSERT_EQ(callee_arg_count, 1);
  SetValueName(callee_args[0], IREE_SV("value"));

  loom_symbol_ref_t caller = MakeSymbol(IREE_SV("choose_cast"));
  loom_type_t caller_arg_types[3] = {i32, i32, i16};
  loom_type_t caller_result_types[1] = {i32};
  loom_op_t* caller_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, caller, caller_arg_types, 3,
      caller_result_types, 1, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
      &caller_op));
  loom_func_like_t caller_func = loom_func_like_cast(module_, caller_op);
  uint16_t caller_arg_count = 0;
  const loom_value_id_t* caller_args =
      loom_func_like_arg_ids(caller_func, &caller_arg_count);
  ASSERT_EQ(caller_arg_count, 3);
  SetValueName(caller_args[0], IREE_SV("lhs"));
  SetValueName(caller_args[1], IREE_SV("rhs"));
  SetValueName(caller_args[2], IREE_SV("narrow"));

  loom_builder_t body_builder = BodyBuilder(caller_op);
  loom_op_t* ext_op = NULL;
  IREE_ASSERT_OK(loom_scalar_extsi_build(&body_builder, caller_args[2], i16,
                                         i32, LOOM_LOCATION_UNKNOWN, &ext_op));
  loom_value_id_t wide = loom_scalar_extsi_result(ext_op);
  SetValueName(wide, IREE_SV("wide"));

  loom_op_t* call_op = NULL;
  IREE_ASSERT_OK(loom_func_call_build(&body_builder, 0, 0, callee, &wide, 1,
                                      callee_result_types, 1, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &call_op));
  loom_value_id_t called =
      loom_value_slice_get(loom_func_call_results(call_op), 0);
  SetValueName(called, IREE_SV("called"));

  loom_op_t* cmp_op = NULL;
  IREE_ASSERT_OK(loom_scalar_cmpi_build(
      &body_builder, LOOM_SCALAR_CMPI_PREDICATE_SGT, called, caller_args[1],
      i32, i1, LOOM_LOCATION_UNKNOWN, &cmp_op));
  loom_value_id_t predicate = loom_scalar_cmpi_result(cmp_op);
  SetValueName(predicate, IREE_SV("predicate"));

  loom_op_t* select_op = NULL;
  IREE_ASSERT_OK(loom_scalar_select_build(&body_builder, predicate, called,
                                          caller_args[0], i32,
                                          LOOM_LOCATION_UNKNOWN, &select_op));
  loom_value_id_t selected = loom_scalar_select_result(select_op);
  SetValueName(selected, IREE_SV("selected"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &selected, 1,
                                        LOOM_LOCATION_UNKNOWN, &select_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("declare i32 @callee_i32(i32 %value)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %wide = sext i16 %narrow to i32\n"), std::string::npos)
      << text;
  EXPECT_NE(text.find("  %called = call i32 @callee_i32(i32 %wide)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %predicate = icmp sgt i32 %called, %rhs\n"),
            std::string::npos)
      << text;
  EXPECT_NE(
      text.find("  %selected = select i1 %predicate, i32 %called, i32 %lhs\n"),
      std::string::npos)
      << text;
}

TEST_F(LlvmIrLowerTest, LowersIndexCastUsingTargetIndexWidth) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t arg_types[1] = {i32};
  loom_type_t result_types[1] = {index};
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("extend_index"));
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types, 1, result_types, 1,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 1);
  SetValueName(args[0], IREE_SV("n"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* cast_op = NULL;
  IREE_ASSERT_OK(loom_index_cast_build(&body_builder, args[0], i32, index,
                                       LOOM_LOCATION_UNKNOWN, &cast_op));
  loom_value_id_t extended = loom_index_cast_result(cast_op);
  SetValueName(extended, IREE_SV("extended"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &extended, 1,
                                        LOOM_LOCATION_UNKNOWN, &cast_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define dso_local i64 @extend_index(i32 %n) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %extended = sext i32 %n to i64\n"), std::string::npos)
      << text;
  EXPECT_NE(text.find("  ret i64 %extended\n"), std::string::npos) << text;
}

TEST_F(LlvmIrLowerTest, LowersDenseBufferViewLoadStore) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t view_type = loom_type_shaped_2d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4),
      loom_dim_pack_static(8), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("dense_memory"));
  loom_type_t arg_types[5] = {buffer, offset, index, index, f32};
  loom_type_t result_types[1] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types, 5, result_types, 1,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 5);
  SetValueName(args[0], IREE_SV("buffer"));
  SetValueName(args[1], IREE_SV("offset"));
  SetValueName(args[2], IREE_SV("row"));
  SetValueName(args[3], IREE_SV("col"));
  SetValueName(args[4], IREE_SV("value"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], args[1],
                                        view_type, LOOM_LOCATION_UNKNOWN,
                                        &view_op));
  loom_value_id_t view = loom_buffer_view_result(view_op);
  SetValueName(view, IREE_SV("view"));

  int64_t dynamic_load_indices[2] = {INT64_MIN, INT64_MIN};
  loom_value_id_t load_indices[2] = {args[2], args[3]};
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_view_load_build(&body_builder, view, load_indices, 2,
                                      dynamic_load_indices, 2, f32,
                                      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_view_load_result(load_op);
  SetValueName(loaded, IREE_SV("loaded"));

  int64_t mixed_store_indices[2] = {0, INT64_MIN};
  loom_value_id_t store_indices[1] = {args[3]};
  loom_op_t* store_op = NULL;
  IREE_ASSERT_OK(loom_view_store_build(&body_builder, args[4], view,
                                       store_indices, 1, mixed_store_indices, 2,
                                       LOOM_LOCATION_UNKNOWN, &store_op));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &loaded, 1,
                                        LOOM_LOCATION_UNKNOWN, &store_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define dso_local float @dense_memory(ptr %buffer, "
                      "i64 %offset, i64 %row, i64 %col, float %value) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %view = getelementptr i8, ptr %buffer, i64 %offset\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find(" = mul i64 %row, 8\n"), std::string::npos) << text;
  EXPECT_NE(text.find(" = getelementptr float, ptr %view, i64 "),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %loaded = load float, ptr "), std::string::npos)
      << text;
  EXPECT_NE(text.find("  store float %value, ptr "), std::string::npos) << text;
  VerifyTextWithLlvmTools(text);
}

TEST_F(LlvmIrLowerTest, LowersBufferAllocaWithAlignedScalarAccess) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t buffer = loom_type_buffer();
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(1), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("stack_memory"));
  loom_type_t arg_types[2] = {offset, f32};
  loom_type_t result_types[1] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types, 2, result_types, 1,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2);
  SetValueName(args[0], IREE_SV("bytes"));
  SetValueName(args[1], IREE_SV("value"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* alloca_op = NULL;
  IREE_ASSERT_OK(loom_buffer_alloca_build(
      &body_builder, args[0], 16, LOOM_BUFFER_MEMORY_SPACE_PRIVATE, buffer,
      LOOM_LOCATION_UNKNOWN, &alloca_op));
  loom_value_id_t scratch = loom_buffer_alloca_result(alloca_op);
  SetValueName(scratch, IREE_SV("scratch"));

  loom_op_t* zero_op = NULL;
  IREE_ASSERT_OK(loom_index_constant_build(&body_builder, loom_attr_i64(0),
                                           offset, LOOM_LOCATION_UNKNOWN,
                                           &zero_op));
  loom_value_id_t zero = loom_index_constant_result(zero_op);

  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, scratch, zero, view_type,
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_value_id_t view = loom_buffer_view_result(view_op);
  SetValueName(view, IREE_SV("view"));

  int64_t static_index[1] = {0};
  loom_op_t* store_op = NULL;
  IREE_ASSERT_OK(loom_view_store_build(&body_builder, args[1], view, NULL, 0,
                                       static_index, 1, LOOM_LOCATION_UNKNOWN,
                                       &store_op));
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_view_load_build(&body_builder, view, NULL, 0,
                                      static_index, 1, f32,
                                      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_view_load_result(load_op);
  SetValueName(loaded, IREE_SV("loaded"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &loaded, 1,
                                        LOOM_LOCATION_UNKNOWN, &load_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("  %scratch = alloca i8, i64 %bytes, align 16\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %view = getelementptr i8, ptr %scratch, i64 0\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  store float %value, ptr "), std::string::npos) << text;
  EXPECT_NE(text.find(", align 4\n"), std::string::npos) << text;
  EXPECT_NE(text.find("  %loaded = load float, ptr "), std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
}

TEST_F(LlvmIrLowerTest, ReportsUnsupportedOpName) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("bad"));
  loom_type_t result_types[1] = {i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, NULL, 0, result_types, 1, NULL,
      0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* poison_op = NULL;
  IREE_ASSERT_OK(loom_scalar_poison_build(&body_builder, i32,
                                          LOOM_LOCATION_UNKNOWN, &poison_op));
  loom_value_id_t poison = loom_scalar_poison_result(poison_op);
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &poison, 1,
                                        LOOM_LOCATION_UNKNOWN, &poison_op));

  loom_llvmir_lowering_options_t options;
  options.target_profile = loom_llvmir_target_profile_x86_64_object();
  options.source_name = IREE_SV("lower_test");
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = loom_llvmir_lower_module(
      module_, &options, iree_allocator_system(), &lowered);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  std::string message = StatusToString(status);
  EXPECT_NE(message.find("scalar.poison"), std::string::npos) << message;
  iree_status_free(status);
  if (lowered) {
    loom_llvmir_module_free(lowered);
  }
}

}  // namespace
}  // namespace loom
