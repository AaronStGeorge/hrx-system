// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/lower.h"

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
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/x86/packed_dot_contract.h"
#include "loom/target/emit/llvmir/amdgpu/lower.h"
#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/tool.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/target/emit/llvmir/x86/lower.h"
#include "loom/target/emit/llvmir/x86/target_env.h"
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

std::string ToolOutputToString(const loom_llvmir_tool_output_t& output) {
  return output.data ? std::string(output.data, output.length) : std::string();
}

iree_string_view_t StringView(const std::string& value) {
  return iree_make_string_view(value.data(), value.size());
}

bool TextHasLineContaining(const std::string& text, const char* first,
                           const char* second) {
  size_t line_start = 0;
  while (line_start <= text.size()) {
    size_t line_end = text.find('\n', line_start);
    if (line_end == std::string::npos) line_end = text.size();
    size_t first_position = text.find(first, line_start);
    size_t second_position = text.find(second, line_start);
    if (first_position != std::string::npos && first_position < line_end &&
        second_position != std::string::npos && second_position < line_end) {
      return true;
    }
    if (line_end == text.size()) return false;
    line_start = line_end + 1;
  }
  return false;
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
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LLVMIR, loom_llvmir_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
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

  loom_string_id_t InternString(iree_string_view_t string) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, string, &string_id));
    return string_id;
  }

  loom_string_id_t IntrinsicKind(iree_string_view_t kind) {
    return InternString(kind);
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

  void SetBlockLabel(loom_block_t* block, iree_string_view_t label) {
    loom_string_id_t label_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, label, &label_id));
    block->label_id = label_id;
  }

  loom_block_t* AppendLabeledBlock(loom_region_t* region,
                                   iree_string_view_t label) {
    loom_block_t* block = NULL;
    IREE_CHECK_OK(loom_region_append_block(module_, region, &block));
    SetBlockLabel(block, label);
    return block;
  }

  loom_value_id_t AddBlockArgument(loom_block_t* block, loom_type_t type,
                                   iree_string_view_t name) {
    loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &arg));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, arg));
    SetValueName(arg, name);
    return arg;
  }

  void SetBodyBuilderBlock(loom_builder_t* builder, loom_op_t* func_op,
                           loom_block_t* block) {
    loom_builder_set_block(builder, block);
    builder->ip.parent_op = func_op;
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

  iree_status_t LowerModule(const loom_llvmir_target_profile_t* profile,
                            loom_llvmir_module_t** out_lowered) {
    const loom_llvmir_lowering_provider_t* providers[] = {
        loom_llvmir_x86_lowering_provider(),
        loom_llvmir_amdgpu_lowering_provider(),
    };
    loom_llvmir_lowering_options_t options = {};
    options.target_profile = profile;
    options.source_name = IREE_SV("lower_test");
    options.providers = providers;
    options.provider_count = IREE_ARRAYSIZE(providers);
    return loom_llvmir_lower_module(module_, &options, iree_allocator_system(),
                                    out_lowered);
  }

  std::string LowerToText(const loom_llvmir_target_profile_t* profile) {
    loom_llvmir_module_t* lowered = NULL;
    IREE_CHECK_OK(LowerModule(profile, &lowered));
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

  std::string LowerToText() {
    return LowerToText(loom_llvmir_target_profile_x86_64_object());
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

  void CompileX86TextToObject(const std::string& text) {
    TempFile input_file(TempPath(".ll"));
    TempFile bitcode_file(TempPath(".bc"));
    TempFile object_file(TempPath(".o"));
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

    status = loom_llvmir_tool_compile_object_file(
        &toolchain, StringView(bitcode_file.path()),
        StringView(object_file.path()), NULL, 0, iree_allocator_system());
    if (IsToolUnavailable(status)) {
      std::string message = StatusToString(status);
      iree_status_ignore(status);
      GTEST_SKIP() << message;
    }
    IREE_ASSERT_OK(status);

    iree_io_file_contents_t* object_contents = NULL;
    IREE_ASSERT_OK(iree_io_file_contents_read(StringView(object_file.path()),
                                              iree_allocator_system(),
                                              &object_contents));
    ASSERT_GT(object_contents->const_buffer.data_length, 0u);
    iree_io_file_contents_free(object_contents);
  }

  void CompileX86TextToAssembly(const std::string& text,
                                const iree_string_view_t* extra_arguments,
                                iree_host_size_t extra_argument_count,
                                std::string* out_assembly) {
    ASSERT_NE(out_assembly, nullptr);
    out_assembly->clear();
    TempFile input_file(TempPath(".ll"));
    TempFile bitcode_file(TempPath(".bc"));
    TempFile assembly_file(TempPath(".s"));
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

    status = loom_llvmir_tool_compile_assembly_file(
        &toolchain, StringView(bitcode_file.path()),
        StringView(assembly_file.path()), extra_arguments, extra_argument_count,
        iree_allocator_system());
    if (IsToolUnavailable(status)) {
      std::string message = StatusToString(status);
      iree_status_ignore(status);
      GTEST_SKIP() << message;
    }
    IREE_ASSERT_OK(status);

    iree_io_file_contents_t* assembly_contents = NULL;
    IREE_ASSERT_OK(iree_io_file_contents_read(StringView(assembly_file.path()),
                                              iree_allocator_system(),
                                              &assembly_contents));
    *out_assembly =
        std::string((const char*)assembly_contents->const_buffer.data,
                    assembly_contents->const_buffer.data_length);
    iree_io_file_contents_free(assembly_contents);
  }

  void CompileAmdgpuTextToObjectIfAvailable(const std::string& text) {
    TempFile input_file(TempPath(".ll"));
    TempFile bitcode_file(TempPath(".bc"));
    TempFile object_file(TempPath(".o"));
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

    loom_llvmir_tool_output_t version_text = {};
    status =
        loom_llvmir_tool_query_version(&toolchain, LOOM_LLVMIR_TOOL_LLC,
                                       iree_allocator_system(), &version_text);
    if (IsToolUnavailable(status)) {
      std::string message = StatusToString(status);
      iree_status_ignore(status);
      GTEST_SKIP() << message;
    }
    IREE_ASSERT_OK(status);
    std::string version = ToolOutputToString(version_text);
    loom_llvmir_tool_output_deinitialize(&version_text,
                                         iree_allocator_system());
    if (version.find("amdgcn") == std::string::npos &&
        version.find("AMDGPU") == std::string::npos) {
      GTEST_SKIP() << "installed llc does not advertise an AMDGPU target";
    }

    iree_string_view_t extra_arguments[] = {
        IREE_SV("-mtriple=amdgcn-amd-amdhsa"),
        IREE_SV("-mcpu=gfx1100"),
    };
    status = loom_llvmir_tool_compile_object_file(
        &toolchain, StringView(bitcode_file.path()),
        StringView(object_file.path()), extra_arguments,
        IREE_ARRAYSIZE(extra_arguments), iree_allocator_system());
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

  void BuildDot2FFunction(iree_string_view_t function_name,
                          loom_scalar_type_t source_element_type,
                          int64_t input_lane_count, int64_t result_lane_count) {
    loom_type_t input_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, source_element_type,
                            loom_dim_pack_static(input_lane_count), 0);
    loom_type_t result_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_static(result_lane_count), 0);
    loom_symbol_ref_t symbol = MakeSymbol(function_name);
    loom_type_t arg_types[3] = {input_type, input_type, result_type};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, symbol,
                                       arg_types, IREE_ARRAYSIZE(arg_types),
                                       &result_type, 1, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 3);
    SetValueName(args[0], IREE_SV("lhs"));
    SetValueName(args[1], IREE_SV("rhs"));
    SetValueName(args[2], IREE_SV("acc"));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* dot_op = NULL;
    IREE_ASSERT_OK(loom_vector_dot2f_build(&body_builder, args[0], args[1],
                                           args[2], result_type,
                                           LOOM_LOCATION_UNKNOWN, &dot_op));
    loom_value_id_t dot = loom_vector_dot2f_result(dot_op);
    SetValueName(dot, IREE_SV("dot"));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &dot, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildDot2Bf16Function(iree_string_view_t function_name) {
    BuildDot2FFunction(function_name, LOOM_SCALAR_TYPE_BF16, 16, 8);
  }

  void BuildDot2F16Avx10Function(iree_string_view_t function_name) {
    BuildDot2FFunction(function_name, LOOM_SCALAR_TYPE_F16, 32, 16);
  }

  void BuildDot4IFunction(iree_string_view_t function_name,
                          loom_vector_dot4i_kind_t kind,
                          int64_t input_lane_count, int64_t result_lane_count) {
    loom_type_t input_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                            loom_dim_pack_static(input_lane_count), 0);
    loom_type_t result_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(result_lane_count), 0);
    loom_symbol_ref_t symbol = MakeSymbol(function_name);
    loom_type_t arg_types[3] = {input_type, input_type, result_type};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, symbol,
                                       arg_types, IREE_ARRAYSIZE(arg_types),
                                       &result_type, 1, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 3);
    SetValueName(args[0], IREE_SV("lhs"));
    SetValueName(args[1], IREE_SV("rhs"));
    SetValueName(args[2], IREE_SV("acc"));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* dot_op = NULL;
    IREE_ASSERT_OK(loom_vector_dot4i_build(&body_builder, kind, args[0],
                                           args[1], args[2], result_type,
                                           LOOM_LOCATION_UNKNOWN, &dot_op));
    loom_value_id_t dot = loom_vector_dot4i_result(dot_op);
    SetValueName(dot, IREE_SV("dot"));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &dot, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildDot4S8S8Function(iree_string_view_t function_name) {
    BuildDot4IFunction(function_name, LOOM_VECTOR_DOT4I_KIND_S8S8, 32, 8);
  }

  void BuildDot4U8S8Function(iree_string_view_t function_name) {
    BuildDot4IFunction(function_name, LOOM_VECTOR_DOT4I_KIND_U8S8, 32, 8);
  }

  void BuildDot4S8S8Avx10Function(iree_string_view_t function_name) {
    BuildDot4IFunction(function_name, LOOM_VECTOR_DOT4I_KIND_S8S8, 64, 16);
  }

  void BuildDot8I4Function(iree_string_view_t function_name) {
    loom_type_t packed_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
    loom_symbol_ref_t symbol = MakeSymbol(function_name);
    loom_type_t arg_types[3] = {packed_type, packed_type, packed_type};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, symbol,
                                       arg_types, IREE_ARRAYSIZE(arg_types),
                                       &packed_type, 1, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 3);
    SetValueName(args[0], IREE_SV("lhs"));
    SetValueName(args[1], IREE_SV("rhs"));
    SetValueName(args[2], IREE_SV("acc"));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* dot_op = NULL;
    IREE_ASSERT_OK(loom_vector_dot8i4_build(
        &body_builder, LOOM_VECTOR_DOT8I4_KIND_S4S4, args[0], args[1], args[2],
        packed_type, LOOM_LOCATION_UNKNOWN, &dot_op));
    loom_value_id_t dot = loom_vector_dot8i4_result(dot_op);
    SetValueName(dot, IREE_SV("dot"));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &dot, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildDot4F8Function(iree_string_view_t function_name) {
    loom_type_t packed_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
    loom_type_t accumulator_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(8), 0);
    loom_symbol_ref_t symbol = MakeSymbol(function_name);
    loom_type_t arg_types[3] = {packed_type, packed_type, accumulator_type};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, symbol,
                                       arg_types, IREE_ARRAYSIZE(arg_types),
                                       &accumulator_type, 1, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 3);
    SetValueName(args[0], IREE_SV("lhs"));
    SetValueName(args[1], IREE_SV("rhs"));
    SetValueName(args[2], IREE_SV("acc"));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* dot_op = NULL;
    IREE_ASSERT_OK(loom_vector_dot4f8_build(
        &body_builder, LOOM_VECTOR_DOT4F8_KIND_FP8BF8, args[0], args[1],
        args[2], accumulator_type, LOOM_LOCATION_UNKNOWN, &dot_op));
    loom_value_id_t dot = loom_vector_dot4f8_result(dot_op);
    SetValueName(dot, IREE_SV("dot"));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &dot, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
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

TEST_F(LlvmIrLowerTest, LoweredTextCompilesToX86Object) {
  BuildAddI32Function();
  CompileX86TextToObject(LowerToText());
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
  IREE_ASSERT_OK(loom_scf_select_build(&body_builder, predicate, called,
                                       caller_args[0], i32,
                                       LOOM_LOCATION_UNKNOWN, &select_op));
  loom_value_id_t selected = loom_scf_select_result(select_op);
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

TEST_F(LlvmIrLowerTest, LowersCfgDiamondBlockArgumentToPhi) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("cfg_select_i32"));
  loom_type_t arg_types[3] = {i1, i32, i32};
  loom_type_t result_types[1] = {i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, symbol, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3);
  SetValueName(args[0], IREE_SV("cond"));
  SetValueName(args[1], IREE_SV("a"));
  SetValueName(args[2], IREE_SV("b"));

  loom_region_t* body = loom_func_like_body(func);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_block_t* then_block = AppendLabeledBlock(body, IREE_SV("then"));
  loom_block_t* else_block = AppendLabeledBlock(body, IREE_SV("else"));
  loom_block_t* join_block = AppendLabeledBlock(body, IREE_SV("join"));
  loom_value_id_t result = AddBlockArgument(join_block, i32, IREE_SV("result"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  SetBodyBuilderBlock(&body_builder, func_op, entry_block);
  loom_op_t* branch_op = NULL;
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&body_builder, args[0], then_block,
                                        else_block, LOOM_LOCATION_UNKNOWN,
                                        &branch_op));

  SetBodyBuilderBlock(&body_builder, func_op, then_block);
  loom_value_id_t then_args[1] = {args[1]};
  IREE_ASSERT_OK(loom_cfg_br_build(&body_builder, join_block, then_args,
                                   IREE_ARRAYSIZE(then_args),
                                   LOOM_LOCATION_UNKNOWN, &branch_op));

  SetBodyBuilderBlock(&body_builder, func_op, else_block);
  loom_value_id_t else_args[1] = {args[2]};
  IREE_ASSERT_OK(loom_cfg_br_build(&body_builder, join_block, else_args,
                                   IREE_ARRAYSIZE(else_args),
                                   LOOM_LOCATION_UNKNOWN, &branch_op));

  SetBodyBuilderBlock(&body_builder, func_op, join_block);
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &result, 1,
                                        LOOM_LOCATION_UNKNOWN, &branch_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define dso_local i32 @cfg_select_i32(i1 %cond, "
                      "i32 %a, i32 %b) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("entry:\n  br i1 %cond, label %then, label %else\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("then:\n  br label %join\n"), std::string::npos) << text;
  EXPECT_NE(text.find("else:\n  br label %join\n"), std::string::npos) << text;
  EXPECT_TRUE(TextHasLineContaining(text, "%result = phi i32",
                                    "[ %a, %then ], [ %b, %else ]"))
      << text;
  EXPECT_NE(text.find("  ret i32 %result\n"), std::string::npos) << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersCfgLoopBlockArgumentsToBackedgePhis) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("cfg_loop_i32"));
  loom_type_t arg_types[2] = {index, i32};
  loom_type_t result_types[1] = {i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, symbol, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2);
  SetValueName(args[0], IREE_SV("n"));
  SetValueName(args[1], IREE_SV("seed"));

  loom_region_t* body = loom_func_like_body(func);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_block_t* loop_block = AppendLabeledBlock(body, IREE_SV("loop"));
  loom_block_t* body_block = AppendLabeledBlock(body, IREE_SV("body"));
  loom_block_t* exit_block = AppendLabeledBlock(body, IREE_SV("exit"));
  loom_value_id_t i = AddBlockArgument(loop_block, index, IREE_SV("i"));
  loom_value_id_t acc = AddBlockArgument(loop_block, i32, IREE_SV("acc"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  SetBodyBuilderBlock(&body_builder, func_op, entry_block);
  loom_op_t* zero_op = NULL;
  IREE_ASSERT_OK(loom_index_constant_build(
      &body_builder, loom_attr_i64(0), index, LOOM_LOCATION_UNKNOWN, &zero_op));
  loom_value_id_t zero = loom_index_constant_result(zero_op);
  loom_op_t* one_op = NULL;
  IREE_ASSERT_OK(loom_index_constant_build(
      &body_builder, loom_attr_i64(1), index, LOOM_LOCATION_UNKNOWN, &one_op));
  loom_value_id_t one = loom_index_constant_result(one_op);
  loom_value_id_t entry_args[2] = {zero, args[1]};
  loom_op_t* branch_op = NULL;
  IREE_ASSERT_OK(loom_cfg_br_build(&body_builder, loop_block, entry_args,
                                   IREE_ARRAYSIZE(entry_args),
                                   LOOM_LOCATION_UNKNOWN, &branch_op));

  SetBodyBuilderBlock(&body_builder, func_op, loop_block);
  loom_op_t* cmp_op = NULL;
  IREE_ASSERT_OK(
      loom_index_cmp_build(&body_builder, LOOM_INDEX_CMP_PREDICATE_SLT, i,
                           args[0], index, i1, LOOM_LOCATION_UNKNOWN, &cmp_op));
  loom_value_id_t keep_going = loom_index_cmp_result(cmp_op);
  SetValueName(keep_going, IREE_SV("keep_going"));
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&body_builder, keep_going, body_block,
                                        exit_block, LOOM_LOCATION_UNKNOWN,
                                        &branch_op));

  SetBodyBuilderBlock(&body_builder, func_op, body_block);
  loom_op_t* next_i_op = NULL;
  IREE_ASSERT_OK(loom_index_add_build(&body_builder, i, one, index,
                                      LOOM_LOCATION_UNKNOWN, &next_i_op));
  loom_value_id_t next_i = loom_index_add_result(next_i_op);
  SetValueName(next_i, IREE_SV("next_i"));
  loom_op_t* next_acc_op = NULL;
  IREE_ASSERT_OK(loom_scalar_addi_build(
      &body_builder, LOOM_SCALAR_INTOVERFLOWFLAGS_NSW, acc, args[1], i32,
      LOOM_LOCATION_UNKNOWN, &next_acc_op));
  loom_value_id_t next_acc = loom_scalar_addi_result(next_acc_op);
  SetValueName(next_acc, IREE_SV("next_acc"));
  loom_value_id_t backedge_args[2] = {next_i, next_acc};
  IREE_ASSERT_OK(loom_cfg_br_build(&body_builder, loop_block, backedge_args,
                                   IREE_ARRAYSIZE(backedge_args),
                                   LOOM_LOCATION_UNKNOWN, &branch_op));

  SetBodyBuilderBlock(&body_builder, func_op, exit_block);
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &acc, 1,
                                        LOOM_LOCATION_UNKNOWN, &branch_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define internal i32 @cfg_loop_i32(i64 %n, "
                      "i32 %seed) {"),
            std::string::npos)
      << text;
  EXPECT_TRUE(TextHasLineContaining(text, "%i = phi i64",
                                    "[ 0, %entry ], [ %next_i, %body ]"))
      << text;
  EXPECT_TRUE(TextHasLineContaining(text, "%acc = phi i32",
                                    "[ %seed, %entry ], [ %next_acc, %body ]"))
      << text;
  EXPECT_NE(text.find("  %keep_going = icmp slt i64 %i, %n\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %next_i = add i64 %i, 1\n"), std::string::npos)
      << text;
  EXPECT_NE(text.find("  %next_acc = add nsw i32 %acc, %seed\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("exit:\n  ret i32 %acc\n"), std::string::npos) << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersScfSelectAsWholeValueSelect) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t v4i32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("choose_vector"));
  loom_type_t arg_types[3] = {i1, v4i32, v4i32};
  loom_type_t result_types[1] = {v4i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3);
  SetValueName(args[0], IREE_SV("cond"));
  SetValueName(args[1], IREE_SV("lhs"));
  SetValueName(args[2], IREE_SV("rhs"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* select_op = NULL;
  IREE_ASSERT_OK(loom_scf_select_build(&body_builder, args[0], args[1], args[2],
                                       v4i32, LOOM_LOCATION_UNKNOWN,
                                       &select_op));
  loom_value_id_t selected = loom_scf_select_result(select_op);
  SetValueName(selected, IREE_SV("selected"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &selected, 1,
                                        LOOM_LOCATION_UNKNOWN, &select_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define dso_local <4 x i32> @choose_vector(i1 %cond, "
                      "<4 x i32> %lhs, <4 x i32> %rhs) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %selected = select i1 %cond, <4 x i32> %lhs, "
                      "<4 x i32> %rhs\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  ret <4 x i32> %selected\n"), std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersBfloatTypes) {
  loom_type_t bf16 = loom_type_scalar(LOOM_SCALAR_TYPE_BF16);
  loom_type_t v8bf16 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_BF16, loom_dim_pack_static(8), 0);

  loom_symbol_ref_t scalar_symbol = MakeSymbol(IREE_SV("identity_bf16"));
  loom_op_t* scalar_func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, scalar_symbol, &bf16, 1, &bf16, 1, NULL, 0,
      NULL, 0, LOOM_LOCATION_UNKNOWN, &scalar_func_op));
  loom_func_like_t scalar_func = loom_func_like_cast(module_, scalar_func_op);
  uint16_t scalar_arg_count = 0;
  const loom_value_id_t* scalar_args =
      loom_func_like_arg_ids(scalar_func, &scalar_arg_count);
  ASSERT_EQ(scalar_arg_count, 1);
  SetValueName(scalar_args[0], IREE_SV("value"));
  loom_builder_t scalar_body_builder = BodyBuilder(scalar_func_op);
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&scalar_body_builder, scalar_args, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_symbol_ref_t vector_symbol = MakeSymbol(IREE_SV("identity_v8bf16"));
  loom_type_t arg_types[2] = {bf16, v8bf16};
  loom_op_t* vector_func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, vector_symbol, arg_types,
      IREE_ARRAYSIZE(arg_types), &v8bf16, 1, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &vector_func_op));
  loom_func_like_t vector_func = loom_func_like_cast(module_, vector_func_op);
  uint16_t vector_arg_count = 0;
  const loom_value_id_t* vector_args =
      loom_func_like_arg_ids(vector_func, &vector_arg_count);
  ASSERT_EQ(vector_arg_count, 2);
  SetValueName(vector_args[0], IREE_SV("scalar"));
  SetValueName(vector_args[1], IREE_SV("vector"));
  loom_builder_t vector_body_builder = BodyBuilder(vector_func_op);
  IREE_ASSERT_OK(loom_func_return_build(&vector_body_builder, &vector_args[1],
                                        1, LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define internal bfloat @identity_bf16(bfloat %value) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  ret bfloat %value\n"), std::string::npos) << text;
  EXPECT_NE(text.find("define internal <8 x bfloat> @identity_v8bf16(bfloat "
                      "%scalar, <8 x bfloat> %vector) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  ret <8 x bfloat> %vector\n"), std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
}

TEST_F(LlvmIrLowerTest, LowersCpuPackedDotIntrinsics) {
  BuildDot2Bf16Function(IREE_SV("dot2_bf16"));
  BuildDot4S8S8Function(IREE_SV("dot4_s8s8"));

  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_x86_64_object(&profile));
  profile.x86_packed_dot_feature_bits =
      LOOM_X86_PACKED_DOT_FEATURE_AVX512_BF16 |
      LOOM_X86_PACKED_DOT_FEATURE_AVX512_VL |
      LOOM_X86_PACKED_DOT_FEATURE_AVX_VNNI_INT8;
  std::string text = LowerToText(&profile);
  EXPECT_NE(text.find("declare <8 x float> "
                      "@llvm.x86.avx512bf16.dpbf16ps.256(<8 x float>, "
                      "<16 x bfloat>, <16 x bfloat>)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %dot = call <8 x float> "
                      "@llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %acc, "
                      "<16 x bfloat> %lhs, <16 x bfloat> %rhs)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare <8 x i32> "
                      "@llvm.x86.avx2.vpdpbssd.256(<8 x i32>, <8 x i32>, "
                      "<8 x i32>)\n"),
            std::string::npos)
      << text;
  EXPECT_TRUE(
      TextHasLineContaining(text, "bitcast <32 x i8> %lhs", "to <8 x i32>"))
      << text;
  EXPECT_TRUE(
      TextHasLineContaining(text, "bitcast <32 x i8> %rhs", "to <8 x i32>"))
      << text;
  EXPECT_TRUE(TextHasLineContaining(
      text, "call <8 x i32> @llvm.x86.avx2.vpdpbssd.256", "<8 x i32> %acc"))
      << text;
  VerifyTextWithLlvmTools(text);
}

TEST_F(LlvmIrLowerTest, RejectsCpuPackedDotWithoutTargetFeatures) {
  BuildDot4S8S8Function(IREE_SV("dot4_s8s8"));

  loom_llvmir_module_t* lowered = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      LowerModule(loom_llvmir_target_profile_x86_64_object(), &lowered));
  EXPECT_EQ(lowered, nullptr);
}

TEST_F(LlvmIrLowerTest, RejectsPackedI4DotWithoutExplicitReferenceLowering) {
  BuildDot8I4Function(IREE_SV("dot8_i4"));

  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_x86_64_object(&profile));
  profile.x86_packed_dot_feature_bits = LOOM_X86_PACKED_DOT_FEATURE_AVX10_2;
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = LowerModule(&profile, &lowered);
  std::string message = StatusToString(status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(lowered, nullptr);
  EXPECT_NE(message.find("packed i4 dot needs explicit unpack"),
            std::string::npos)
      << message;
  iree_status_free(status);
}

TEST_F(LlvmIrLowerTest, RejectsPackedF8DotWithoutExplicitReferenceLowering) {
  BuildDot4F8Function(IREE_SV("dot4_f8"));

  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_x86_64_object(&profile));
  profile.x86_packed_dot_feature_bits = LOOM_X86_PACKED_DOT_FEATURE_AVX10_2;
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = LowerModule(&profile, &lowered);
  std::string message = StatusToString(status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(lowered, nullptr);
  EXPECT_NE(message.find("packed fp8/bf8 dot needs explicit decode"),
            std::string::npos)
      << message;
  iree_status_free(status);
}

TEST_F(LlvmIrLowerTest, CompilesCpuPackedF16Dot2ToAvx10Assembly) {
  BuildDot2F16Avx10Function(IREE_SV("dot2_f16_avx10"));

  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_x86_64_object(&profile));
  profile.x86_packed_dot_feature_bits = LOOM_X86_PACKED_DOT_FEATURE_AVX10_2;
  std::string text = LowerToText(&profile);

  iree_string_view_t extra_arguments[] = {
      IREE_SV("-mtriple=x86_64-unknown-linux-gnu"),
      IREE_SV("-mattr=+avx10.2-512"),
  };
  std::string assembly;
  CompileX86TextToAssembly(text, extra_arguments,
                           IREE_ARRAYSIZE(extra_arguments), &assembly);
  EXPECT_NE(assembly.find("vdpphps"), std::string::npos) << assembly;
}

TEST_F(LlvmIrLowerTest, CompilesCpuPackedS8Dot4ToAvx10Assembly) {
  BuildDot4S8S8Avx10Function(IREE_SV("dot4_s8s8_avx10"));

  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_x86_64_object(&profile));
  profile.x86_packed_dot_feature_bits = LOOM_X86_PACKED_DOT_FEATURE_AVX10_2;
  std::string text = LowerToText(&profile);

  iree_string_view_t extra_arguments[] = {
      IREE_SV("-mtriple=x86_64-unknown-linux-gnu"),
      IREE_SV("-mattr=+avx10.2-512"),
  };
  std::string assembly;
  CompileX86TextToAssembly(text, extra_arguments,
                           IREE_ARRAYSIZE(extra_arguments), &assembly);
  EXPECT_NE(assembly.find("vpdpbssd"), std::string::npos) << assembly;
}

TEST_F(LlvmIrLowerTest, LowersVectorNumericOps) {
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t v4i32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("vector_numeric"));
  loom_type_t arg_types[4] = {v4f32, v4f32, v4i32, v4i32};
  loom_type_t result_types[1] = {v4f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 4);
  SetValueName(args[0], IREE_SV("lhsf"));
  SetValueName(args[1], IREE_SV("rhsf"));
  SetValueName(args[2], IREE_SV("lhsi"));
  SetValueName(args[3], IREE_SV("rhsi"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* addi_op = NULL;
  IREE_ASSERT_OK(loom_vector_addi_build(
      &body_builder, LOOM_VECTOR_INTOVERFLOWFLAGS_NSW, args[2], args[3], v4i32,
      LOOM_LOCATION_UNKNOWN, &addi_op));
  loom_value_id_t sumi = loom_vector_addi_result(addi_op);
  SetValueName(sumi, IREE_SV("sumi"));

  loom_type_t v4i1 = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1,
                                         loom_dim_pack_static(4), 0);
  loom_op_t* icmp_op = NULL;
  IREE_ASSERT_OK(loom_vector_cmpi_build(
      &body_builder, LOOM_VECTOR_CMPI_PREDICATE_SGT, sumi, args[3], v4i32, v4i1,
      LOOM_LOCATION_UNKNOWN, &icmp_op));
  loom_value_id_t int_mask = loom_vector_cmpi_result(icmp_op);
  SetValueName(int_mask, IREE_SV("int_mask"));

  loom_op_t* int_select_op = NULL;
  IREE_ASSERT_OK(loom_vector_select_build(&body_builder, int_mask, sumi,
                                          args[2], v4i32, LOOM_LOCATION_UNKNOWN,
                                          &int_select_op));
  loom_value_id_t selectedi = loom_vector_select_result(int_select_op);
  SetValueName(selectedi, IREE_SV("selectedi"));

  loom_op_t* cast_op = NULL;
  IREE_ASSERT_OK(loom_vector_sitofp_build(
      &body_builder, selectedi, v4i32, v4f32, LOOM_LOCATION_UNKNOWN, &cast_op));
  loom_value_id_t sumf = loom_vector_sitofp_result(cast_op);
  SetValueName(sumf, IREE_SV("sumf"));

  loom_op_t* mulf_op = NULL;
  IREE_ASSERT_OK(loom_vector_mulf_build(
      &body_builder, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NNAN, args[0], args[1],
      v4f32, LOOM_LOCATION_UNKNOWN, &mulf_op));
  loom_value_id_t product = loom_vector_mulf_result(mulf_op);
  SetValueName(product, IREE_SV("product"));

  loom_op_t* cmp_op = NULL;
  IREE_ASSERT_OK(loom_vector_cmpf_build(
      &body_builder, LOOM_VECTOR_CMPF_PREDICATE_OLT, product, sumf, v4f32, v4i1,
      LOOM_LOCATION_UNKNOWN, &cmp_op));
  loom_value_id_t mask = loom_vector_cmpf_result(cmp_op);
  SetValueName(mask, IREE_SV("mask"));

  loom_op_t* select_op = NULL;
  IREE_ASSERT_OK(loom_vector_select_build(&body_builder, mask, product, sumf,
                                          v4f32, LOOM_LOCATION_UNKNOWN,
                                          &select_op));
  loom_value_id_t selected = loom_vector_select_result(select_op);
  SetValueName(selected, IREE_SV("selected"));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &selected, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define internal <4 x float> @vector_numeric(<4 x "
                      "float> %lhsf, <4 x float> %rhsf, <4 x i32> %lhsi, "
                      "<4 x i32> %rhsi) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %sumi = add nsw <4 x i32> %lhsi, %rhsi\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %int_mask = icmp sgt <4 x i32> %sumi, %rhsi\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %selectedi = select <4 x i1> %int_mask, <4 x i32> "
                      "%sumi, <4 x i32> %lhsi\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %sumf = sitofp <4 x i32> %selectedi to <4 x float>\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %product = fmul nnan <4 x float> %lhsf, %rhsf\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %mask = fcmp olt <4 x float> %product, %sumf\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %selected = select <4 x i1> %mask, <4 x float> "
                      "%product, <4 x float> %sumf\n"),
            std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);

  std::string amdgpu_text =
      LowerToText(loom_llvmir_target_profile_amdgpu_hal());
  EXPECT_NE(amdgpu_text.find("target triple = \"amdgcn-amd-amdhsa\"\n"),
            std::string::npos)
      << amdgpu_text;
  VerifyTextWithLlvmTools(amdgpu_text);
  CompileAmdgpuTextToObjectIfAvailable(amdgpu_text);
}

TEST_F(LlvmIrLowerTest, LowersIntegerVectorConstants) {
  loom_type_t v4i32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("vector_constant"));
  loom_type_t result_types[1] = {v4i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, callee, NULL,
                                     0, result_types,
                                     IREE_ARRAYSIZE(result_types), NULL, 0,
                                     NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* constant_op = NULL;
  IREE_ASSERT_OK(loom_vector_constant_build(&body_builder, loom_attr_i64(7),
                                            v4i32, LOOM_LOCATION_UNKNOWN,
                                            &constant_op));
  loom_value_id_t constant = loom_vector_constant_result(constant_op);
  SetValueName(constant, IREE_SV("constant"));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &constant, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define internal <4 x i32> @vector_constant() {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  ret <4 x i32> <i32 7, i32 7, i32 7, i32 7>\n"),
            std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersVectorConstructorsAndLaneOps) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("vector_construct"));
  loom_type_t arg_types[2] = {f32, f32};
  loom_type_t result_types[1] = {v4f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2);
  SetValueName(args[0], IREE_SV("scalar"));
  SetValueName(args[1], IREE_SV("other"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* splat_op = NULL;
  IREE_ASSERT_OK(loom_vector_splat_build(&body_builder, args[0], v4f32,
                                         LOOM_LOCATION_UNKNOWN, &splat_op));
  loom_value_id_t splat = loom_vector_splat_result(splat_op);
  SetValueName(splat, IREE_SV("splat"));

  loom_value_id_t elements[4] = {args[0], args[1], args[0], args[1]};
  loom_op_t* from_elements_op = NULL;
  IREE_ASSERT_OK(loom_vector_from_elements_build(
      &body_builder, elements, IREE_ARRAYSIZE(elements), v4f32,
      LOOM_LOCATION_UNKNOWN, &from_elements_op));
  loom_value_id_t built = loom_vector_from_elements_result(from_elements_op);
  SetValueName(built, IREE_SV("built"));

  int64_t extract_index[1] = {1};
  loom_op_t* extract_op = NULL;
  IREE_ASSERT_OK(loom_vector_extract_build(
      &body_builder, built, NULL, 0, extract_index,
      IREE_ARRAYSIZE(extract_index), f32, LOOM_LOCATION_UNKNOWN, &extract_op));
  loom_value_id_t lane = loom_vector_extract_result(extract_op);
  SetValueName(lane, IREE_SV("lane"));

  int64_t insert_index[1] = {2};
  loom_op_t* insert_op = NULL;
  IREE_ASSERT_OK(loom_vector_insert_build(
      &body_builder, lane, splat, NULL, 0, insert_index,
      IREE_ARRAYSIZE(insert_index), v4f32, LOOM_LOCATION_UNKNOWN, &insert_op));
  loom_value_id_t updated = loom_vector_insert_result(insert_op);
  SetValueName(updated, IREE_SV("updated"));

  int64_t source_lanes[4] = {3, 2, 1, 0};
  loom_op_t* shuffle_op = NULL;
  IREE_ASSERT_OK(loom_vector_shuffle_build(
      &body_builder, source_lanes, IREE_ARRAYSIZE(source_lanes), updated, v4f32,
      LOOM_LOCATION_UNKNOWN, &shuffle_op));
  loom_value_id_t reversed = loom_vector_shuffle_result(shuffle_op);
  SetValueName(reversed, IREE_SV("reversed"));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &reversed, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define internal <4 x float> @vector_construct(float "
                      "%scalar, float %other) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("insertelement <4 x float> poison, float %scalar, i32 "
                      "0\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %splat = shufflevector <4 x float> "),
            std::string::npos)
      << text;
  EXPECT_NE(text.find(", <4 x float> poison, <4 x i32> <i32 0, i32 0, i32 0, "
                      "i32 0>\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %built = insertelement <4 x float> "),
            std::string::npos)
      << text;
  EXPECT_NE(text.find(", float %other, i32 3\n"), std::string::npos) << text;
  EXPECT_NE(text.find("  %lane = extractelement <4 x float> %built, i32 1\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %updated = insertelement <4 x float> %splat, float "
                      "%lane, i32 2\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %reversed = shufflevector <4 x float> %updated, <4 x "
                      "float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 "
                      "0>\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  ret <4 x float> %reversed\n"), std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);

  std::string amdgpu_text =
      LowerToText(loom_llvmir_target_profile_amdgpu_hal());
  EXPECT_NE(amdgpu_text.find("target triple = \"amdgcn-amd-amdhsa\"\n"),
            std::string::npos)
      << amdgpu_text;
  VerifyTextWithLlvmTools(amdgpu_text);
  CompileAmdgpuTextToObjectIfAvailable(amdgpu_text);
}

TEST_F(LlvmIrLowerTest, RejectsStructuredScfControlFlowBeforeCfgLowering) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("structured_if"));
  loom_type_t arg_types[1] = {i1};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), NULL, 0, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 1);
  SetValueName(args[0], IREE_SV("cond"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* if_op = NULL;
  IREE_ASSERT_OK(loom_scf_if_build(&body_builder, args[0], NULL, 0, NULL, 0,
                                   LOOM_LOCATION_UNKNOWN, &if_op));

  loom_builder_t then_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_scf_if_then_region(if_op)), &then_builder);
  loom_op_t* then_yield_op = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&then_builder, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &then_yield_op));

  loom_builder_t else_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_scf_if_else_region(if_op)), &else_builder);
  loom_op_t* else_yield_op = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&else_builder, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &else_yield_op));

  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status =
      LowerModule(loom_llvmir_target_profile_x86_64_object(), &lowered);
  std::string message = StatusToString(status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(lowered, nullptr);
  EXPECT_NE(message.find("scf.if"), std::string::npos) << message;
  EXPECT_NE(message.find("lowered to CFG"), std::string::npos) << message;
  iree_status_free(status);
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
  IREE_ASSERT_OK(loom_view_load_build(&body_builder, 0, view, load_indices, 2,
                                      dynamic_load_indices, 2, 0, 0, f32,
                                      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_view_load_result(load_op);
  SetValueName(loaded, IREE_SV("loaded"));

  int64_t mixed_store_indices[2] = {0, INT64_MIN};
  loom_value_id_t store_indices[1] = {args[3]};
  loom_op_t* store_op = NULL;
  IREE_ASSERT_OK(loom_view_store_build(&body_builder, 0, args[4], view,
                                       store_indices, 1, mixed_store_indices, 2,
                                       0, 0, LOOM_LOCATION_UNKNOWN, &store_op));
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

TEST_F(LlvmIrLowerTest, LowersNonTemporalCachePolicyMetadata) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(16), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("cache_policy"));
  loom_type_t arg_types[5] = {buffer, offset, index, f32, v4f32};
  loom_type_t result_types[1] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, IREE_ARRAYSIZE(arg_types));
  SetValueName(args[0], IREE_SV("buffer"));
  SetValueName(args[1], IREE_SV("offset"));
  SetValueName(args[2], IREE_SV("i"));
  SetValueName(args[3], IREE_SV("scalar_value"));
  SetValueName(args[4], IREE_SV("vector_value"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], args[1],
                                        view_type, LOOM_LOCATION_UNKNOWN,
                                        &view_op));
  loom_value_id_t view = loom_buffer_view_result(view_op);
  SetValueName(view, IREE_SV("view"));

  int64_t dynamic_indices[1] = {INT64_MIN};
  loom_value_id_t indices[1] = {args[2]};
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_view_load_build(
      &body_builder,
      LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE |
          LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL,
      view, indices, IREE_ARRAYSIZE(indices), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), LOOM_VIEW_CACHE_SCOPE_DEVICE,
      LOOM_VIEW_CACHE_TEMPORAL_NON_TEMPORAL, f32, LOOM_LOCATION_UNKNOWN,
      &load_op));
  loom_value_id_t scalar_loaded = loom_view_load_result(load_op);
  SetValueName(scalar_loaded, IREE_SV("scalar_loaded"));

  loom_op_t* store_op = NULL;
  IREE_ASSERT_OK(loom_view_store_build(
      &body_builder,
      LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_SCOPE |
          LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL,
      args[3], view, indices, IREE_ARRAYSIZE(indices), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), LOOM_VIEW_CACHE_SCOPE_DEVICE,
      LOOM_VIEW_CACHE_TEMPORAL_NON_TEMPORAL, LOOM_LOCATION_UNKNOWN, &store_op));

  loom_op_t* vector_load_op = NULL;
  IREE_ASSERT_OK(loom_vector_load_build(
      &body_builder,
      LOOM_VECTOR_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE |
          LOOM_VECTOR_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL,
      view, indices, IREE_ARRAYSIZE(indices), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), LOOM_VECTOR_CACHE_SCOPE_DEVICE,
      LOOM_VECTOR_CACHE_TEMPORAL_NON_TEMPORAL, v4f32, LOOM_LOCATION_UNKNOWN,
      &vector_load_op));
  loom_value_id_t vector_loaded = loom_vector_load_result(vector_load_op);
  SetValueName(vector_loaded, IREE_SV("vector_loaded"));

  loom_op_t* vector_store_op = NULL;
  IREE_ASSERT_OK(loom_vector_store_build(
      &body_builder,
      LOOM_VECTOR_STORE_BUILD_FLAG_HAS_CACHE_SCOPE |
          LOOM_VECTOR_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL,
      args[4], view, indices, IREE_ARRAYSIZE(indices), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), LOOM_VECTOR_CACHE_SCOPE_DEVICE,
      LOOM_VECTOR_CACHE_TEMPORAL_NON_TEMPORAL, LOOM_LOCATION_UNKNOWN,
      &vector_store_op));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &scalar_loaded, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  EXPECT_TRUE(TextHasLineContaining(text, "%scalar_loaded = load float, ptr ",
                                    "!nontemporal !0"))
      << text;
  EXPECT_TRUE(TextHasLineContaining(text, "store float %scalar_value, ptr ",
                                    "!nontemporal !0"))
      << text;
  EXPECT_TRUE(TextHasLineContaining(
      text, "%vector_loaded = load <4 x float>, ptr ", "!nontemporal !0"))
      << text;
  EXPECT_TRUE(TextHasLineContaining(
      text, "store <4 x float> %vector_value, ptr ", "!nontemporal !0"))
      << text;
  EXPECT_NE(text.find("!0 = !{i32 1}\n"), std::string::npos) << text;
  VerifyTextWithLlvmTools(text);
}

TEST_F(LlvmIrLowerTest, RejectsUnsupportedCachePolicyDuringLowering) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(16), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("unsupported_cache_policy"));
  loom_type_t arg_types[2] = {buffer, offset};
  loom_type_t result_types[1] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, IREE_ARRAYSIZE(arg_types));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], args[1],
                                        view_type, LOOM_LOCATION_UNKNOWN,
                                        &view_op));
  loom_value_id_t view = loom_buffer_view_result(view_op);

  int64_t static_indices[1] = {0};
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_view_load_build(
      &body_builder,
      LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE |
          LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL,
      view, NULL, 0, static_indices, IREE_ARRAYSIZE(static_indices),
      LOOM_VIEW_CACHE_SCOPE_DEVICE, LOOM_VIEW_CACHE_TEMPORAL_HIGH_TEMPORAL, f32,
      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_view_load_result(load_op);
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &loaded, 1,
                                        LOOM_LOCATION_UNKNOWN, &load_op));

  loom_llvmir_module_t* lowered = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      LowerModule(loom_llvmir_target_profile_x86_64_object(), &lowered));
  if (lowered != NULL) {
    loom_llvmir_module_free(lowered);
  }
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
  IREE_ASSERT_OK(loom_view_store_build(&body_builder, 0, args[1], view, NULL, 0,
                                       static_index, 1, 0, 0,
                                       LOOM_LOCATION_UNKNOWN, &store_op));
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_view_load_build(&body_builder, 0, view, NULL, 0,
                                      static_index, 1, 0, 0, f32,
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

TEST_F(LlvmIrLowerTest, LowersBufferAllocaWithAlignedVectorAccess) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t buffer = loom_type_buffer();
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("stack_vector_memory"));
  loom_type_t arg_types[2] = {offset, v4f32};
  loom_type_t result_types[1] = {v4f32};
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
  IREE_ASSERT_OK(loom_vector_store_build(&body_builder, 0, args[1], view, NULL,
                                         0, static_index, 1, 0, 0,
                                         LOOM_LOCATION_UNKNOWN, &store_op));
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_vector_load_build(&body_builder, 0, view, NULL, 0,
                                        static_index, 1, 0, 0, v4f32,
                                        LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_vector_load_result(load_op);
  SetValueName(loaded, IREE_SV("loaded"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &loaded, 1,
                                        LOOM_LOCATION_UNKNOWN, &load_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define dso_local <4 x float> @stack_vector_memory("
                      "i64 %bytes, <4 x float> %value) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %scratch = alloca i8, i64 %bytes, align 16\n"),
            std::string::npos)
      << text;
  EXPECT_TRUE(TextHasLineContaining(text, "store <4 x float> %value, ptr ",
                                    ", align 16"))
      << text;
  EXPECT_TRUE(TextHasLineContaining(text, "%loaded = load <4 x float>, ptr ",
                                    ", align 16"))
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersAmdgpuDenseVectorLoadStore) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(1024), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("vector_dispatch"));
  loom_type_t arg_types[3] = {buffer, buffer, index};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY | LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC,
      LOOM_FUNC_VISIBILITY_PUBLIC, LOOM_FUNC_CC_DEVICE, 0, callee, arg_types, 3,
      NULL, 0, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3);
  SetValueName(args[0], IREE_SV("input"));
  SetValueName(args[1], IREE_SV("output"));
  SetValueName(args[2], IREE_SV("base"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* zero_op = NULL;
  IREE_ASSERT_OK(loom_index_constant_build(&body_builder, loom_attr_i64(0),
                                           offset, LOOM_LOCATION_UNKNOWN,
                                           &zero_op));
  loom_value_id_t zero = loom_index_constant_result(zero_op);

  loom_op_t* input_view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], zero, view_type,
                                        LOOM_LOCATION_UNKNOWN, &input_view_op));
  loom_value_id_t input_view = loom_buffer_view_result(input_view_op);
  SetValueName(input_view, IREE_SV("input_view"));

  loom_op_t* output_view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[1], zero, view_type,
                                        LOOM_LOCATION_UNKNOWN,
                                        &output_view_op));
  loom_value_id_t output_view = loom_buffer_view_result(output_view_op);
  SetValueName(output_view, IREE_SV("output_view"));

  int64_t dynamic_indices[1] = {INT64_MIN};
  loom_value_id_t indices[1] = {args[2]};
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_vector_load_build(
      &body_builder, 0, input_view, indices, IREE_ARRAYSIZE(indices),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), 0, 0, v4f32,
      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_vector_load_result(load_op);
  SetValueName(loaded, IREE_SV("loaded"));

  loom_op_t* store_op = NULL;
  IREE_ASSERT_OK(loom_vector_store_build(
      &body_builder, 0, loaded, output_view, indices, IREE_ARRAYSIZE(indices),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), 0, 0,
      LOOM_LOCATION_UNKNOWN, &store_op));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText(loom_llvmir_target_profile_amdgpu_hal());
  EXPECT_NE(text.find("define amdgpu_kernel void @vector_dispatch("),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("ptr addrspace(1) inreg noalias noundef nonnull align "
                      "16 %input"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %input_view = getelementptr i8, ptr addrspace(1) "
                      "%input, i64 0\n"),
            std::string::npos)
      << text;
  EXPECT_TRUE(TextHasLineContaining(
      text, "%loaded = load <4 x float>, ptr addrspace(1) ", ", align 4"))
      << text;
  EXPECT_TRUE(TextHasLineContaining(
      text, "store <4 x float> %loaded, ptr addrspace(1) ", ", align 4"))
      << text;
  VerifyTextWithLlvmTools(text);
  CompileAmdgpuTextToObjectIfAvailable(text);
}

TEST_F(LlvmIrLowerTest, LowersX86ScalarVectorObjectFixture) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(1024), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("saxpy4_object"));
  loom_type_t arg_types[4] = {buffer, buffer, buffer, index};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), NULL, 0, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 4);
  SetValueName(args[0], IREE_SV("x"));
  SetValueName(args[1], IREE_SV("y"));
  SetValueName(args[2], IREE_SV("out"));
  SetValueName(args[3], IREE_SV("base"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* zero_op = NULL;
  IREE_ASSERT_OK(loom_index_constant_build(&body_builder, loom_attr_i64(0),
                                           offset, LOOM_LOCATION_UNKNOWN,
                                           &zero_op));
  loom_value_id_t zero = loom_index_constant_result(zero_op);

  loom_op_t* step_op = NULL;
  IREE_ASSERT_OK(loom_index_constant_build(
      &body_builder, loom_attr_i64(4), index, LOOM_LOCATION_UNKNOWN, &step_op));
  loom_value_id_t step = loom_index_constant_result(step_op);

  loom_op_t* next_op = NULL;
  IREE_ASSERT_OK(loom_index_add_build(&body_builder, args[3], step, index,
                                      LOOM_LOCATION_UNKNOWN, &next_op));
  loom_value_id_t next = loom_index_add_result(next_op);
  SetValueName(next, IREE_SV("next"));

  loom_op_t* x_view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], zero, view_type,
                                        LOOM_LOCATION_UNKNOWN, &x_view_op));
  loom_value_id_t x_view = loom_buffer_view_result(x_view_op);
  SetValueName(x_view, IREE_SV("x_view"));

  loom_op_t* y_view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[1], zero, view_type,
                                        LOOM_LOCATION_UNKNOWN, &y_view_op));
  loom_value_id_t y_view = loom_buffer_view_result(y_view_op);
  SetValueName(y_view, IREE_SV("y_view"));

  loom_op_t* out_view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[2], zero, view_type,
                                        LOOM_LOCATION_UNKNOWN, &out_view_op));
  loom_value_id_t out_view = loom_buffer_view_result(out_view_op);
  SetValueName(out_view, IREE_SV("out_view"));

  int64_t dynamic_indices[1] = {INT64_MIN};
  loom_value_id_t base_indices[1] = {args[3]};
  loom_op_t* x_load_op = NULL;
  IREE_ASSERT_OK(loom_vector_load_build(
      &body_builder, 0, x_view, base_indices, IREE_ARRAYSIZE(base_indices),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), 0, 0, v4f32,
      LOOM_LOCATION_UNKNOWN, &x_load_op));
  loom_value_id_t xv = loom_vector_load_result(x_load_op);
  SetValueName(xv, IREE_SV("xv"));

  loom_value_id_t next_indices[1] = {next};
  loom_op_t* y_load_op = NULL;
  IREE_ASSERT_OK(loom_vector_load_build(
      &body_builder, 0, y_view, next_indices, IREE_ARRAYSIZE(next_indices),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), 0, 0, v4f32,
      LOOM_LOCATION_UNKNOWN, &y_load_op));
  loom_value_id_t yv = loom_vector_load_result(y_load_op);
  SetValueName(yv, IREE_SV("yv"));

  loom_op_t* add_op = NULL;
  IREE_ASSERT_OK(loom_vector_addf_build(
      &body_builder, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NNAN, xv, yv, v4f32,
      LOOM_LOCATION_UNKNOWN, &add_op));
  loom_value_id_t sum = loom_vector_addf_result(add_op);
  SetValueName(sum, IREE_SV("sum"));

  loom_op_t* store_op = NULL;
  IREE_ASSERT_OK(loom_vector_store_build(
      &body_builder, 0, sum, out_view, next_indices,
      IREE_ARRAYSIZE(next_indices), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), 0, 0, LOOM_LOCATION_UNKNOWN, &store_op));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("define dso_local void @saxpy4_object("
                      "ptr %x, ptr %y, ptr %out, i64 %base) {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %next = add i64 %base, 4\n"), std::string::npos)
      << text;
  EXPECT_NE(text.find("  %xv = load <4 x float>, ptr "), std::string::npos)
      << text;
  EXPECT_NE(text.find("  %yv = load <4 x float>, ptr "), std::string::npos)
      << text;
  EXPECT_NE(text.find("  %sum = fadd nnan <4 x float> %xv, %yv\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  store <4 x float> %sum, ptr "), std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersAmdgpuScalarVectorHalKernelFixture) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(1024), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("vector_hal_kernel"));
  loom_type_t arg_types[3] = {buffer, buffer, index};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY | LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC,
      LOOM_FUNC_VISIBILITY_PUBLIC, LOOM_FUNC_CC_DEVICE, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), NULL, 0, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3);
  SetValueName(args[0], IREE_SV("input"));
  SetValueName(args[1], IREE_SV("output"));
  SetValueName(args[2], IREE_SV("base"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* zero_op = NULL;
  IREE_ASSERT_OK(loom_index_constant_build(&body_builder, loom_attr_i64(0),
                                           offset, LOOM_LOCATION_UNKNOWN,
                                           &zero_op));
  loom_value_id_t zero = loom_index_constant_result(zero_op);

  loom_type_t intrinsic_result_types[1] = {i32};
  loom_op_t* tid_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.amdgcn.workitem.id.x")), NULL,
      0, intrinsic_result_types, IREE_ARRAYSIZE(intrinsic_result_types), NULL,
      0, LOOM_LOCATION_UNKNOWN, &tid_op));
  loom_value_id_t tid_x =
      loom_value_slice_get(loom_llvmir_intrinsic_results(tid_op), 0);
  SetValueName(tid_x, IREE_SV("tid_x"));

  loom_op_t* tid_index_op = NULL;
  IREE_ASSERT_OK(loom_index_cast_build(&body_builder, tid_x, i32, index,
                                       LOOM_LOCATION_UNKNOWN, &tid_index_op));
  loom_value_id_t tid_index = loom_index_cast_result(tid_index_op);
  SetValueName(tid_index, IREE_SV("tid_index"));

  loom_op_t* element_op = NULL;
  IREE_ASSERT_OK(loom_index_add_build(&body_builder, args[2], tid_index, index,
                                      LOOM_LOCATION_UNKNOWN, &element_op));
  loom_value_id_t element = loom_index_add_result(element_op);
  SetValueName(element, IREE_SV("element"));

  loom_op_t* input_view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], zero, view_type,
                                        LOOM_LOCATION_UNKNOWN, &input_view_op));
  loom_value_id_t input_view = loom_buffer_view_result(input_view_op);
  SetValueName(input_view, IREE_SV("input_view"));

  loom_op_t* output_view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[1], zero, view_type,
                                        LOOM_LOCATION_UNKNOWN,
                                        &output_view_op));
  loom_value_id_t output_view = loom_buffer_view_result(output_view_op);
  SetValueName(output_view, IREE_SV("output_view"));

  int64_t dynamic_indices[1] = {INT64_MIN};
  loom_value_id_t element_indices[1] = {element};
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(
      loom_vector_load_build(&body_builder, 0, input_view, element_indices,
                             IREE_ARRAYSIZE(element_indices), dynamic_indices,
                             IREE_ARRAYSIZE(dynamic_indices), 0, 0, v4f32,
                             LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_vector_load_result(load_op);
  SetValueName(loaded, IREE_SV("loaded"));

  loom_op_t* double_op = NULL;
  IREE_ASSERT_OK(loom_vector_addf_build(
      &body_builder, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NNAN, loaded, loaded,
      v4f32, LOOM_LOCATION_UNKNOWN, &double_op));
  loom_value_id_t doubled = loom_vector_addf_result(double_op);
  SetValueName(doubled, IREE_SV("doubled"));

  loom_op_t* store_op = NULL;
  IREE_ASSERT_OK(loom_vector_store_build(
      &body_builder, 0, doubled, output_view, element_indices,
      IREE_ARRAYSIZE(element_indices), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), 0, 0, LOOM_LOCATION_UNKNOWN, &store_op));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText(loom_llvmir_target_profile_amdgpu_hal());
  EXPECT_NE(text.find("define amdgpu_kernel void @vector_hal_kernel("),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("ptr addrspace(1) inreg noalias noundef nonnull align "
                      "16 %input"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %tid_x = call i32 @llvm.amdgcn.workitem.id.x()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %element = add i32 %base, %tid_x\n"),
            std::string::npos)
      << text;
  EXPECT_TRUE(TextHasLineContaining(
      text, "%loaded = load <4 x float>, ptr addrspace(1) ", ", align 4"))
      << text;
  EXPECT_NE(text.find("  %doubled = fadd nnan <4 x float> %loaded, %loaded\n"),
            std::string::npos)
      << text;
  EXPECT_TRUE(TextHasLineContaining(
      text, "store <4 x float> %doubled, ptr addrspace(1) ", ", align 4"))
      << text;
  EXPECT_NE(text.find("attributes #0 = { alwaysinline "
                      "\"amdgpu-flat-work-group-size\"=\"64,64\" "
                      "\"uniform-work-group-size\" }"),
            std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileAmdgpuTextToObjectIfAvailable(text);
}

TEST_F(LlvmIrLowerTest, RejectsVectorLoadWithoutDenseEncoding) {
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t v4f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("bad_vector_memory"));
  loom_type_t arg_types[2] = {buffer, offset};
  loom_type_t result_types[1] = {v4f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types, 2, result_types, 1,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2);

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], args[1],
                                        view_type, LOOM_LOCATION_UNKNOWN,
                                        &view_op));
  loom_value_id_t view = loom_buffer_view_result(view_op);

  int64_t static_index[1] = {0};
  loom_op_t* load_op = NULL;
  IREE_ASSERT_OK(loom_vector_load_build(&body_builder, 0, view, NULL, 0,
                                        static_index, 1, 0, 0, v4f32,
                                        LOOM_LOCATION_UNKNOWN, &load_op));
  loom_value_id_t loaded = loom_vector_load_result(load_op);
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &loaded, 1,
                                        LOOM_LOCATION_UNKNOWN, &load_op));

  loom_llvmir_lowering_options_t options = {};
  options.target_profile = loom_llvmir_target_profile_x86_64_object();
  options.source_name = IREE_SV("lower_test");
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = loom_llvmir_lower_module(
      module_, &options, iree_allocator_system(), &lowered);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  std::string message = StatusToString(status);
  EXPECT_NE(message.find("dense address layouts"), std::string::npos)
      << message;
  iree_status_ignore(status);
}

TEST_F(LlvmIrLowerTest, LowersViewPrefetchToBuiltinIntrinsic) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(16), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("prefetch_row"));
  loom_type_t arg_types[3] = {buffer, offset, index};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types, 3, NULL, 0, NULL, 0,
      NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3);
  SetValueName(args[0], IREE_SV("buffer"));
  SetValueName(args[1], IREE_SV("offset"));
  SetValueName(args[2], IREE_SV("index"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(loom_buffer_view_build(&body_builder, args[0], args[1],
                                        view_type, LOOM_LOCATION_UNKNOWN,
                                        &view_op));
  loom_value_id_t view = loom_buffer_view_result(view_op);
  SetValueName(view, IREE_SV("view"));

  int64_t dynamic_indices[1] = {INT64_MIN};
  loom_op_t* prefetch_op = NULL;
  IREE_ASSERT_OK(loom_view_prefetch_build(
      &body_builder, view, &args[2], 1, dynamic_indices, 1,
      LOOM_VIEW_PREFETCH_INTENT_READ, LOOM_VIEW_PREFETCH_LOCALITY_L2,
      LOOM_LOCATION_UNKNOWN, &prefetch_op));
  loom_op_t* second_prefetch_op = NULL;
  IREE_ASSERT_OK(loom_view_prefetch_build(
      &body_builder, view, &args[2], 1, dynamic_indices, 1,
      LOOM_VIEW_PREFETCH_INTENT_WRITE, LOOM_VIEW_PREFETCH_LOCALITY_L3,
      LOOM_LOCATION_UNKNOWN, &second_prefetch_op));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  const char* prefetch_decl =
      "declare void @llvm.prefetch.p0(ptr readonly nocapture, i32 immarg, "
      "i32 immarg, i32 immarg)\n";
  size_t prefetch_decl_pos = text.find(prefetch_decl);
  ASSERT_NE(prefetch_decl_pos, std::string::npos) << text;
  EXPECT_EQ(text.find(prefetch_decl, prefetch_decl_pos + 1), std::string::npos)
      << text;
  EXPECT_NE(text.find("  %view = getelementptr i8, ptr %buffer, i64 %offset\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  call void @llvm.prefetch.p0(ptr "), std::string::npos)
      << text;
  EXPECT_NE(text.find(", i32 0, i32 2, i32 1)\n"), std::string::npos) << text;
  EXPECT_NE(text.find(", i32 1, i32 3, i32 1)\n"), std::string::npos) << text;
  VerifyTextWithLlvmTools(text);
}

TEST_F(LlvmIrLowerTest, LowersSourceMemoryIntrinsics) {
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t i8 = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_type_t i64 = loom_type_scalar(LOOM_SCALAR_TYPE_I64);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("source_memory_intrinsics"));
  loom_type_t arg_types[3] = {buffer, buffer, offset};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), NULL, 0, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3);
  SetValueName(args[0], IREE_SV("target"));
  SetValueName(args[1], IREE_SV("source"));
  SetValueName(args[2], IREE_SV("length"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* fill_value_op = NULL;
  IREE_ASSERT_OK(loom_scalar_constant_build(&body_builder, loom_attr_i64(0), i8,
                                            LOOM_LOCATION_UNKNOWN,
                                            &fill_value_op));
  loom_value_id_t fill_value = loom_scalar_constant_result(fill_value_op);

  loom_op_t* is_volatile_op = NULL;
  IREE_ASSERT_OK(
      loom_scalar_constant_build(&body_builder, loom_attr_bool(false), i1,
                                 LOOM_LOCATION_UNKNOWN, &is_volatile_op));
  loom_value_id_t is_volatile = loom_scalar_constant_result(is_volatile_op);

  loom_op_t* lifetime_size_op = NULL;
  IREE_ASSERT_OK(loom_scalar_constant_build(&body_builder, loom_attr_i64(16),
                                            i64, LOOM_LOCATION_UNKNOWN,
                                            &lifetime_size_op));
  loom_value_id_t lifetime_size = loom_scalar_constant_result(lifetime_size_op);

  loom_value_id_t lifetime_args[2] = {lifetime_size, args[0]};
  loom_op_t* lifetime_start_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.lifetime.start")),
      lifetime_args, IREE_ARRAYSIZE(lifetime_args), NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &lifetime_start_op));

  loom_value_id_t memset_args[4] = {args[0], fill_value, args[2], is_volatile};
  loom_op_t* memset_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.memset")), memset_args,
      IREE_ARRAYSIZE(memset_args), NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
      &memset_op));

  loom_value_id_t memcpy_args[4] = {args[0], args[1], args[2], is_volatile};
  loom_op_t* first_memcpy_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.memcpy")), memcpy_args,
      IREE_ARRAYSIZE(memcpy_args), NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
      &first_memcpy_op));
  loom_op_t* second_memcpy_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.memcpy")), memcpy_args,
      IREE_ARRAYSIZE(memcpy_args), NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
      &second_memcpy_op));

  loom_op_t* lifetime_end_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.lifetime.end")), lifetime_args,
      IREE_ARRAYSIZE(lifetime_args), NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
      &lifetime_end_op));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText();
  const char* memcpy_decl =
      "declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, "
      "ptr noalias nocapture readonly, i64, i1 immarg)\n";
  size_t memcpy_decl_pos = text.find(memcpy_decl);
  ASSERT_NE(memcpy_decl_pos, std::string::npos) << text;
  EXPECT_EQ(text.find(memcpy_decl, memcpy_decl_pos + 1), std::string::npos)
      << text;
  EXPECT_NE(text.find("declare void @llvm.memset.p0.i64(ptr nocapture "
                      "writeonly, i8, i64, i1 immarg)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare void @llvm.lifetime.start.p0(i64 immarg, "
                      "ptr nocapture)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare void @llvm.lifetime.end.p0(i64 immarg, "
                      "ptr nocapture)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  call void @llvm.lifetime.start.p0(i64 16, "
                      "ptr %target)\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  call void @llvm.memset.p0.i64(ptr %target, i8 0, "
                      "i64 %length, i1 0)\n"),
            std::string::npos)
      << text;
  const char* memcpy_call =
      "  call void @llvm.memcpy.p0.p0.i64(ptr %target, ptr %source, "
      "i64 %length, i1 0)\n";
  size_t first_memcpy_call_pos = text.find(memcpy_call);
  ASSERT_NE(first_memcpy_call_pos, std::string::npos) << text;
  EXPECT_NE(text.find(memcpy_call, first_memcpy_call_pos + 1),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  call void @llvm.lifetime.end.p0(i64 16, "
                      "ptr %target)\n"),
            std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, RejectsMemoryIntrinsicDynamicImmarg) {
  loom_type_t buffer = loom_type_buffer();
  loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("bad_memory_intrinsic"));
  loom_type_t arg_types[4] = {buffer, buffer, offset, i1};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types,
      IREE_ARRAYSIZE(arg_types), NULL, 0, NULL, 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 4);

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_value_id_t memcpy_args[4] = {args[0], args[1], args[2], args[3]};
  loom_op_t* memcpy_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.memcpy")), memcpy_args,
      IREE_ARRAYSIZE(memcpy_args), NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
      &memcpy_op));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_llvmir_lowering_options_t options = {};
  options.target_profile = loom_llvmir_target_profile_x86_64_object();
  options.source_name = IREE_SV("lower_test");
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = loom_llvmir_lower_module(
      module_, &options, iree_allocator_system(), &lowered);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  std::string message = StatusToString(status);
  EXPECT_NE(message.find("llvm.memcpy expects"), std::string::npos) << message;
  iree_status_free(status);
  if (lowered) {
    loom_llvmir_module_free(lowered);
  }
}

TEST_F(LlvmIrLowerTest, LowersInlineAsmToStructuredLlvmIr) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("inline_asm_add"));
  loom_type_t arg_types[2] = {i32, i32};
  loom_type_t result_types[1] = {i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, arg_types, 2, result_types, 1,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2);
  SetValueName(args[0], IREE_SV("lhs"));
  SetValueName(args[1], IREE_SV("rhs"));

  loom_string_id_t asm_template = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("addl $2, $0"),
                                           &asm_template));
  loom_string_id_t constraints = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("=r,0,r"), &constraints));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* asm_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_inline_asm_build(
      &body_builder, LOOM_LLVMIR_ASMFLAGS_SIDEEFFECT, asm_template, constraints,
      args, arg_count, result_types, IREE_ARRAYSIZE(result_types), NULL, 0,
      LOOM_LOCATION_UNKNOWN, &asm_op));
  loom_value_id_t sum =
      loom_value_slice_get(loom_llvmir_inline_asm_results(asm_op), 0);
  SetValueName(sum, IREE_SV("sum"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &sum, 1,
                                        LOOM_LOCATION_UNKNOWN, &asm_op));

  std::string text = LowerToText();
  EXPECT_NE(text.find("  %sum = call i32 asm sideeffect \"addl $2, $0\", "
                      "\"=r,0,r\"(i32 %lhs, i32 %rhs)\n"),
            std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersX86IntrinsicsToStructuredCalls) {
  loom_type_t i64 = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("read_tsc"));
  loom_type_t result_types[1] = {i64};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, NULL, 0, result_types, 1, NULL,
      0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* pause_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.x86.sse2.pause")), NULL, 0,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &pause_op));
  loom_op_t* first_tsc_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.x86.rdtsc")), NULL, 0,
      result_types, IREE_ARRAYSIZE(result_types), NULL, 0,
      LOOM_LOCATION_UNKNOWN, &first_tsc_op));
  SetValueName(
      loom_value_slice_get(loom_llvmir_intrinsic_results(first_tsc_op), 0),
      IREE_SV("ignored"));
  loom_op_t* second_tsc_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.x86.rdtsc")), NULL, 0,
      result_types, IREE_ARRAYSIZE(result_types), NULL, 0,
      LOOM_LOCATION_UNKNOWN, &second_tsc_op));
  loom_value_id_t ticks =
      loom_value_slice_get(loom_llvmir_intrinsic_results(second_tsc_op), 0);
  SetValueName(ticks, IREE_SV("ticks"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &ticks, 1,
                                        LOOM_LOCATION_UNKNOWN, &pause_op));

  std::string text = LowerToText();
  const char* rdtsc_decl = "declare i64 @llvm.x86.rdtsc()\n";
  size_t rdtsc_decl_pos = text.find(rdtsc_decl);
  ASSERT_NE(rdtsc_decl_pos, std::string::npos) << text;
  EXPECT_EQ(text.find(rdtsc_decl, rdtsc_decl_pos + 1), std::string::npos)
      << text;
  EXPECT_NE(text.find("declare void @llvm.x86.sse2.pause()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  call void @llvm.x86.sse2.pause()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %ignored = call i64 @llvm.x86.rdtsc()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %ticks = call i64 @llvm.x86.rdtsc()\n"),
            std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileX86TextToObject(text);
}

TEST_F(LlvmIrLowerTest, LowersAmdgpuWorkitemIntrinsicsToStructuredCalls) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("dispatch"));
  loom_type_t result_types[1] = {i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY | LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC,
      LOOM_FUNC_VISIBILITY_PUBLIC, LOOM_FUNC_CC_DEVICE, 0, callee, NULL, 0,
      NULL, 0, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* tid_x_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.amdgcn.workitem.id.x")), NULL,
      0, result_types, IREE_ARRAYSIZE(result_types), NULL, 0,
      LOOM_LOCATION_UNKNOWN, &tid_x_op));
  loom_value_id_t tid_x =
      loom_value_slice_get(loom_llvmir_intrinsic_results(tid_x_op), 0);
  SetValueName(tid_x, IREE_SV("tid_x"));
  loom_op_t* tid_y_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.amdgcn.workitem.id.y")), NULL,
      0, result_types, IREE_ARRAYSIZE(result_types), NULL, 0,
      LOOM_LOCATION_UNKNOWN, &tid_y_op));
  loom_value_id_t tid_y =
      loom_value_slice_get(loom_llvmir_intrinsic_results(tid_y_op), 0);
  SetValueName(tid_y, IREE_SV("tid_y"));
  loom_op_t* tid_z_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.amdgcn.workitem.id.z")), NULL,
      0, result_types, IREE_ARRAYSIZE(result_types), NULL, 0,
      LOOM_LOCATION_UNKNOWN, &tid_z_op));
  loom_value_id_t tid_z =
      loom_value_slice_get(loom_llvmir_intrinsic_results(tid_z_op), 0);
  SetValueName(tid_z, IREE_SV("tid_z"));
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &tid_z_op));

  std::string text = LowerToText(loom_llvmir_target_profile_amdgpu_hal());
  EXPECT_NE(text.find("declare i32 @llvm.amdgcn.workitem.id.x()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare i32 @llvm.amdgcn.workitem.id.y()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare i32 @llvm.amdgcn.workitem.id.z()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %tid_x = call i32 @llvm.amdgcn.workitem.id.x()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %tid_y = call i32 @llvm.amdgcn.workitem.id.y()\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("  %tid_z = call i32 @llvm.amdgcn.workitem.id.z()\n"),
            std::string::npos)
      << text;
  VerifyTextWithLlvmTools(text);
  CompileAmdgpuTextToObjectIfAvailable(text);
}

TEST_F(LlvmIrLowerTest, RejectsAmdgpuIntrinsicOnX86Profile) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("bad_intrinsic_target"));
  loom_type_t result_types[1] = {i32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, callee, NULL, 0, result_types, 1, NULL,
      0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* tid_op = NULL;
  IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
      &body_builder, IntrinsicKind(IREE_SV("llvm.amdgcn.workitem.id.y")), NULL,
      0, result_types, IREE_ARRAYSIZE(result_types), NULL, 0,
      LOOM_LOCATION_UNKNOWN, &tid_op));
  loom_value_id_t tid =
      loom_value_slice_get(loom_llvmir_intrinsic_results(tid_op), 0);
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &tid, 1,
                                        LOOM_LOCATION_UNKNOWN, &tid_op));

  const loom_llvmir_lowering_provider_t* providers[] = {
      loom_llvmir_amdgpu_lowering_provider(),
  };
  loom_llvmir_lowering_options_t options = {};
  options.target_profile = loom_llvmir_target_profile_x86_64_object();
  options.source_name = IREE_SV("lower_test");
  options.providers = providers;
  options.provider_count = IREE_ARRAYSIZE(providers);
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = loom_llvmir_lower_module(
      module_, &options, iree_allocator_system(), &lowered);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  std::string message = StatusToString(status);
  EXPECT_NE(message.find("AMDGPU llvmir.intrinsic requires an AMDGPU target"),
            std::string::npos)
      << message;
  iree_status_free(status);
  if (lowered) {
    loom_llvmir_module_free(lowered);
  }
}

TEST_F(LlvmIrLowerTest, LowersPublicDeviceFunctionWithAmdgpuHalAbi) {
  loom_type_t buffer = loom_type_buffer();
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("dispatch"));
  loom_type_t arg_types[3] = {buffer, buffer, index};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY | LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC,
      LOOM_FUNC_VISIBILITY_PUBLIC, LOOM_FUNC_CC_DEVICE, 0, callee, arg_types, 3,
      NULL, 0, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3);
  SetValueName(args[0], IREE_SV("input"));
  SetValueName(args[1], IREE_SV("output"));
  SetValueName(args[2], IREE_SV("n"));

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  std::string text = LowerToText(loom_llvmir_target_profile_amdgpu_hal());
  EXPECT_NE(text.find("target triple = \"amdgcn-amd-amdhsa\"\n"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("define amdgpu_kernel void @dispatch("
                      "ptr addrspace(1) inreg noalias noundef nonnull align "
                      "16 %input, "
                      "ptr addrspace(1) inreg noalias noundef nonnull align "
                      "16 %output, i32 %n) #0 !reqd_work_group_size !0 {"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("attributes #0 = { alwaysinline "
                      "\"amdgpu-flat-work-group-size\"=\"64,64\" "
                      "\"uniform-work-group-size\" }"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("!0 = !{i32 64, i32 1, i32 1}\n"), std::string::npos)
      << text;
  CompileAmdgpuTextToObjectIfAvailable(text);
}

TEST_F(LlvmIrLowerTest, RejectsAmdgpuHalKernelViewParameter) {
  uint16_t dense_encoding = AddDenseEncoding();
  loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(1), dense_encoding);

  loom_symbol_ref_t callee = MakeSymbol(IREE_SV("bad_view_dispatch"));
  loom_type_t arg_types[1] = {view_type};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY | LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC,
      LOOM_FUNC_VISIBILITY_PUBLIC, LOOM_FUNC_CC_DEVICE, 0, callee, arg_types, 1,
      NULL, 0, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_llvmir_lowering_options_t options = {};
  options.target_profile = loom_llvmir_target_profile_amdgpu_hal();
  options.source_name = IREE_SV("lower_test");
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = loom_llvmir_lower_module(
      module_, &options, iree_allocator_system(), &lowered);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  std::string message = StatusToString(status);
  EXPECT_NE(message.find("view parameters need an explicit ABI adapter"),
            std::string::npos)
      << message;
  iree_status_free(status);
  if (lowered) {
    loom_llvmir_module_free(lowered);
  }
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

  loom_llvmir_lowering_options_t options = {};
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
