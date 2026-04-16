// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/varint.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* error_ids = static_cast<std::vector<std::string>*>(user_data);
  error_ids->push_back(diagnostic->error->error_id);
  return iree_ok_status();
}

class ReaderTest : public ::testing::Test {
 protected:
  struct SectionEntry {
    // Section kind from loom_bytecode_section_kind_t.
    uint16_t kind = 0;
    // Byte offset of this entry in the module section directory.
    size_t directory_entry_offset = 0;
    // Byte offset from the start of the module.
    uint64_t offset = 0;
    // Byte length of the section payload.
    uint64_t length = 0;
  };

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* CreateModule(const char* name) {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, iree_make_cstring_view(name),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module));
    return module;
  }

  loom_module_t* CreateFunctionModule() {
    loom_module_t* module = CreateModule("reader_func");
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_CHECK_OK(loom_module_intern_type(module, i32_type, &i32_type));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    loom_type_t arg_types[1] = {i32_type};
    loom_type_t result_types[1] = {i32_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;
    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    if (arg_count != 1) {
      ADD_FAILURE() << "expected one function argument";
      return module;
    }
    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_op_t* addi_op = nullptr;
    IREE_CHECK_OK(loom_test_addi_build(&body_builder, arg_ids[0], arg_ids[0],
                                       i32_type, LOOM_LOCATION_UNKNOWN,
                                       &addi_op));
    loom_value_id_t addi_result = loom_test_addi_result(addi_op);
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, &addi_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  loom_module_t* CreateLocatedModule() {
    loom_module_t* module = CreateModule("located");
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_CHECK_OK(loom_context_register_source(&context_, IREE_SV("model.loom"),
                                               &source_id));
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(
        module, loom_location_file_range(source_id, 1, 1, 1, 2), &location_id));
    return module;
  }

  std::vector<uint8_t> WriteModule(
      const loom_module_t* module,
      const loom_bytecode_write_options_t* options = nullptr) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, options, &block_pool_));

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  loom_bytecode_read_result_t ReadMetadata(
      const std::vector<uint8_t>& bytes, loom_context_t* context,
      std::vector<std::string>* error_ids) {
    loom_bytecode_read_result_t result = {0};
    loom_bytecode_read_options_t options = {
        .diagnostic_sink =
            {
                .fn = CaptureDiagnostic,
                .user_data = error_ids,
            },
    };
    IREE_CHECK_OK(loom_bytecode_read_metadata(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("test.loombc"), context, &block_pool_, &options, &result));
    return result;
  }

  loom_bytecode_read_result_t ReadMetadata(
      const std::vector<uint8_t>& bytes, std::vector<std::string>* error_ids) {
    return ReadMetadata(bytes, &context_, error_ids);
  }

  loom_bytecode_read_result_t ReadModule(const std::vector<uint8_t>& bytes,
                                         loom_context_t* context,
                                         loom_module_t** out_module,
                                         std::vector<std::string>* error_ids,
                                         bool verify_module = false) {
    loom_bytecode_read_result_t result = {0};
    loom_bytecode_read_options_t options = {
        .diagnostic_sink =
            {
                .fn = CaptureDiagnostic,
                .user_data = error_ids,
            },
        .verify_module = verify_module,
    };
    IREE_CHECK_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("test.loombc"), context, &block_pool_, &options, &result,
        out_module, iree_allocator_system()));
    return result;
  }

  loom_bytecode_read_result_t ReadModule(const std::vector<uint8_t>& bytes,
                                         loom_module_t** out_module,
                                         std::vector<std::string>* error_ids,
                                         bool verify_module = false) {
    return ReadModule(bytes, &context_, out_module, error_ids, verify_module);
  }

  uint16_t ReadU16LE(const std::vector<uint8_t>& bytes, size_t offset) {
    return (uint16_t)bytes[offset] | ((uint16_t)bytes[offset + 1] << 8);
  }

  uint64_t ReadU64LE(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value |= (uint64_t)bytes[offset + i] << (i * 8);
    }
    return value;
  }

  void WriteU16LE(std::vector<uint8_t>* bytes, size_t offset, uint16_t value) {
    (*bytes)[offset] = (uint8_t)value;
    (*bytes)[offset + 1] = (uint8_t)(value >> 8);
  }

  void WriteU64LE(std::vector<uint8_t>* bytes, size_t offset, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      (*bytes)[offset + i] = (uint8_t)(value >> (i * 8));
    }
  }

  uint64_t ReadUVarint(const std::vector<uint8_t>& bytes, size_t* offset) {
    loom_bytecode_cursor_t cursor;
    loom_bytecode_cursor_initialize(bytes.data() + *offset,
                                    bytes.size() - *offset, &cursor);
    uint64_t value = 0;
    IREE_CHECK_OK(loom_uvarint_decode(&cursor, &value));
    *offset += cursor.position;
    return value;
  }

  size_t FileHeaderEnd(const std::vector<uint8_t>& bytes) {
    size_t offset = 16;
    while (offset < bytes.size() && bytes[offset] != 0) {
      ++offset;
    }
    ++offset;
    return (offset + 7) & ~(size_t)7;
  }

  size_t ModuleDirectoryOffset(const std::vector<uint8_t>& bytes) {
    return FileHeaderEnd(bytes);
  }

  uint64_t ModuleOffset(const std::vector<uint8_t>& bytes) {
    return ReadU64LE(bytes, ModuleDirectoryOffset(bytes) + 8);
  }

  std::vector<SectionEntry> ReadSectionDirectory(
      const std::vector<uint8_t>& bytes) {
    uint64_t module_offset = ModuleOffset(bytes);
    size_t section_offset = (size_t)module_offset;
    uint64_t section_count = ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);

    std::vector<SectionEntry> entries;
    entries.reserve((size_t)section_count);
    for (uint64_t i = 0; i < section_count; ++i) {
      entries.push_back(SectionEntry{
          .kind = ReadU16LE(bytes, section_offset),
          .directory_entry_offset = section_offset,
          .offset = ReadU64LE(bytes, section_offset + 8),
          .length = ReadU64LE(bytes, section_offset + 16),
      });
      section_offset += sizeof(loom_bytecode_section_dir_entry_t);
    }
    return entries;
  }

  SectionEntry FindSection(const std::vector<uint8_t>& bytes, uint16_t kind) {
    for (SectionEntry entry : ReadSectionDirectory(bytes)) {
      if (entry.kind == kind) return entry;
    }
    return SectionEntry{};
  }

  size_t SectionPayloadOffset(const std::vector<uint8_t>& bytes,
                              uint16_t kind) {
    SectionEntry entry = FindSection(bytes, kind);
    return (size_t)ModuleOffset(bytes) + (size_t)entry.offset;
  }

  size_t FirstBodyOperandRefOffset(const std::vector<uint8_t>& bytes) {
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_IR);
    ReadUVarint(bytes, &offset);  // value_count
    ReadUVarint(bytes, &offset);  // region_count
    ReadUVarint(bytes, &offset);  // block_count
    ReadUVarint(bytes, &offset);  // op_count
    uint64_t root_block_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(root_block_count, 1u);
    uint8_t has_label = bytes[offset++];
    if (has_label) {
      ReadUVarint(bytes, &offset);
    }
    uint64_t arg_count = ReadUVarint(bytes, &offset);
    for (uint64_t i = 0; i < arg_count; ++i) {
      ReadUVarint(bytes, &offset);  // name_id
      ReadUVarint(bytes, &offset);  // type_id
      uint64_t dim_binding_count = ReadUVarint(bytes, &offset);
      for (uint64_t j = 0; j < dim_binding_count; ++j) {
        ReadUVarint(bytes, &offset);
      }
      ReadUVarint(bytes, &offset);  // encoding_binding
    }
    uint64_t op_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(op_count, 1u);
    ReadUVarint(bytes, &offset);  // op_table_index_plus1
    ++offset;                     // flags
    ReadUVarint(bytes, &offset);  // location_id
    uint64_t operand_count = ReadUVarint(bytes, &offset);
    EXPECT_GE(operand_count, 1u);
    return offset;
  }

  void ReplaceBytes(std::vector<uint8_t>* bytes, const char* from,
                    const char* to) {
    size_t length = std::strlen(from);
    ASSERT_EQ(length, std::strlen(to));
    auto it = std::search(bytes->begin(), bytes->end(), from, from + length);
    ASSERT_NE(it, bytes->end());
    std::copy(to, to + length, it);
  }

  void ExpectReadError(const std::vector<uint8_t>& bytes,
                       const char* expected_error_id) {
    std::vector<std::string> error_ids;
    loom_bytecode_read_result_t result = ReadMetadata(bytes, &error_ids);
    EXPECT_GT(result.error_count, 0u);
    ASSERT_FALSE(error_ids.empty());
    EXPECT_EQ(error_ids.front(), expected_error_id);
  }

  void ExpectReadModuleError(const std::vector<uint8_t>& bytes,
                             const char* expected_error_id) {
    std::vector<std::string> error_ids;
    loom_module_t* module = nullptr;
    loom_bytecode_read_result_t result = ReadModule(bytes, &module, &error_ids);
    EXPECT_GT(result.error_count, 0u);
    EXPECT_EQ(module, nullptr);
    ASSERT_FALSE(error_ids.empty());
    EXPECT_EQ(error_ids.front(), expected_error_id);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(ReaderTest, AcceptsEmptyModuleMetadata) {
  loom_module_t* module = CreateModule("empty");
  auto bytes = WriteModule(module);

  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result = ReadMetadata(bytes, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  EXPECT_EQ(result.module_count, 1u);
  EXPECT_EQ(result.location_mode, LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS);
  EXPECT_EQ(result.first_module.string_count, 1u);
  EXPECT_EQ(result.first_module.symbol_count, 0u);

  loom_module_free(module);
}

TEST_F(ReaderTest, AcceptsFunctionMetadata) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);

  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result = ReadMetadata(bytes, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  EXPECT_EQ(result.first_module.symbol_count, 1u);
  EXPECT_GT(result.first_module.type_count, 0u);
  EXPECT_GT(result.first_module.op_name_count, 0u);

  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsFunctionBodyModule) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_module, &error_ids,
                 /*verify_module=*/true);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_module->symbols.count, 1u);
  EXPECT_EQ(read_module->symbols.entries[0].kind, LOOM_SYMBOL_FUNC_DEF);
  ASSERT_NE(read_module->symbols.entries[0].defining_op, nullptr);
  loom_op_t* func_op = read_module->symbols.entries[0].defining_op;
  ASSERT_TRUE(loom_test_func_isa(func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 1u);
  loom_block_t* entry = loom_region_entry_block(body);
  ASSERT_EQ(entry->arg_count, 1u);
  ASSERT_EQ(entry->op_count, 2u);
  loom_op_t* body_op = entry->first_op;
  ASSERT_NE(body_op, nullptr);
  EXPECT_TRUE(loom_test_addi_isa(body_op));
  EXPECT_EQ(loom_test_addi_lhs(body_op), loom_test_addi_rhs(body_op));
  ASSERT_NE(entry->last_op, nullptr);
  EXPECT_TRUE(loom_test_yield_isa(entry->last_op));

  loom_module_free(read_module);
  loom_module_free(module);
}

TEST_F(ReaderTest, ReadsLocationTablesWithRemappedSources) {
  loom_module_t* module = CreateLocatedModule();
  auto bytes = WriteModule(module);

  loom_context_t read_context;
  IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                     &read_context));
  loom_source_id_t preexisting_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_context_register_source(
      &read_context, IREE_SV("preexisting.loom"), &preexisting_source_id));

  loom_module_t* read_module = nullptr;
  std::vector<std::string> error_ids;
  loom_bytecode_read_result_t result =
      ReadModule(bytes, &read_context, &read_module, &error_ids);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(error_ids.empty());
  ASSERT_NE(read_module, nullptr);
  ASSERT_EQ(read_context.sources.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(read_context.sources.entries[1],
                                     IREE_SV("model.loom")));
  ASSERT_EQ(read_module->locations.count, 2u);
  const loom_location_entry_t& file_location =
      read_module->locations.entries[1];
  EXPECT_EQ(file_location.kind, LOOM_LOCATION_FILE);
  EXPECT_EQ(file_location.file.source_id, 1u);
  EXPECT_EQ(file_location.file.start_line, 1u);
  EXPECT_EQ(file_location.file.end_col, 2u);

  loom_module_free(read_module);
  loom_context_deinitialize(&read_context);
  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidBodyValueReference) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  size_t operand_offset = FirstBodyOperandRefOffset(bytes);
  bytes[operand_offset] = 0x7F;

  ExpectReadModuleError(bytes, "ERR_BYTECODE_016");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsTruncatedHeader) {
  std::vector<uint8_t> bytes(8, 0);

  ExpectReadError(bytes, "ERR_BYTECODE_003");
}

TEST_F(ReaderTest, RejectsInvalidMagic) {
  loom_module_t* module = CreateModule("magic");
  auto bytes = WriteModule(module);
  bytes[0] = 0;

  ExpectReadError(bytes, "ERR_BYTECODE_001");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnsupportedVersion) {
  loom_module_t* module = CreateModule("version");
  auto bytes = WriteModule(module);
  bytes[4] = LOOM_BYTECODE_FORMAT_VERSION + 1;

  ExpectReadError(bytes, "ERR_BYTECODE_002");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsFullLocationsModeUntilFieldSpansExist) {
  loom_module_t* module = CreateModule("full");
  auto bytes = WriteModule(module);
  bytes[5] = LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsModuleRangeBeforeMetadataEnd) {
  loom_module_t* module = CreateModule("range");
  auto bytes = WriteModule(module);
  WriteU64LE(&bytes, ModuleDirectoryOffset(bytes) + 8, 0);

  ExpectReadError(bytes, "ERR_BYTECODE_007");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsDuplicateSectionKind) {
  loom_module_t* module = CreateModule("sections");
  auto bytes = WriteModule(module);
  auto sections = ReadSectionDirectory(bytes);
  ASSERT_GE(sections.size(), 2u);
  WriteU16LE(&bytes, sections[1].directory_entry_offset, sections[0].kind);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnsortedSectionRange) {
  loom_module_t* module = CreateModule("sections");
  auto bytes = WriteModule(module);
  auto sections = ReadSectionDirectory(bytes);
  ASSERT_GE(sections.size(), 2u);
  WriteU64LE(&bytes, sections[1].directory_entry_offset + 8, 0);

  ExpectReadError(bytes, "ERR_BYTECODE_007");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsMissingRequiredSection) {
  loom_module_t* module = CreateModule("sections");
  auto bytes = WriteModule(module);
  SectionEntry strings = FindSection(bytes, LOOM_BYTECODE_SECTION_STRINGS);
  ASSERT_NE(strings.length, 0u);
  WriteU16LE(&bytes, strings.directory_entry_offset,
             LOOM_BYTECODE_SECTION_RESOURCES);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidUtf8StringPayload) {
  loom_module_t* module = CreateModule("utf8");
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_STRINGS);
  ReadUVarint(bytes, &offset);
  uint64_t first_length = ReadUVarint(bytes, &offset);
  ASSERT_GT(first_length, 0u);
  bytes[offset] = 0xFF;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidOpNameStringReference) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_OPS);
  uint64_t op_count = ReadUVarint(bytes, &offset);
  ASSERT_GT(op_count, 0u);
  bytes[offset] = 0x7F;

  ExpectReadError(bytes, "ERR_BYTECODE_010");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnknownOpName) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  ReplaceBytes(&bytes, "test.addi", "bogus.add");

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsUnknownEncodingFamily) {
  loom_context_t permissive_context;
  loom_context_initialize(iree_allocator_system(), &permissive_context);
  IREE_ASSERT_OK(loom_context_finalize(&permissive_context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&permissive_context, IREE_SV("encoding"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &module));
  loom_string_id_t encoding_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("mystery"), &encoding_name));
  loom_encoding_t encoding = {
      .name_id = encoding_name,
      .alias_id = LOOM_STRING_ID_INVALID,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
  auto bytes = WriteModule(module);

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
  loom_context_deinitialize(&permissive_context);
}

TEST_F(ReaderTest, RejectsNoLocationsHeaderWithLocationsSection) {
  loom_module_t* module = CreateModule("locations");
  auto bytes = WriteModule(module);
  bytes[5] = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsSourceLocationsHeaderWithoutLocationsSection) {
  loom_module_t* module = CreateModule("locations");
  loom_bytecode_write_options_t options = {{0}};
  options.location_mode = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS;
  auto bytes = WriteModule(module, &options);
  bytes[5] = LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS;

  ExpectReadError(bytes, "ERR_BYTECODE_006");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidLocationTableReference) {
  loom_module_t* module = CreateLocatedModule();
  auto bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_LOCATIONS);
  uint64_t location_count = ReadUVarint(bytes, &offset);
  ASSERT_GE(location_count, 2u);
  offset += 2;  // Entry 0: kind=NONE, flags=0.
  ASSERT_EQ(bytes[offset], LOOM_LOCATION_FILE);
  offset += 2;  // Entry 1: kind=FILE, flags=0.
  bytes[offset] = 0x7F;

  ExpectReadError(bytes, "ERR_BYTECODE_012");

  loom_module_free(module);
}

TEST_F(ReaderTest, RejectsInvalidSymbolOffsetTableReference) {
  loom_module_t* module = CreateFunctionModule();
  auto bytes = WriteModule(module);
  SectionEntry symbols = FindSection(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  ASSERT_NE(symbols.length, 0u);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  uint64_t symbol_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(symbol_count, 1u);
  uint64_t import_count = ReadUVarint(bytes, &offset);
  uint64_t export_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(import_count, 0u);
  ASSERT_EQ(export_count, 1u);
  WriteU64LE(&bytes, offset, symbols.length);

  ExpectReadError(bytes, "ERR_BYTECODE_007");

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
