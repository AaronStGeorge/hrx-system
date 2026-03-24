// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/json_output.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/util/stream.h"

namespace {

class JsonOutputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_string_builder_initialize(iree_allocator_system(), &output_);
  }

  void TearDown() override {
    iree_string_builder_deinitialize(&output_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Parses a check file from source. Arena-allocated.
  loom_check_file_t Parse(const char* source) {
    iree_string_view_t source_view = iree_make_cstring_view(source);
    loom_check_file_t file = {0};
    IREE_EXPECT_OK(loom_check_parse(source_view, &arena_, &file));
    return file;
  }

  // Creates a result with the given outcomes and optional detail text.
  loom_check_result_t MakeResult(loom_check_outcome_t raw,
                                 loom_check_outcome_t final_outcome,
                                 const char* detail = "") {
    loom_check_result_t result;
    loom_check_result_initialize(iree_allocator_system(), &result);
    result.raw_outcome = raw;
    result.final_outcome = final_outcome;
    if (detail[0] != '\0') {
      IREE_EXPECT_OK(
          iree_string_builder_append_cstring(&result.detail, detail));
    }
    return result;
  }

  // Writes JSON for the given file and results, returns the output string.
  std::string WriteJson(iree_string_view_t filename,
                        const loom_check_file_t& file,
                        const loom_check_result_t* results,
                        iree_host_size_t pass_count,
                        iree_host_size_t fail_count,
                        iree_host_size_t skip_count) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&output_, &stream);
    IREE_EXPECT_OK(loom_check_json_write_file_result(
        filename, &file, results, pass_count, fail_count, skip_count, &stream));
    return std::string(iree_string_builder_buffer(&output_),
                       iree_string_builder_size(&output_));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  iree_string_builder_t output_;
};

TEST_F(JsonOutputTest, SinglePassingCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("test.loom"), file, results, 1, 0, 0);

  EXPECT_NE(json.find("\"file\": \"test.loom\""), std::string::npos);
  EXPECT_NE(json.find("\"default_mode\": \"roundtrip\""), std::string::npos);
  EXPECT_NE(json.find("\"index\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"mode\": \"roundtrip\""), std::string::npos);
  EXPECT_NE(json.find("\"xfail\": false"), std::string::npos);
  EXPECT_NE(json.find("\"raw_outcome\": \"pass\""), std::string::npos);
  EXPECT_NE(json.find("\"final_outcome\": \"pass\""), std::string::npos);
  EXPECT_NE(json.find("\"detail\": \"\""), std::string::npos);
  EXPECT_NE(json.find("\"total\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"passed\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 0"), std::string::npos);
  EXPECT_NE(json.find("\"skipped\": 0"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, SingleFailingCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "mismatch on line 2\n"),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("fail.loom"), file, results, 0, 1, 0);

  EXPECT_NE(json.find("\"final_outcome\": \"fail\""), std::string::npos);
  EXPECT_NE(json.find("\"total\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 1"), std::string::npos);
  // Detail should contain the message (escaped).
  EXPECT_NE(json.find("mismatch on line 2"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, XfailCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "// XFAIL: known bug\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 1u);
  ASSERT_TRUE(file.cases[0].xfail);

  // Raw outcome is fail, but final is pass due to xfail inversion.
  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_PASS),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("xfail.loom"), file, results, 1, 0, 0);

  EXPECT_NE(json.find("\"xfail\": true"), std::string::npos);
  EXPECT_NE(json.find("\"xfail_reason\": \"known bug\""), std::string::npos);
  EXPECT_NE(json.find("\"raw_outcome\": \"fail\""), std::string::npos);
  EXPECT_NE(json.find("\"final_outcome\": \"pass\""), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, VerifyMode) {
  auto file = Parse(
      "// RUN: verify\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("verify.loom"), file, results, 1, 0, 0);

  EXPECT_NE(json.find("\"default_mode\": \"verify\""), std::string::npos);
  EXPECT_NE(json.find("\"mode\": \"verify\""), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, MultipleCases) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_result_t results[2] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "diff output\n"),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("multi.loom"), file, results, 1, 1, 0);

  EXPECT_NE(json.find("\"index\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"index\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"total\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"passed\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 1"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
  loom_check_result_deinitialize(&results[1]);
}

TEST_F(JsonOutputTest, EscapedStrings) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  // Detail contains characters that need JSON escaping.
  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL,
                 "line 1: \"expected\" vs \"actual\"\nline 2: tab\there\n"),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("escape.loom"), file, results, 0, 1, 0);

  // Quotes and newlines should be escaped in the JSON.
  EXPECT_NE(json.find("\\\"expected\\\""), std::string::npos);
  EXPECT_NE(json.find("\\n"), std::string::npos);
  EXPECT_NE(json.find("\\t"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, FilenameWithSpecialCharacters) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  // Path with backslash (Windows-style) should be escaped.
  std::string json = WriteJson(iree_make_cstring_view("C:\\tests\\test.loom"),
                               file, results, 1, 0, 0);

  EXPECT_NE(json.find("C:\\\\tests\\\\test.loom"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmptyCaseArray) {
  // A file with only a preamble and no cases produces zero-count output.
  loom_check_file_t file = {0};
  file.default_mode = LOOM_CHECK_MODE_ROUNDTRIP;

  std::string json = WriteJson(iree_make_cstring_view("empty.loom"), file,
                               /*results=*/nullptr, 0, 0, 0);

  EXPECT_NE(json.find("\"cases\": [\n  ]"), std::string::npos);
  EXPECT_NE(json.find("\"total\": 0"), std::string::npos);
}

}  // namespace
