// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/chat.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/program.h"

namespace {

class ChatResponseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &output_);
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls_);
  }
  void TearDown() override {
    loom_serve_text_tool_calls_deinitialize(&calls_);
    iree_string_builder_deinitialize(&output_);
  }
  std::string Output() const {
    const auto view = iree_string_builder_view(&output_);
    return std::string(view.data, view.size);
  }

  // Wire metadata is borrowed only while each encoder runs.
  loom_serve_text_chat_response_t response_ = {
      .model_name = IREE_SV("local-λ"),
      .request_id = 7,
      .created_unix_seconds = 12,
      .usage = {.input_tokens = 120, .cached_tokens = 100, .output_tokens = 4},
  };
  // Owned output from the production JSON/SSE encoder.
  iree_string_builder_t output_;
  // Validated generated records consumed directly by the production encoder.
  loom_serve_text_tool_calls_t calls_;
};

TEST_F(ChatResponseTest, CompleteTextIncludesUsageAndEscapesContent) {
  IREE_ASSERT_OK(loom_serve_text_chat_response(
      &response_, IREE_SV("a\n\"λ\""), iree_string_view_empty(), nullptr,
      IREE_SV("stop"), &output_));
  EXPECT_EQ(
      Output(),
      R"({"id":"loom-7","object":"chat.completion","created":12,"model":"local-λ","choices":[{"index":0,"message":{"role":"assistant","content":"a\n\"λ\""},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":120,"completion_tokens":4,"total_tokens":124,"prompt_tokens_details":{"cached_tokens":100}}})");
}

TEST_F(ChatResponseTest, ToolOnlyMessageHasNullContent) {
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(&calls_, 7, IREE_SV("read"),
                                                   IREE_SV(R"({"path":"a"})")));
  IREE_ASSERT_OK(loom_serve_text_chat_response(
      &response_, iree_string_view_empty(), iree_string_view_empty(), &calls_,
      IREE_SV("tool_calls"), &output_));
  EXPECT_EQ(
      Output(),
      R"({"id":"loom-7","object":"chat.completion","created":12,"model":"local-λ","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_7_0","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},"finish_reason":"tool_calls","logprobs":null}],"usage":{"prompt_tokens":120,"completion_tokens":4,"total_tokens":124,"prompt_tokens_details":{"cached_tokens":100}}})");
}

TEST_F(ChatResponseTest, ReasoningIsSeparateFromAnswerAndToolFields) {
  IREE_ASSERT_OK(loom_serve_text_chat_response(
      &response_, IREE_SV("4"), IREE_SV("check \"λ\"\n"), nullptr,
      IREE_SV("stop"), &output_));
  EXPECT_NE(
      Output().find(R"("content":"4","reasoning_content":"check \"λ\"\n")"),
      std::string::npos);
}

TEST_F(ChatResponseTest, StreamingUsageIsNullUntilTheUsageOnlyChunk) {
  response_.flags = LOOM_SERVE_TEXT_CHAT_FLAG_STREAM |
                    LOOM_SERVE_TEXT_CHAT_FLAG_INCLUDE_USAGE;
  IREE_ASSERT_OK(
      loom_serve_text_chat_event(&response_, IREE_SV(R"({"content":"a"})"),
                                 iree_string_view_empty(), &output_));
  IREE_ASSERT_OK(loom_serve_text_chat_event(&response_, IREE_SV("{}"),
                                            IREE_SV("length"), &output_));
  IREE_ASSERT_OK(loom_serve_text_chat_usage(&response_, &output_));
  EXPECT_EQ(
      Output(),
      "data: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null,"logprobs":null}],"usage":null})"
      "\n\ndata: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[{"index":0,"delta":{},"finish_reason":"length","logprobs":null}],"usage":null})"
      "\n\ndata: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[],"usage":{"prompt_tokens":120,"completion_tokens":4,"total_tokens":124,"prompt_tokens_details":{"cached_tokens":100}}})"
      "\n\n");
}

TEST_F(ChatResponseTest, StreamingWithoutUsageOmitsTheField) {
  response_.flags = LOOM_SERVE_TEXT_CHAT_FLAG_STREAM;
  IREE_ASSERT_OK(
      loom_serve_text_chat_event(&response_, IREE_SV(R"({"role":"assistant"})"),
                                 iree_string_view_empty(), &output_));
  EXPECT_EQ(
      Output(),
      "data: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}]})"
      "\n\n");
}

TEST_F(ChatResponseTest, UnixSecondsAreNotScaledOrNarrowed) {
  for (int64_t timestamp :
       {INT64_C(0), INT64_C(2147483648), INT64_C(4294967297)}) {
    SCOPED_TRACE(timestamp);
    response_.created_unix_seconds = timestamp;
    iree_string_builder_reset(&output_);
    IREE_ASSERT_OK(loom_serve_text_chat_event(&response_, IREE_SV("{}"),
                                              IREE_SV("stop"), &output_));
    EXPECT_EQ(
        Output(),
        "data: "
        R"({"id":"loom-7","object":"chat.completion.chunk","created":)" +
            std::to_string(timestamp) +
            R"(,"model":"local-λ","choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}]})"
            "\n\n");
  }
}

class ChatToolsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(host_allocator_, &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &types_));
    IREE_ASSERT_OK(loom_serve_text_chat_tools_module_create(
        environment_, host_allocator_, &module_));
    IREE_ASSERT_OK(loom_serve_program_link(module_, {nullptr, 0},
                                           host_allocator_, &program_));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        loom_serve_program_process(program_), IREE_SV("tools"),
        IREE_SV("parse_xml"), &parse_));
  }

  void TearDown() override {
    loom_serve_program_free(program_);
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
  }

  iree_vm_variant_t Text(iree_string_view_t text) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(text.data, text.size), 1,
                             host_allocator_, &buffer));
    return iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  }

  iree_status_t Parse(iree_string_view_t text, std::string* out_calls) {
    out_calls->clear();
    iree_vm_variant_t arguments[] = {
        Text(IREE_SV(R"([{"type":"function","function":{"name":"search",
          "parameters":{"type":"object","properties":{
            "query":{"type":"string"},"limit":{"type":"integer"}},
            "required":["query"]}}}])")),
        Text(text),
    };
    iree_vm_variant_t result = iree_vm_variant_empty();
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(program_), parse_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_ptr(&result, 1));
    if (iree_status_is_ok(status)) {
      iree_vm_buffer_t* buffer = nullptr;
      status =
          iree_vm_buffer_ptr_from_variant_borrowed(&types_, result, &buffer);
      iree_const_byte_span_t bytes = iree_const_byte_span_empty();
      if (iree_status_is_ok(status)) {
        status = iree_vm_buffer_map_read(buffer, 0,
                                         iree_vm_buffer_length(buffer), &bytes);
      }
      if (iree_status_is_ok(status)) {
        out_calls->assign(reinterpret_cast<const char*>(bytes.data),
                          bytes.data_length);
      }
    } else {
      EXPECT_TRUE(iree_vm_variant_is_empty(result));
    }
    iree_vm_variant_reset(&result);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    return status;
  }

  // Allocation policy for the real native module and its invocation.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Type owner outliving the module, program, and result buffers.
  iree_vm_environment_t* environment_ = nullptr;
  // Canonical identities used to exchange UTF-8 buffers with the native codec.
  iree_vm_ref_types_t types_ = {};
  // Production schema-aware codec, retained by the linked program.
  iree_vm_module_t* module_ = nullptr;
  // Shared invocation reused after both successful and failed completions.
  loom_serve_program_t* program_ = nullptr;
  // Native export resolved from the linked program.
  iree_vm_function_t parse_ = {};
};

TEST_F(ChatToolsTest, GeneratedArgumentsPreserveDeclaredTypesAndStringContent) {
  std::string calls;
  IREE_ASSERT_OK(
      Parse(IREE_SV("<tool_call><function=search>"
                    "<parameter=query>\n\"λ\"\n</parameter>"
                    "<parameter=limit>3</parameter></function></tool_call>"),
            &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"search","arguments":"{\"query\":\"\\\"λ\\\"\",\"limit\":3}"}}])");
}

TEST_F(ChatToolsTest, InvalidLaterCallPublishesNothingAndInvocationIsReusable) {
  const std::string valid =
      "<tool_call><function=search><parameter=query>kept</parameter>"
      "</function></tool_call>";
  struct Failure {
    // Invalid second call following an otherwise valid first call.
    const char* text;
    // Diagnostic identifying the generated tool and invalid argument.
    const char* message;
  };
  const Failure failures[] = {
      {"<tool_call><function=search><parameter=q>query</parameter>"
       "</function></tool_call>",
       "generated tool 'search' uses undeclared parameter 'q'"},
      {"<tool_call><function=search><parameter=query>first</parameter>"
       "<parameter=query>second</parameter></function></tool_call>",
       "generated tool 'search' repeats parameter 'query'"},
      {"<tool_call><function=undeclared></function></tool_call>",
       "generated tool 'undeclared' is not declared"},
  };
  for (const auto& failure : failures) {
    SCOPED_TRACE(failure.text);
    const std::string text = valid + failure.text;
    std::string calls;
    auto status =
        Parse(iree_make_string_view(text.data(), text.size()), &calls);
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
    const auto message = iree_status_message(status);
    EXPECT_EQ(std::string(message.data, message.size), failure.message);
    iree_status_free(status);
    EXPECT_TRUE(calls.empty());
    IREE_ASSERT_OK(
        Parse(iree_make_string_view(valid.data(), valid.size()), &calls));
    EXPECT_EQ(
        calls,
        R"([{"type":"function","function":{"name":"search","arguments":"{\"query\":\"kept\"}"}}])");
  }
}

}  // namespace
