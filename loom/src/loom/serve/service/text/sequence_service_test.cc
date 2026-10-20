// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/base64.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/stream.h"

namespace {

using loom_serve::testing::text::SourceTextServiceTest;

class SourceSequenceServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.mode = LOOM_SERVE_TEXT_MODE_ENCODE;
    options.tokenizer_path = iree_string_view_empty();
    options.checkpoint_capacity = 4;
    return options;
  }

  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = SourceTextServiceTest::ServiceOptions();
    options.session_cache_capacity = 1;
    return options;
  }

  std::string Encode(const std::string& tokens, size_t position = 0,
                     const std::string& handle = "",
                     const std::string& headers = "") {
    return Read(Send(
        "{\"model\":\"fixture-text\",\"position\":" + std::to_string(position) +
            ",\"token_ids\":" + tokens +
            (handle.empty() ? "" : ",\"session\":\"" + handle + "\"") + "}",
        headers, "/v1/sequences"));
  }

  static std::string Handle(const std::string& response) {
    const auto begin = response.find("\"session\":\"");
    return begin == std::string::npos ? "" : response.substr(begin + 11, 32);
  }

  static uint32_t Hidden(const std::vector<int32_t>& tokens) {
    uint32_t value = 0;
    for (int32_t token : tokens) {
      value = (5 * value + token + 2) % 97;
    }
    return value;
  }

  static void CheckChunk(const std::string& response,
                         const std::vector<int32_t>& history, size_t begin) {
    uint8_t bytes[4];
    const uint32_t hidden = Hidden(history);
    iree_unaligned_store_le_u16(bytes, hidden >> 16);
    iree_unaligned_store_le_u16(bytes + 2, hidden & 0xFFFF);
    char encoded[8];
    iree_host_size_t length = 0;
    IREE_ASSERT_OK(iree_base64_encode(
        iree_make_const_byte_span(bytes, sizeof(bytes)),
        iree_make_mutable_string_view(encoded, sizeof(encoded)), &length));
    std::string expected =
        "\"position\":" + std::to_string(begin) +
        ",\"next_position\":" + std::to_string(history.size()) +
        ",\"token_ids\":[";
    for (size_t i = begin; i < history.size(); ++i) {
      expected += (i == begin ? "" : ",") + std::to_string(history[i]);
    }
    expected += "],\"encoding\":\"base64\",\"data\":\"" +
                std::string(encoded, length) + "\"}";
    EXPECT_NE(response.find(expected), std::string::npos) << response;
  }
};

TEST_F(SourceSequenceServiceTest, HttpAppendsCacheRestoreAndCheckpointFork) {
  ASSERT_NO_FATAL_FAILURE(Start());
  const auto first = Encode("[0,1,2,0,1,1,0,0,2,1,0]");
  ASSERT_NE(first.find("200 OK"), std::string::npos) << first;
  ASSERT_NE(first.find("event: done"), std::string::npos) << first;
  const auto handle = Handle(first);
  ASSERT_EQ(handle.size(), 32u) << first;
  EXPECT_NE(first.find("\"position\":11}"), std::string::npos) << first;
  std::vector<int32_t> history = {0, 1, 2, 0, 1, 1, 0, 0, 2, 1, 0};
  for (size_t begin = 0; begin < history.size(); begin += 3) {
    CheckChunk(first,
               std::vector<int32_t>(
                   history.begin(),
                   history.begin() + std::min(begin + 3, history.size())),
               begin);
  }
  const auto second = Encode("[1]", 11, handle);
  ASSERT_NE(second.find("200 OK"), std::string::npos) << second;
  EXPECT_NE(second.find("\"position\":12}"), std::string::npos) << second;
  history.push_back(1);
  CheckChunk(second, history, 11);
  EXPECT_NE(Encode("[1]", 11, handle).find("409 Conflict"), std::string::npos);
  EXPECT_NE(Encode("[1]", 0, std::string(32, '0')).find("404 Not Found"),
            std::string::npos);
  const auto pinned = Read(Send(
      "", "X-Loom-Session: " + handle + "\r\nX-Loom-Model: fixture-text\r\n",
      "/v1/checkpoints/branch"));
  ASSERT_NE(pinned.find("201 Created"), std::string::npos) << pinned;
  for (int i = 0; i < 4; ++i) {
    const auto peer = Encode("[0]");
    ASSERT_NE(peer.find("200 OK"), std::string::npos) << peer;
  }
  const auto resumed = Encode("[2]", 12, handle);
  ASSERT_NE(resumed.find("200 OK"), std::string::npos) << resumed;
  EXPECT_EQ(Handle(resumed), handle);
  EXPECT_NE(resumed.find("\"position\":13}"), std::string::npos) << resumed;
  const auto fork = Encode("[2]", 12, "", "X-Loom-Checkpoint: branch\r\n");
  ASSERT_NE(fork.find("200 OK"), std::string::npos) << fork;
  EXPECT_NE(Handle(fork), handle);
  history.push_back(2);
  CheckChunk(resumed, history, 12);
  CheckChunk(fork, history, 12);
  for (int i = 0; i < 6; ++i) {
    ASSERT_NE(Encode("[0]").find("200 OK"), std::string::npos);
  }
  EXPECT_NE(Encode("[1]", 13, handle).find("404 Not Found"), std::string::npos);
}

TEST_F(SourceSequenceServiceTest, HttpRejectsMalformedInputBeforeMutation) {
  ASSERT_NO_FATAL_FAILURE(Start());
  const auto first = Encode("[0,1]");
  const auto handle = Handle(first);
  ASSERT_EQ(handle.size(), 32u) << first;
  for (const char* tokens :
       {"[]", "[-1]", "[2147483648]", "[1.5]", "[true]", "[\"1\"]"}) {
    SCOPED_TRACE(tokens);
    EXPECT_NE(Encode(tokens, 2, handle).find("400 Bad Request"),
              std::string::npos);
  }
  for (const char* fields :
       {R"("token_ids":[0])", R"("position":0)",
        R"("position":0,"position":0,"token_ids":[0])",
        R"("position":0,"token_ids":[0],"max_tokens":1)",
        R"("position":0,"token_ids":[0],"session":"row-0")",
        R"("position":0,"token_ids":[0],"options":null)"}) {
    SCOPED_TRACE(fields);
    const auto response =
        Read(Send(std::string(R"({"model":"fixture-text",)") + fields + "}", "",
                  "/v1/sequences"));
    EXPECT_NE(response.find("400 Bad Request"), std::string::npos) << response;
  }
  EXPECT_NE(Encode("[0]", 1).find("409 Conflict"), std::string::npos);
  const auto trailing =
      Read(Send(R"({"model":"fixture-text","position":0,"token_ids":[0]} {})",
                "", "/v1/sequences"));
  EXPECT_NE(trailing.find("400 Bad Request"), std::string::npos) << trailing;
  EXPECT_NE(Encode("[0]", 0, "", "X-Loom-Session:\r\n").find("400 Bad Request"),
            std::string::npos);
  const auto resumed = Encode("[2]", 2, handle);
  ASSERT_NE(resumed.find("200 OK"), std::string::npos) << resumed;
  CheckChunk(resumed, {0, 1, 2}, 2);
}

TEST_F(SourceSequenceServiceTest, NativeChatRejectionPreservesOwnedInput) {
  const auto options = ServiceOptions();
  loom_serve_text_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &value));
  auto destroy = [](loom_serve_text_service_t* service) {
    loom_serve_text_service_free(service, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
      value, destroy);
  const auto interface = loom_serve_text_service_interface(value);
  EXPECT_EQ(interface.try_open_text, nullptr);
  EXPECT_NE(interface.try_open_text_checkpoint, nullptr);

  loom_serve_text_request_t request = {.session = IREE_SV("caller")};
  iree_string_builder_initialize(host_allocator_, &request.body);
  IREE_ASSERT_OK(iree_string_builder_append_cstring(
      &request.body, R"({"messages":[{"role":"user","content":"a"}]})"));
  const auto* input = iree_string_builder_buffer(&request.body);
  loom_serve_text_stream_t* stream = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_text_service_try_open(value, &request, &stream));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(iree_string_builder_buffer(&request.body), input);
  EXPECT_TRUE(iree_string_view_equal(request.session, IREE_SV("caller")));
  EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 0u);
  if (stream) {
    loom_serve_text_stream_close(stream);
  }
  iree_string_builder_deinitialize(&request.body);
}

TEST_F(SourceSequenceServiceTest,
       DisconnectRetiresCaptureAndInvalidatesHandle) {
  const auto options = ServiceOptions();
  loom_serve_text_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &value));
  auto destroy = [](loom_serve_text_service_t* service) {
    loom_serve_text_service_free(service, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
      value, destroy);
  const auto interface = loom_serve_text_service_interface(value);
  Observe(interface);
  const auto idle = [&] {
    return !interface.is_pending(interface.self) &&
           !interface.is_active(interface.self) &&
           !interface.snapshot(interface.self).queued_requests;
  };
  loom_serve_http_connection_t* connection = nullptr;
  const int first =
      Send(R"({"position":0,"token_ids":[0,1]})", "", "/v1/sequences");
  ASSERT_NO_FATAL_FAILURE(Accept(first, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
  const auto handle = Handle(Read(first));
  ASSERT_EQ(handle.size(), 32u);
  const std::string append =
      "{\"session\":\"" + handle + R"(","position":2,"token_ids":[2,1,0,2]})";
  // A queued request has not acquired or mutated retained state. Cancelling
  // it returns host credit without invalidating the completed handle.
  const linger reset = {1, 0};
  const int queued = Send(append, "", "/v1/sequences");
  ASSERT_NO_FATAL_FAILURE(Accept(queued, interface, &connection));
  ASSERT_EQ(setsockopt(queued, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(queued), 0);
  ASSERT_TRUE(iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout()));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
  EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
  const int pin =
      Send("", "X-Loom-Session: " + handle + "\r\n", "/v1/checkpoints/before");
  ASSERT_NO_FATAL_FAILURE(Accept(pin, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
  ASSERT_NE(Read(pin).find("201 Created"), std::string::npos);
  const int client = Send(append, "", "/v1/sequences");
  ASSERT_NO_FATAL_FAILURE(Accept(client, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return std::string(interface.snapshot(interface.self).phase) == "execute";
  }));
  ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(client), 0);
  ASSERT_TRUE(iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout()));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
  EXPECT_EQ(interface.telemetry.counters->cancelled, 2u);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, 0u);
  const int stale = Send(append, "", "/v1/sequences");
  ASSERT_NO_FATAL_FAILURE(Accept(stale, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
  EXPECT_NE(Read(stale).find("404 Not Found"), std::string::npos);
  const int fork = Send(R"({"position":2,"token_ids":[2]})",
                        "X-Loom-Checkpoint: before\r\n", "/v1/sequences");
  ASSERT_NO_FATAL_FAILURE(Accept(fork, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
  const auto restored = Read(fork);
  ASSERT_NE(restored.find("200 OK"), std::string::npos) << restored;
  CheckChunk(restored, {0, 1, 2}, 2);
  interface.set_observer(interface.self, {});
}

class RegisteredSequenceServiceTest : public SourceSequenceServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
};

TEST_F(RegisteredSequenceServiceTest,
       RegistrationReplacementInvalidatesHandles) {
  ASSERT_NO_FATAL_FAILURE(StartManaged());
  const std::string declaration =
      "{\"name\":\"fixture-text\",\"base_directory\":\"" +
      std::filesystem::absolute(directory_).string() +
      R"(","definition":{"kind":"sequence","prepare":"prepare.loom","control":"control.loom","sources":["advance.loom"],"weights":")" +
      weights_ +
      R"(","rows":4,"context_capacity":96,"pool_capacity":256,"prefill_capacity":32,"checkpoint_capacity":4,"session_cache_capacity":1}})";
  auto response = Read(Send(declaration, "", "/v1/models/register"));
  ASSERT_EQ(response.find("HTTP/1.1 202"), 0u) << response;
  const auto first = Encode("[0,1,2]");
  ASSERT_NE(first.find("200 OK"), std::string::npos) << first;
  const auto handle = Handle(first);
  CheckChunk(first, {0, 1, 2}, 0);
  response = Read(Send("", "", "/v1/models", "GET"));
  EXPECT_NE(response.find("/v1/sequences"), std::string::npos) << response;
  response = Read(Send(
      "", "X-Loom-Session: " + handle + "\r\nX-Loom-Model: fixture-text\r\n",
      "/v1/checkpoints/pinned"));
  ASSERT_NE(response.find("201 Created"), std::string::npos) << response;
  response = Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/drain"));
  ASSERT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
  response = Read(Send("", "X-Loom-Model: fixture-text\r\n",
                       "/v1/checkpoints/pinned", "DELETE"));
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
  EXPECT_NE(response.find("\"pinned\":false"), std::string::npos) << response;
  response =
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"));
  ASSERT_EQ(response.find("HTTP/1.1 202"), 0u) << response;
  WaitModel("fixture-text", {});
  response = Read(Send(declaration, "", "/v1/models/register"));
  ASSERT_EQ(response.find("HTTP/1.1 202"), 0u) << response;
  EXPECT_NE(Encode("[1]", 3, handle).find("404 Not Found"), std::string::npos);
  const auto replacement = Encode("[0,1,2]");
  ASSERT_NE(replacement.find("200 OK"), std::string::npos) << replacement;
  EXPECT_NE(Handle(replacement), handle);
  CheckChunk(replacement, {0, 1, 2}, 0);
}

}  // namespace
