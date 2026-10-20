// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/request.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ImageRequestTest, OwnsDecodedPromptAndFullWidthSeed) {
  std::string body =
      R"({"prompt":"café \u732b\n\ud83e\udd16\u0000x","seed":"18446744073709551615","loras":[{"id":"café \u732b","scale":0.25},{"id":"other","scale":-0.5}],"model":"test-image","size":"384x256","n":1,"response_format":"b64_json"})";
  loom_serve_image_request_t request;
  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      iree_make_string_view(body.data(), body.size()), IREE_SV("test-image"),
      384, 256, iree_allocator_system(), &request));
  body.assign(body.size(), 'x');
  const auto prompt = iree_string_builder_view(&request.prompt);
  const char expected[] = "café 猫\n🤖\0x";
  EXPECT_EQ(std::string(prompt.data, prompt.size),
            std::string(expected, sizeof(expected) - 1));
  EXPECT_EQ(request.seed, UINT64_MAX);
  const auto selections = iree_string_builder_view(&request.loras);
  ASSERT_EQ(selections.size, 56u);
  EXPECT_EQ(iree_unaligned_load_le_u64(selections.data), 9u);
  EXPECT_EQ(iree_unaligned_load_le_f32(selections.data + 8), 0.25f);
  EXPECT_EQ(iree_unaligned_load_le_u32(selections.data + 12), 0u);
  EXPECT_EQ(std::string(selections.data + 16, 9), "café 猫");
  EXPECT_EQ(std::string(selections.data + 25, 7), std::string(7, '\0'));
  EXPECT_EQ(iree_unaligned_load_le_u64(selections.data + 32), 5u);
  EXPECT_EQ(iree_unaligned_load_le_f32(selections.data + 40), -0.5f);
  EXPECT_EQ(std::string(selections.data + 48, 5), "other");
  EXPECT_EQ(std::string(selections.data + 53, 3), std::string(3, '\0'));
  EXPECT_EQ(request.width, 384u);
  EXPECT_EQ(request.height, 256u);
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, EmptyPromptDefaultsAndJsonc) {
  loom_serve_image_request_t request;
  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      IREE_SV(" \n{/* image */\"prompt\":\"\",} // done\n"),
      IREE_SV("test-image"), 384, 256, iree_allocator_system(), &request));
  EXPECT_EQ(iree_string_builder_size(&request.prompt), 0u);
  EXPECT_EQ(request.seed, 0u);
  EXPECT_EQ(iree_string_builder_size(&request.loras), 0u);
  EXPECT_EQ(request.width, 384u);
  EXPECT_EQ(request.height, 256u);
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, RejectsMalformedOrUnsupportedRequests) {
  for (const char* body : {
           "{}",
           "[]",
           R"({"prompt":null})",
           R"({"prompt":12})",
           R"({"prompt":"x","prompt":"y"})",
           R"({"prompt":"x","unknown":0})",
           R"({"prompt":"x","seed":-1})",
           R"({"prompt":"x","seed":1.2})",
           R"({"prompt":"x","seed":"42junk"})",
           R"({"prompt":"x","strength":1e100})",
           R"({"prompt":"x","strength":"0"})",
           R"({"prompt":"x","loras":null})",
           R"({"prompt":"x","loras":{}})",
           R"({"prompt":"x","loras":[],"loras":[]})",
           R"({"prompt":"x","loras":["red"]})",
           R"({"prompt":"x","loras":[{}]})",
           R"({"prompt":"x","loras":[{"id":""}]})",
           R"({"prompt":"x","loras":[{"id":1}]})",
           R"({"prompt":"x","loras":[{"id":"x","id":"y"}]})",
           R"({"prompt":"x","loras":[{"id":"x","scale":1,"scale":2}]})",
           R"({"prompt":"x","loras":[{"path":"x"}]})",
           R"({"prompt":"x","loras":[{"id":"x","unknown":1}]})",
           R"({"prompt":"x","loras":[{"id":"x","scale":"1"}]})",
           R"({"prompt":"x","loras":[{"id":"x","scale":null}]})",
           R"({"prompt":"x","loras":[{"id":"x","scale":1e100}]})",
           R"({"prompt":"x","loras":[{"id":"ok"},{"id":"\ud800"}]})",
           R"({"prompt":"x","size":"auto"})",
           R"({"prompt":"x","size":"0x384"})",
           R"({"prompt":"x","size":"256x0"})",
           R"({"prompt":"x","size":"256"})",
           R"({"prompt":"x","size":"256x"})",
           R"({"prompt":"x","size":"x384"})",
           R"({"prompt":"x","size":"256X384"})",
           R"({"prompt":"x","size":"256x384x1"})",
           R"({"prompt":"x","size":"-256x384"})",
           R"({"prompt":"x","size":"256x 384"})",
           R"({"prompt":"x","size":"4294967296x384"})",
           R"({"prompt":"x","size":"4294967295x1"})",
           R"({"prompt":"x","size":"4294967295x4294967295"})",
           R"({"prompt":"x","model":"other"})",
           R"({"prompt":"x","n":2})",
           R"({"prompt":"x","response_format":"url"})",
           R"({"prompt":"x"} {})",
           R"({"prompt":"\ud800"})",
           "{\"prompt\":\"\xff\"}",
       }) {
    SCOPED_TRACE(body);
    loom_serve_image_request_t request;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_image_request_initialize(iree_make_cstring_view(body),
                                            IREE_SV("test-image"), 384, 256,
                                            iree_allocator_system(), &request));
    EXPECT_EQ(request.prompt.buffer, nullptr);
    EXPECT_EQ(request.loras.buffer, nullptr);
    loom_serve_image_request_deinitialize(&request);
  }
}

TEST(ImageRequestTest, EmptyAndDefaultScaleSelections) {
  loom_serve_image_request_t request;
  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      IREE_SV(R"({"prompt":"x","loras":[]})"), IREE_SV("test-image"), 384, 256,
      iree_allocator_system(), &request));
  EXPECT_EQ(iree_string_builder_size(&request.loras), 0u);
  loom_serve_image_request_deinitialize(&request);

  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      IREE_SV(R"({"prompt":"x","loras":[{"id":"red"}]})"),
      IREE_SV("test-image"), 384, 256, iree_allocator_system(), &request));
  const auto selections = iree_string_builder_view(&request.loras);
  ASSERT_EQ(selections.size, 24u);
  EXPECT_EQ(iree_unaligned_load_le_u64(selections.data), 3u);
  EXPECT_EQ(iree_unaligned_load_le_f32(selections.data + 8), 1.0f);
  EXPECT_EQ(std::string(selections.data + 16, 3), "red");
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, SelectionStorageGrowsWithoutAnArbitraryCountLimit) {
  std::string body = R"({"prompt":"x","loras":[)";
  for (size_t i = 0; i < 1024; ++i) {
    if (i) {
      body += ',';
    }
    body += "{\"id\":\"" + std::to_string(i) + "\",\"scale\":0}";
  }
  body += "]}";
  loom_serve_image_request_t request;
  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      iree_make_string_view(body.data(), body.size()), IREE_SV("test-image"),
      384, 256, iree_allocator_system(), &request));
  const auto selections = iree_string_builder_view(&request.loras);
  EXPECT_EQ(selections.size, 1024u * 24);
  for (size_t i = 0; i < 1024; ++i) {
    const char* record = selections.data + i * 24;
    const auto length = iree_unaligned_load_le_u64(record);
    EXPECT_EQ(std::string(record + 16, length), std::to_string(i));
    EXPECT_EQ(iree_unaligned_load_le_f32(record + 8), 0.0f);
  }
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, ExplicitGeometryDoesNotChangeDefaults) {
  for (const char* size : {"256x384", "1536x1024", "1024x1536", "1x1"}) {
    const std::string body =
        std::string(R"({"prompt":"x","size":")") + size + "\"}";
    loom_serve_image_request_t request;
    IREE_ASSERT_OK(loom_serve_image_request_initialize(
        iree_make_string_view(body.data(), body.size()), IREE_SV("test-image"),
        384, 256, iree_allocator_system(), &request));
    EXPECT_EQ(
        std::to_string(request.width) + "x" + std::to_string(request.height),
        size);
    loom_serve_image_request_deinitialize(&request);
  }
  loom_serve_image_request_t request;
  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      IREE_SV(R"({"prompt":"x"})"), IREE_SV("test-image"), 384, 256,
      iree_allocator_system(), &request));
  EXPECT_EQ(request.width, 384u);
  EXPECT_EQ(request.height, 256u);
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, AllocationFailurePublishesNoOwnership) {
  for (const char* body :
       {R"({"prompt":"x"})", R"({"prompt":"x","loras":[{"id":"red"}]})"}) {
    loom_serve_image_request_t request;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_image_request_initialize(iree_make_cstring_view(body),
                                            IREE_SV("test-image"), 384, 256,
                                            iree_allocator_null(), &request));
    EXPECT_EQ(request.prompt.buffer, nullptr);
    EXPECT_EQ(request.loras.buffer, nullptr);
    loom_serve_image_request_deinitialize(&request);
  }
}

TEST(ImageRequestTest, SeedOverflowPublishesNoOwnership) {
  loom_serve_image_request_t request;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_image_request_initialize(
          IREE_SV("{\"prompt\":\"x\",\"seed\":18446744073709551616}"),
          IREE_SV("test-image"), 384, 256, iree_allocator_system(), &request));
  EXPECT_EQ(request.prompt.buffer, nullptr);
  loom_serve_image_request_deinitialize(&request);
}

}  // namespace
