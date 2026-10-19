// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>
#include <thread>

#include "iree/testing/gtest.h"
#include "loomc/link.h"
#include "loomc/link_index.h"
#include "loomc/module_config.h"
#include "module.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_free>;
using IndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;

constexpr char kSource[] = R"(config.decl @answer : index
func.def public @entry() -> (index) {
  %answer = config.get @answer : index
  func.return %answer : index
}
)";

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

SourcePtr Serialize(loomc_module_t* module, loomc_source_format_t format) {
  const loomc_module_serialize_options_t options = {.format = format};
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

std::string Text(loomc_module_t* module) {
  SourcePtr source = Serialize(module, LOOMC_SOURCE_FORMAT_TEXT);
  loomc_byte_span_t contents = loomc_source_contents(source.get());
  return std::string(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
}

class LinkIndexModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(
        loomc_context_create(nullptr, loomc_allocator_system(), &context));
    context_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    workspace_.reset(workspace);
    loomc_link_index_builder_t* builder = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_allocate(
        context, nullptr, loomc_allocator_system(), &builder));
    builder_.reset(builder);
  }

  ModulePtr Parse() {
    const loomc_source_options_t options = {
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view("snapshot.loom"),
        .contents = loomc_make_byte_span(kSource, strlen(kSource)),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    LOOMC_EXPECT_OK(
        loomc_source_create(&options, loomc_allocator_system(), &source));
    SourcePtr source_owner(source);
    return Deserialize(source);
  }

  ModulePtr Deserialize(loomc_source_t* source) {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &module, &result));
    ResultPtr result_owner(result);
    EXPECT_TRUE(loomc_result_succeeded(result));
    return ModulePtr(module);
  }

  IndexPtr Finish() {
    loomc_link_index_t* index = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(
        loomc_link_index_builder_finish(builder_.get(), &index, &result));
    ResultPtr result_owner(result);
    EXPECT_TRUE(loomc_result_succeeded(result));
    return IndexPtr(index);
  }

  // Shared immutable dialect and target metadata.
  ContextPtr context_;
  // Caller storage deliberately released before the snapshot is consumed.
  WorkspacePtr workspace_;
  // Builder owning the independently captured inputs.
  BuilderPtr builder_;
};

TEST_F(LinkIndexModuleTest, SnapshotPrecedesMutationAndOutlivesCallerStorage) {
  ModulePtr original = Parse();
  const std::string original_text = Text(original.get());
  LOOMC_ASSERT_OK(loomc_link_index_builder_add_module(
      builder_.get(), original.get(), nullptr, nullptr));

  const loomc_config_binding_t binding = {
      .key = loomc_make_cstring_view("answer"),
      .value = loomc_make_cstring_view("7"),
  };
  const loomc_module_config_options_t options = {
      .config = {.bindings = &binding, .binding_count = 1},
  };
  loomc_result_t* applied_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_apply_config(original.get(), workspace_.get(),
                                            &options, loomc_allocator_system(),
                                            &applied_result));
  ResultPtr applied_owner(applied_result);
  ASSERT_TRUE(loomc_result_succeeded(applied_result));
  ASSERT_NE(Text(original.get()), original_text);
  original.reset();
  loomc_workspace_trim(workspace_.get());
  workspace_.reset();

  IndexPtr index = Finish();
  ASSERT_NE(index, nullptr);
  builder_.reset();
  loomc_link_index_provider_t provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(index.get(), 0, &provider));
  EXPECT_EQ(provider.kind, LOOMC_LINK_PROVIDER_KIND_MATERIALIZED);

  loomc_workspace_t* link_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &link_workspace));
  workspace_.reset(link_workspace);
  loomc_linker_t* linker = nullptr;
  LOOMC_ASSERT_OK(loomc_linker_create(context_.get(), nullptr,
                                      loomc_allocator_system(), &linker));
  LinkerPtr linker_owner(linker);
  const loomc_link_options_t link_options = {
      .link_index = index.get(),
      .mode = LOOMC_LINK_MODE_MERGE,
  };
  ModulePtr linked;
  for (int invocation = 0; invocation < 2; ++invocation) {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_link_module(linker, workspace_.get(), &link_options,
                                      &module, &result));
    ResultPtr result_owner(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    linked.reset(module);
    EXPECT_EQ(Text(linked.get()), original_text);
  }
  index.reset();
  linker_owner.reset();

  // The linked product keeps the original authored source, even after the
  // caller's input, its workspace, the builder, and the index have gone away.
  const loom_source_table_resolver_t* sources =
      loomc_module_source_table(linked.get());
  ASSERT_EQ(sources->count, 1u);
  EXPECT_EQ(std::string(sources->entries[0].filename.data,
                        sources->entries[0].filename.size),
            "snapshot.loom");
  EXPECT_EQ(std::string(sources->entries[0].source.data,
                        sources->entries[0].source.size),
            kSource);

  SourcePtr bytecode = Serialize(linked.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  linked.reset();
  ModulePtr replay = Deserialize(bytecode.get());
  EXPECT_EQ(Text(replay.get()), original_text);
}

TEST_F(LinkIndexModuleTest, RejectsInvalidAdmissionWithoutConsumingSlots) {
  ModulePtr module = Parse();
  loomc_link_index_source_slot_t slot = {};
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_index_builder_add_module(
                             builder_.get(), nullptr, nullptr, &slot));
  EXPECT_EQ(slot.ordinal, LOOMC_HOST_SIZE_MAX);
  LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_source_slot(builder_.get(),
                                                               nullptr, &slot));
  EXPECT_EQ(slot.ordinal, 0u);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_OUT_OF_RANGE,
                         loomc_link_index_builder_fill_module_slot(
                             builder_.get(), {1}, module.get()));

  loomc_context_t* other_context = nullptr;
  LOOMC_ASSERT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &other_context));
  ContextPtr other_owner(other_context);
  loomc_link_index_builder_t* other_builder = nullptr;
  LOOMC_ASSERT_OK(loomc_link_index_builder_allocate(
      other_context, nullptr, loomc_allocator_system(), &other_builder));
  BuilderPtr other_builder_owner(other_builder);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_index_builder_add_module(
                             other_builder, module.get(), nullptr, nullptr));

  LOOMC_ASSERT_OK(loomc_link_index_builder_fill_module_slot(
      builder_.get(), slot, module.get()));
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_ALREADY_EXISTS,
                         loomc_link_index_builder_fill_module_slot(
                             builder_.get(), slot, module.get()));
  SourcePtr source = Serialize(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_ALREADY_EXISTS,
                         loomc_link_index_builder_fill_source_slot(
                             builder_.get(), slot, source.get()));
  IndexPtr index = Finish();
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                         loomc_link_index_builder_fill_module_slot(
                             builder_.get(), slot, module.get()));
}

TEST_F(LinkIndexModuleTest, ConcurrentFillsPreserveReservationOrder) {
  ModulePtr module = Parse();
  std::array<loomc_link_index_source_slot_t, 4> slots = {};
  for (size_t i = 0; i < slots.size(); ++i) {
    const std::string name = "provider-" + std::to_string(i);
    const loomc_link_index_source_options_t options = {
        .provider_name = loomc_make_cstring_view(name.c_str()),
        .role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
    };
    LOOMC_ASSERT_OK(loomc_link_index_builder_reserve_source_slot(
        builder_.get(), &options, &slots[i]));
  }
  std::array<loomc_status_t, 4> statuses = {};
  std::array<std::thread, 4> threads;
  for (size_t i = 0; i < slots.size(); ++i) {
    threads[i] = std::thread([&, i] {
      statuses[i] = loomc_link_index_builder_fill_module_slot(
          builder_.get(), slots[i], module.get());
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (loomc_status_t status : statuses) {
    LOOMC_ASSERT_OK(status);
  }
  module.reset();
  workspace_.reset();
  IndexPtr index = Finish();
  EXPECT_EQ(loomc_link_index_provider_count(index.get()), slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    loomc_link_index_provider_t provider = {};
    ASSERT_TRUE(loomc_link_index_provider_at(index.get(), i, &provider));
    EXPECT_EQ(ToString(provider.name), "provider-" + std::to_string(i));
  }
}

}  // namespace
