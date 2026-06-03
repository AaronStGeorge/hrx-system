// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/module.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;

using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;

using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

void ExpectFailedResultCode(const loomc_result_t* result, const char* code) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  const loomc_host_size_t diagnostic_count =
      loomc_result_diagnostic_count(result);
  ASSERT_NE(diagnostic_count, 0u);
  bool found = false;
  for (loomc_host_size_t i = 0; i < diagnostic_count; ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    ASSERT_NE(diagnostic, nullptr);
    found |= ToString(diagnostic->code) == code;
  }
  EXPECT_TRUE(found);
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, source, nullptr, loomc_allocator_system(), &module, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

ModulePtr CreateFunctionModule(loomc_context_t* context) {
  SourcePtr source = CreateTextSource("functions.loom", R"(
func.def public @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

kernel.def export("dispatch") @entry() {
  %workgroups_z = index.constant 4 : index
  %workgroups_y = index.constant 3 : index
  %workgroups_x = index.constant 2 : index
  %workgroup_size_z = index.constant 1 : index
  %workgroup_size_y = index.constant 6 : index
  %workgroup_size_x = index.constant 5 : index
  kernel.launch.config workgroups(%workgroups_x, %workgroups_y, %workgroups_z) workgroup_size(%workgroup_size_x, %workgroup_size_y, %workgroup_size_z) : index
} launch {
  kernel.return
}
)");
  return DeserializeModule(context, source.get());
}

const loomc_module_function_t* FindFunction(
    const std::vector<loomc_module_function_t>& functions, const char* name) {
  for (const loomc_module_function_t& function : functions) {
    if (ToString(function.symbol_name) == name) {
      return &function;
    }
  }
  return nullptr;
}

TEST(ModuleTest, QueriesFunctionsAndKernelSidecars) {
  ContextPtr context = CreateContext();
  ModulePtr module = CreateFunctionModule(context.get());

  loomc_module_function_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_UNKNOWN,
  };
  loomc_host_size_t function_count = 0;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), 0, nullptr,
      &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr count_result(result);
  ExpectSucceededResult(count_result.get());
  ASSERT_EQ(function_count, 2u);

  std::vector<loomc_module_function_t> functions(function_count);
  result = nullptr;
  status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), functions.size(),
      functions.data(), &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr query_result(result);
  ExpectSucceededResult(query_result.get());
  ASSERT_EQ(function_count, functions.size());

  const loomc_module_function_t* helper = FindFunction(functions, "helper");
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(helper->function_ordinal, 0u);
  EXPECT_EQ(helper->kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION);
  EXPECT_TRUE(helper->flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC);
  loomc_module_kernel_function_info_t kernel_info = {};
  EXPECT_FALSE(loomc_module_function_try_get_kernel_info(module.get(), helper,
                                                         &kernel_info));
  EXPECT_FALSE(loomc_module_function_try_get_kernel_info_at(
      module.get(), helper->function_ordinal, &kernel_info));
  status =
      loomc_module_function_get_kernel_info(module.get(), helper, &kernel_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
  status = loomc_module_function_get_kernel_info_at(
      module.get(), helper->function_ordinal, &kernel_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
  loomc_module_function_export_info_t export_info = {};
  EXPECT_FALSE(loomc_module_function_try_get_export_info_at(
      module.get(), helper->function_ordinal, &export_info));
  status = loomc_module_function_get_export_info_at(
      module.get(), helper->function_ordinal, &export_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);

  const loomc_module_function_t* entry = FindFunction(functions, "entry");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->function_ordinal, 1u);
  EXPECT_EQ(entry->kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
  EXPECT_TRUE(entry->flags & LOOMC_MODULE_FUNCTION_FLAG_HAS_EXPORT_INFO);
  ASSERT_TRUE(loomc_module_function_try_get_kernel_info(module.get(), entry,
                                                        &kernel_info));
  ASSERT_TRUE(loomc_module_function_try_get_kernel_info_at(
      module.get(), entry->function_ordinal, &kernel_info));
  EXPECT_TRUE(
      kernel_info.flags &
      LOOMC_MODULE_KERNEL_FUNCTION_FLAG_HAS_STATIC_DISPATCH_WORKGROUP_COUNT);
  EXPECT_EQ(kernel_info.static_dispatch_workgroup_count.x, 2u);
  EXPECT_EQ(kernel_info.static_dispatch_workgroup_count.y, 3u);
  EXPECT_EQ(kernel_info.static_dispatch_workgroup_count.z, 4u);
  EXPECT_TRUE(kernel_info.flags &
              LOOMC_MODULE_KERNEL_FUNCTION_FLAG_HAS_STATIC_WORKGROUP_SIZE);
  EXPECT_EQ(kernel_info.static_workgroup_size.x, 5u);
  EXPECT_EQ(kernel_info.static_workgroup_size.y, 6u);
  EXPECT_EQ(kernel_info.static_workgroup_size.z, 1u);

  export_info = {};
  ASSERT_TRUE(loomc_module_function_try_get_export_info(module.get(), entry,
                                                        &export_info));
  ASSERT_TRUE(loomc_module_function_try_get_export_info_at(
      module.get(), entry->function_ordinal, &export_info));
  EXPECT_TRUE(export_info.flags & LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL);
  EXPECT_EQ(ToString(export_info.export_symbol), "dispatch");
  EXPECT_FALSE(export_info.flags &
               LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_ORDINAL);
}

TEST(ModuleTest, LooksUpFunctionNamesWithOrWithoutSigil) {
  ContextPtr context = CreateContext();
  ModulePtr module = CreateFunctionModule(context.get());

  loomc_module_function_t by_plain_name = {};
  loomc_status_t status = loomc_module_lookup_function(
      module.get(), loomc_make_cstring_view("entry"), &by_plain_name);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(ToString(by_plain_name.symbol_name), "entry");

  loomc_module_function_t by_symbol_name = {};
  ASSERT_TRUE(loomc_module_try_lookup_function(
      module.get(), loomc_make_cstring_view("@entry"), &by_symbol_name));
  EXPECT_EQ(by_symbol_name.function_ordinal, by_plain_name.function_ordinal);
  EXPECT_EQ(by_symbol_name.kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
}

TEST(ModuleTest, GetsFunctionsByPublicOrdinal) {
  ContextPtr context = CreateContext();
  ModulePtr module = CreateFunctionModule(context.get());

  loomc_module_function_t function = {};
  ASSERT_TRUE(loomc_module_try_get_function_at(module.get(), 0, &function));
  EXPECT_EQ(ToString(function.symbol_name), "helper");
  EXPECT_EQ(function.function_ordinal, 0u);

  loomc_status_t status =
      loomc_module_get_function_at(module.get(), 1, &function);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(ToString(function.symbol_name), "entry");
  EXPECT_EQ(function.function_ordinal, 1u);

  EXPECT_FALSE(loomc_module_try_get_function_at(module.get(), 2, &function));
  status = loomc_module_get_function_at(module.get(), 2, &function);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
}

TEST(ModuleTest, QueryReportsTotalFunctionCountForPartialStorage) {
  ContextPtr context = CreateContext();
  ModulePtr module = CreateFunctionModule(context.get());

  loomc_module_function_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_UNKNOWN,
  };
  loomc_module_function_t function = {};
  loomc_host_size_t function_count = 0;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), 1, &function,
      &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  EXPECT_EQ(function_count, 2u);
  EXPECT_FALSE(ToString(function.symbol_name).empty());
}

TEST(ModuleTest, RejectsMalformedFunctionViewsWithoutCrashing) {
  ContextPtr context = CreateContext();
  ModulePtr module = CreateFunctionModule(context.get());

  loomc_module_function_t lookup_function = {};
  loomc_string_view_t invalid_symbol_name = loomc_make_string_view(nullptr, 1);
  EXPECT_FALSE(loomc_module_try_lookup_function(
      module.get(), invalid_symbol_name, &lookup_function));
  loomc_status_t status = loomc_module_lookup_function(
      module.get(), invalid_symbol_name, &lookup_function);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);

  loomc_module_function_t function = {};
  status = loomc_module_lookup_function(
      module.get(), loomc_make_cstring_view("entry"), &function);
  LOOMC_ASSERT_OK(status);
  function.symbol_name =
      loomc_make_string_view(nullptr, function.symbol_name.size);

  loomc_module_kernel_function_info_t kernel_info = {};
  EXPECT_FALSE(loomc_module_function_try_get_kernel_info(
      module.get(), &function, &kernel_info));
  status = loomc_module_function_get_kernel_info(module.get(), &function,
                                                 &kernel_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);

  loomc_module_function_export_info_t export_info = {};
  EXPECT_FALSE(loomc_module_function_try_get_export_info(
      module.get(), &function, &export_info));
  status = loomc_module_function_get_export_info(module.get(), &function,
                                                 &export_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
}

TEST(ModuleTest, ReportsNamedFunctionQueryMissInResult) {
  ContextPtr context = CreateContext();
  ModulePtr module = CreateFunctionModule(context.get());

  loomc_module_function_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_make_cstring_view("@missing"),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_UNKNOWN,
  };
  loomc_module_function_t function = {};
  loomc_host_size_t function_count = 1;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_query_functions(
      module.get(), &options, loomc_allocator_system(), 1, &function,
      &function_count, &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_EQ(function_count, 0u);
  ExpectFailedResultCode(result_ptr.get(), "MODULE_FUNCTION/NOT_FOUND");
}

}  // namespace
