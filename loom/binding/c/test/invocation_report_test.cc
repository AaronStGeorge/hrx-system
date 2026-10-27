// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "loom/error/error_catalog.h"
#include "loom/pass/registry.h"
#include "loomc/compile_report.h"
#include "loomc/interop.h"
#include "loomc/pass.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

struct AllocationState {
  // Allocation attempt to reject; the maximum means no failure.
  size_t fail_at = SIZE_MAX;
  // Total allocation attempts, including attempts after failure.
  size_t attempts = 0;
  // Outstanding result allocations.
  size_t live = 0;

  loomc_allocator_t allocator() { return {this, Control}; }
  static loomc_status_t Control(void* self, loomc_allocator_command_t command,
                                const void* params, void** pointer) {
    auto* state = static_cast<AllocationState*>(self);
    const bool had_allocation = (command == LOOMC_ALLOCATOR_COMMAND_FREE ||
                                 command == LOOMC_ALLOCATOR_COMMAND_REALLOC) &&
                                *pointer != nullptr;
    if (command != LOOMC_ALLOCATOR_COMMAND_FREE &&
        state->attempts++ == state->fail_at) {
      return loomc_status_from_code(LOOMC_STATUS_RESOURCE_EXHAUSTED);
    }
    const auto allocator = loomc_allocator_system();
    loomc_status_t status =
        allocator.ctl(allocator.self, command, params, pointer);
    if (loomc_status_is_ok(status)) {
      if (command == LOOMC_ALLOCATOR_COMMAND_FREE) {
        state->live -= had_allocation;
      } else {
        state->live += !had_allocation;
      }
    }
    return status;
  }
};

class InvocationReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static const loom_pass_descriptor_t descriptor = {
        .key = IREE_SVL("test.reject"),
        .info =
            []() {
              static const loom_pass_info_t info = {
                  .name = IREE_SVL("test.reject"),
                  .kind = LOOM_PASS_MODULE,
              };
              return &info;
            },
        .module_run =
            [](loom_pass_t* pass, loom_module_t* module) {
              const loom_diagnostic_emission_t emission = {
                  .module = module,
                  .error = LOOM_ERR_EXPECT_001,
              };
              return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
            },
    };
    static const loom_pass_registry_t registry = {
        .descriptors = &descriptor,
        .descriptor_count = 1,
    };
    static const loom_target_provider_t provider = {.pass_registry = &registry};
    static const loom_target_provider_t* const providers[] = {&provider};
    static const loom_target_provider_set_t provider_set = {
        .providers = providers,
        .provider_count = IREE_ARRAYSIZE(providers),
    };
    loomc_target_environment_t* environment = nullptr;
    LOOMC_ASSERT_OK(loomc_target_environment_create_from_provider_set(
        &provider_set, loomc_allocator_system(), &environment));
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>
        environment_owner(environment);
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .target_environment = environment,
    };
    const loomc_context_options_t context_options = {.next = &target_options};
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                         loomc_allocator_system(), &context));
    context_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    workspace_.reset(workspace);
    // No authored program is needed to test invocation ownership and options.
    const loomc_source_options_t source_options = {
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &source));
    HandlePtr<loomc_source_t, loomc_source_release> source_owner(source);
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_text_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(), &module,
        &result));
    ResultPtr result_owner(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    module_.reset(module);
    loomc_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_compiler_create(context, nullptr,
                                          loomc_allocator_system(), &compiler));
    compiler_.reset(compiler);
  }

  void Prepare(const char* pipeline) {
    loomc_pass_program_t* program = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_create_from_pipeline_text(
        context_.get(), loomc_make_cstring_view(pipeline), nullptr,
        loomc_allocator_system(), &program, &result));
    ResultPtr owner(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    program_.reset(program);
  }

  loomc_status_t Run(const void* next, bool compile,
                     loomc_allocator_t allocator, loomc_result_t** out_result) {
    if (compile) {
      const loomc_compile_options_t options = {.next = next};
      return loomc_compile_module(compiler_.get(), workspace_.get(),
                                  program_.get(), module_.get(), &options,
                                  allocator, out_result);
    }
    const loomc_pass_run_options_t options = {.next = next};
    return loomc_pass_program_run(program_.get(), workspace_.get(),
                                  module_.get(), &options, allocator,
                                  out_result);
  }

  // Compiler owners are deliberately released before reading returned reports.
  HandlePtr<loomc_context_t, loomc_context_release> context_;
  // Recycled invocation storage, not retained by reports.
  HandlePtr<loomc_workspace_t, loomc_workspace_release> workspace_;
  // Compiler for the compile-module entry point.
  HandlePtr<loomc_compiler_t, loomc_compiler_release> compiler_;
  // Empty module executing module-anchored passes.
  HandlePtr<loomc_module_t, loomc_module_release> module_;
  // Program prepared outside each invocation.
  HandlePtr<loomc_pass_program_t, loomc_pass_program_release> program_;
};

TEST_F(InvocationReportTest, ResultOwnsSuccessAndDiagnosedFailureReports) {
  // Cover outcome, invocation entry point, format, and verbosity independently.
  ResultPtr results[16];
  for (int i = 0; i < 16; ++i) {
    SCOPED_TRACE(i);
    const bool rejected = (i & 1) != 0;
    Prepare(rejected ? "symbol-dce,test.reject" : "symbol-dce");
    std::string identifier = "passes.json";
    const loomc_compile_report_options_t target_report = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        .mode = (i & 8) ? LOOMC_COMPILE_REPORT_MODE_DETAILS
                        : LOOMC_COMPILE_REPORT_MODE_SUMMARY,
        .format = (i & 4) ? LOOMC_COMPILE_REPORT_FORMAT_TEXT
                          : LOOMC_COMPILE_REPORT_FORMAT_JSON,
    };
    const loomc_pass_report_options_t report = {
        .type = LOOMC_STRUCTURE_TYPE_PASS_REPORT_OPTIONS,
        .next = &target_report,
        .identifier = loomc_make_cstring_view(identifier.c_str()),
    };
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(
        Run(&report, (i & 2) != 0, loomc_allocator_system(), &result));
    results[i].reset(result);
    EXPECT_EQ(loomc_result_succeeded(result), !rejected);
    ASSERT_EQ(loomc_result_artifact_count(result), 2u);
    identifier.assign(identifier.size(), 'x');
  }
  program_.reset();
  module_.reset();
  compiler_.reset();
  workspace_.reset();
  context_.reset();
  for (int i = 0; i < 16; ++i) {
    SCOPED_TRACE(i);
    const auto* artifact = loomc_result_artifact_at(results[i].get(), 0);
    EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_REPORT);
    EXPECT_TRUE(loomc_string_view_equal(
        artifact->format,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_PASS_REPORT_JSON)));
    EXPECT_TRUE(loomc_string_view_equal(
        artifact->identifier, loomc_make_cstring_view("passes.json")));
    loomc_byte_span_t contents = {};
    LOOMC_ASSERT_OK(loomc_byte_sequence_clone(
        artifact->contents, loomc_allocator_system(), &contents));
    const std::string json(reinterpret_cast<const char*>(contents.data),
                           contents.data_length);
    EXPECT_NE(json.find("\"pass\":\"symbol-dce\""), std::string::npos);
    EXPECT_NE(json.find("\"duration_ns\":0"), std::string::npos);
    EXPECT_EQ(json.find("\"pass\":\"test.reject\"") != std::string::npos,
              (i & 1) != 0);
    loomc_allocator_free(loomc_allocator_system(),
                         const_cast<uint8_t*>(contents.data));

    artifact = loomc_result_artifact_at(results[i].get(), 1);
    const bool text = (i & 4) != 0;
    EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_REPORT);
    EXPECT_TRUE(loomc_string_view_equal(
        artifact->format,
        loomc_make_cstring_view(
            text ? LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT
                 : LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON)));
    EXPECT_TRUE(loomc_string_view_is_empty(artifact->identifier));
    LOOMC_ASSERT_OK(loomc_byte_sequence_clone(
        artifact->contents, loomc_allocator_system(), &contents));
    const std::string target_contents(
        reinterpret_cast<const char*>(contents.data), contents.data_length);
    EXPECT_NE(target_contents.find((i & 1) ? "FAILED_PRECONDITION" : "OK"),
              std::string::npos);
    if ((i & 1) && (i & 8) && !text) {
      EXPECT_NE(target_contents.find("\"diagnostics\":[{"), std::string::npos);
      EXPECT_NE(target_contents.find("\"error_id\":\"ERR_EXPECT_001\""),
                std::string::npos);
    }
    loomc_allocator_free(loomc_allocator_system(),
                         const_cast<uint8_t*>(contents.data));
  }
}

TEST_F(InvocationReportTest, RejectsInvalidAndConflictingPassOptions) {
  Prepare("symbol-dce");
  for (int invalid_case = 0; invalid_case < 7; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    loomc_pass_report_options_t report = {
        .type = LOOMC_STRUCTURE_TYPE_PASS_REPORT_OPTIONS,
    };
    const auto duplicate = report;
    loom_pass_report_t native = {};
    loomc_native_pass_report_options_t native_options = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
        .report = &native,
    };
    const void* next = &report;
    switch (invalid_case) {
      case 0:
        report.structure_size = 1;
        break;
      case 1:
        report.flags = UINT32_MAX;
        break;
      case 2:
        report.identifier.size = 1;
        break;
      case 3:
        report.next = &duplicate;
        break;
      case 4:
        report.next = &native_options;
        break;
      case 5:
        native_options.next = &report;
        next = &native_options;
        break;
      case 6:
        report.type = LOOMC_STRUCTURE_TYPE_NONE;
        break;
    }
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                           Run(next, false, loomc_allocator_system(), &result));
    EXPECT_EQ(result, nullptr);
  }
}

TEST_F(InvocationReportTest, RejectsInvalidAndConflictingCompileOptions) {
  Prepare("symbol-dce");
  for (int invalid_case = 0; invalid_case < 9; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    loomc_compile_report_options_t report = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        .mode = LOOMC_COMPILE_REPORT_MODE_DETAILS,
    };
    const auto duplicate = report;
    loom_target_compile_report_t native = {};
    loomc_native_compile_report_options_t native_options = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
        .report = &native,
    };
    const void* next = &report;
    switch (invalid_case) {
      case 0:
        report.structure_size = 1;
        break;
      case 1:
        report.mode = static_cast<loomc_compile_report_mode_t>(3);
        break;
      case 2:
        report.format = static_cast<loomc_compile_report_format_t>(2);
        break;
      case 3:
        report.identifier.size = 1;
        break;
      case 4:
        report.next = &duplicate;
        break;
      case 5:
        report.next = &native_options;
        break;
      case 6:
        native_options.next = &report;
        next = &native_options;
        break;
      case 7:
        report.type = LOOMC_STRUCTURE_TYPE_NONE;
        break;
      case 8:
        report.mode = LOOMC_COMPILE_REPORT_MODE_NONE;
        report.identifier = loomc_make_cstring_view("report");
        break;
    }
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                           Run(next, false, loomc_allocator_system(), &result));
    EXPECT_EQ(result, nullptr);
  }
}

TEST_F(InvocationReportTest,
       CollectionIsOptInAndAnEmptyProgramStillHasAReport) {
  loomc_pass_program_t* program = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      context_.get(), nullptr, loomc_allocator_system(), &program));
  program_.reset(program);
  for (int variant = 0; variant < 4; ++variant) {
    const loomc_compile_report_options_t target_report = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        .mode = variant == 3 ? LOOMC_COMPILE_REPORT_MODE_DETAILS
                             : LOOMC_COMPILE_REPORT_MODE_NONE,
    };
    const loomc_pass_report_options_t report = {
        .type = LOOMC_STRUCTURE_TYPE_PASS_REPORT_OPTIONS,
        .flags = LOOMC_PASS_REPORT_FLAG_TIMING,
    };
    const void* next = nullptr;
    if (variant == 1) {
      next = &report;
    } else if (variant >= 2) {
      next = &target_report;
    }
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(Run(next, false, loomc_allocator_system(), &result));
    ResultPtr owner(result);
    EXPECT_TRUE(loomc_result_succeeded(result));
    EXPECT_EQ(loomc_result_artifact_count(result), variant & 1);
  }
}

TEST_F(InvocationReportTest,
       AllocationFailureStopsWithoutPublishingOrRetrying) {
  Prepare("symbol-dce");
  for (bool text : {false, true}) {
    const loomc_compile_report_options_t target_report = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        .mode = LOOMC_COMPILE_REPORT_MODE_DETAILS,
        .format = text ? LOOMC_COMPILE_REPORT_FORMAT_TEXT
                       : LOOMC_COMPILE_REPORT_FORMAT_JSON,
    };
    const loomc_pass_report_options_t report = {
        .type = LOOMC_STRUCTURE_TYPE_PASS_REPORT_OPTIONS,
        .next = &target_report,
    };
    for (size_t fail_at = 0;; ++fail_at) {
      AllocationState allocation = {.fail_at = fail_at};
      loomc_result_t* result = nullptr;
      loomc_status_t status =
          Run(&report, false, allocation.allocator(), &result);
      const bool succeeded = loomc_status_is_ok(status);
      if (succeeded) {
        EXPECT_EQ(loomc_result_artifact_count(result), 2u);
      } else {
        LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_RESOURCE_EXHAUSTED, status);
        EXPECT_EQ(result, nullptr);
        EXPECT_EQ(allocation.attempts, fail_at + 1);
      }
      loomc_result_release(result);
      EXPECT_EQ(allocation.live, 0u);
      if (succeeded) {
        break;
      }
    }
  }
}

}  // namespace
