// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/executables.h"

#include "iree/base/tooling/flags.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/parameter/weights_test_base.h"
#include "loom/serve/runtime/program/program.h"

IREE_FLAG(string, executable_sources, "", "Request executable source fixture.");

namespace loom_serve::testing {

class ExecutablesTest : public WeightsTest {
 protected:
  void TearDown() override {
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    if (executables_) {
      loom_serve_executables_release(executables_);
    }
    loom_serve_executables_free(executables_);
    loom_serve_program_free(control_);
    iree_vm_module_release(runner_);
    loom_serve_stage_table_free(table_);
    loom_serve_preparation_free(preparation_);
    iree_vm_environment_free(environment_);
    loom_serve_parameter_bank_retire(bank_, nullptr, nullptr);
    loom_serve_parameter_bank_await_retirement(bank_);
    loom_serve_residency_free(residency_);
    IREE_EXPECT_OK(loom_serve_parameter_bank_free(bank_));
    iree_notification_deinitialize(&notification_);
    WeightsTest::TearDown();
  }

  iree_status_t Declare(const char* root, bool swapped, int32_t count,
                        int64_t tag, uint64_t limit,
                        loom_serve_preparation_t** out_preparation,
                        const char* entry = "declare") {
    const std::string policy = directory_ + "/policy.loom";
    const std::string strings[] = {
        root, swapped ? adapter_path_.path() : path_.path(),
        swapped ? path_.path() : adapter_path_.path(), policy};
    iree_vm_variant_t arguments[7] = {};
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < IREE_ARRAYSIZE(strings) && iree_status_is_ok(status);
         ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      status =
          iree_vm_buffer_create(strings[i].size(), 1, host_allocator_, &buffer);
      if (iree_status_is_ok(status)) {
        iree_byte_span_t bytes;
        status = iree_vm_buffer_map_write(buffer, 0, strings[i].size(), &bytes);
        if (iree_status_is_ok(status)) {
          memcpy(bytes.data, strings[i].data(), strings[i].size());
        }
        arguments[i] =
            iree_vm_buffer_variant_from_ptr_move(&vm_types_, &buffer);
      }
    }
    arguments[4] = iree_vm_variant_from_i32(count);
    arguments[5] = iree_vm_variant_from_i64(tag);
    arguments[6] = iree_vm_variant_from_i64(static_cast<int64_t>(limit));
    if (iree_status_is_ok(status)) {
      loom_serve_compilation_report_t report = {};
      status = loom_serve_preparation_allocate(
          loom_serve_host_target_environment(
              loom_serve_device_host(device_owner_)),
          environment_, &sources_,
          iree_make_cstring_view(FLAG_executable_sources),
          iree_make_cstring_view(entry),
          iree_vm_variant_span_from_array(arguments),
          iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &report,
          host_allocator_, out_preparation);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    return status;
  }

  iree_status_t Create(iree_host_size_t capacity = 4,
                       iree_host_size_t max_bytes = 16 * 1024 * 1024) {
    iree_notification_initialize(&notification_);
    WriteCheckpoint(adapter_path_.path(), 100);
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(host_allocator_, &environment_));
    IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &vm_types_));
    const iree_vm_ref_type_table_t* types = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment_, &types));
    IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(types, &hal_types_));
    IREE_RETURN_IF_ERROR(Declare("domains", false, 2, 0, 0, &preparation_));
    IREE_RETURN_IF_ERROR(Compile(0, "domains"));
    IREE_RETURN_IF_ERROR(loom_serve_parameter_bank_allocate(
        device_owner_, loom_serve_jit_stage_program(compiled_[0]),
        host_allocator_, &bank_));
    const auto* declaration = loom_serve_preparation_stage(preparation_, 0);
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      status = loom_serve_weights_prepare(
          loom_serve_parameter_bank_weights(bank_, i),
          loom_serve_device_host(device_owner_), jit_,
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, &sources_,
          &declaration->parameters[i]);
    }
    IREE_RETURN_IF_ERROR(status);
    auto* group = loom_serve_parameter_bank_group(bank_);
    IREE_RETURN_IF_ERROR(loom_serve_residency_allocate(
        loom_serve_device_residency_cache(device_owner_), execution_, 1, &group,
        host_allocator_, &residency_));
    loom_serve_executable_options_t options = {
        .device = device_owner_,
        .jit = jit_,
        .preparation = preparation_,
        .parameters = bank_,
        .recording =
            {
                .queue_family = iree_hal_queue_family(dispatch_),
                .parameters = loom_serve_parameter_bank_buffers(bank_),
                .binding_count = 1,
            },
        .cache = {capacity, max_bytes},
    };
    IREE_RETURN_IF_ERROR(loom_serve_stage_table_allocate(
        &options.recording, nullptr, 1, compiled_.data(), host_allocator_,
        &table_));
    const loom_serve_module_options_t module_options = {
        .binding_capacity = 1,
        .stages = {table_->part_count, table_->parts},
    };
    IREE_RETURN_IF_ERROR(loom_serve_module_create(
        &hal_types_, execution_, module_options, host_allocator_, &runner_));
    options.stages = table_;
    options.runner = runner_;
    IREE_RETURN_IF_ERROR(loom_serve_executables_allocate(
        &options, host_allocator_, &executables_));
    loom_serve_executables_set_observer(
        executables_, {[](void* self) {
                         iree_notification_post(
                             static_cast<iree_notification_t*>(self),
                             IREE_ALL_WAITERS);
                       },
                       &notification_});
    const iree_string_view_t root = IREE_SV("submit");
    loom_serve_compilation_report_t report = {};
    IREE_RETURN_IF_ERROR(loom_serve_program_allocate(
        loom_serve_host_target_environment(
            loom_serve_device_host(device_owner_)),
        environment_, &sources_,
        iree_make_cstring_view(FLAG_executable_sources), 1, &root,
        {&runner_, 1}, &report, host_allocator_, &control_));
    return iree_vm_process_lookup_function(loom_serve_program_process(control_),
                                           IREE_SV("model"), root, &submit_);
  }

  iree_status_t Select(const char* root, bool swapped, int32_t count = 2,
                       int64_t tag = 0, uint64_t limit = 0) {
    IREE_RETURN_IF_ERROR(Begin(root, swapped, count, tag, limit));
    return Advance(false);
  }

  iree_status_t Begin(const char* root, bool swapped, int32_t count = 2,
                      int64_t tag = 0, uint64_t limit = 0,
                      const char* entry = "declare") {
    loom_serve_preparation_t* declaration = nullptr;
    IREE_RETURN_IF_ERROR(
        Declare(root, swapped, count, tag, limit, &declaration, entry));
    loom_serve_executables_capture(executables_);
    loom_serve_executables_declare(executables_, declaration);
    return iree_ok_status();
  }

  iree_status_t Advance(bool cancel) {
    bool complete = false;
    iree_status_t status = iree_ok_status();
    while (!complete && iree_status_is_ok(status)) {
      const uint32_t ticket = iree_notification_prepare_wait(&notification_);
      status = loom_serve_executables_advance(executables_, cancel, &complete);
      if (!complete && iree_status_is_ok(status)) {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      } else {
        iree_notification_cancel_wait(&notification_);
      }
    }
    return status;
  }

  void Execute(int32_t first, int32_t stride = 3) {
    IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residency_));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(1),
        iree_hal_buffer_variant_from_ptr_borrowed(&hal_types_, output_buffer_)};
    iree_vm_variant_t results[1] = {};
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(control_), submit_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = output_buffer_,
                .target = output_.data(),
                .length = sizeof(output_),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution_, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
    loom_serve_residency_release(residency_);
    for (size_t i = 0; i < output_.size(); ++i) {
      EXPECT_EQ(output_[i], first + stride * i);
    }
    loom_serve_executables_release(executables_);
  }

  // Authored locators are rooted in the test's declared runfiles namespace.
  const loom_serve_sources_t sources_ = {.base_directory = IREE_SV(".")};
  // VM identities outliving every source capture and native module.
  iree_vm_environment_t* environment_ = nullptr;
  // Canonical builtin VM reference identities.
  iree_vm_ref_types_t vm_types_ = {};
  // Canonical native buffer identities.
  iree_hal_module_types_t hal_types_ = {};
  // Immutable registration declarations.
  loom_serve_preparation_t* preparation_ = nullptr;
  // Registration's independently loaded roots with colliding tensor names.
  loom_serve_parameter_bank_t* bank_ = nullptr;
  // Atomic admission for all registration-owned parameters.
  loom_serve_residency_t* residency_ = nullptr;
  // Original recordings borrowed by the executable owner.
  loom_serve_stage_table_t* table_ = nullptr;
  // Native runner consuming the selected request table.
  iree_vm_module_t* runner_ = nullptr;
  // Source-selected request ownership under test.
  loom_serve_executables_t* executables_ = nullptr;
  // Real VM caller of the runner ABI.
  loom_serve_program_t* control_ = nullptr;
  // Bound request entry, borrowing control_.
  iree_vm_function_t submit_ = {};
  // Outer-thread completion notification; no worker waits here.
  iree_notification_t notification_ = {};
};

TEST_P(ExecutablesTest, RebindsCheckpointDataWithoutRecompilingSharedCode) {
  IREE_ASSERT_OK(Create(1));
  const auto* base = loom_serve_parameter_bank_buffers(bank_)[0];
  const auto before = compilation_.commands;
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  EXPECT_EQ(compilation_.commands, before + 1);
  const auto kernels = compilation_.kernels;
  // Same compiler inputs, different checkpoint bindings. Evicting the first
  // choice must not retire the code or retain its declaration as a cache key.
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, before + 1);
  EXPECT_EQ(compilation_.kernels, kernels);
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  EXPECT_EQ(compilation_.commands, before + 1);
  EXPECT_EQ(loom_serve_parameter_bank_buffers(bank_)[0], base);
  EXPECT_EQ(loom_serve_residency_activity(residency_).activations, 2u);
}

TEST_P(ExecutablesTest, SelectsSubsetWithoutChangingParameterStorage) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(90));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("raw", true, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(390));
  EXPECT_EQ(compilation_.commands, commands);
  // An exact repeated selection uses the already-recorded table too.
  IREE_ASSERT_OK(Select("raw", true, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(390));
  EXPECT_EQ(compilation_.commands, commands);
}

TEST_P(ExecutablesTest, DisabledRetentionReleasesSharedCodeAndRecordsAgain) {
  IREE_ASSERT_OK(Create(0));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands + 2);
}

TEST_P(ExecutablesTest, ByteBudgetEvictsEvenWhenEntryCapacityRemains) {
  IREE_ASSERT_OK(Create(4, 1));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands + 2);
}

TEST_P(ExecutablesTest, CancellationJoinsRecordingWithoutActivatingParameters) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  const auto commands = compilation_.commands;
  const auto activity = loom_serve_residency_activity(residency_);
  IREE_ASSERT_OK(Begin("domains", true));
  bool complete = false;
  IREE_ASSERT_OK(
      loom_serve_executables_advance(executables_, false, &complete));
  IREE_ASSERT_OK(Advance(true));
  loom_serve_executables_release(executables_);
  EXPECT_EQ(compilation_.commands, commands);
  EXPECT_EQ(loom_serve_residency_activity(residency_).activations,
            activity.activations);
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands);
}

TEST_P(ExecutablesTest, MixedImplicitAndExplicitDomainsRejectBeforeBinding) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Begin("domains", false, 2, 0, 0, "mixed"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Advance(false));
  loom_serve_executables_release(executables_);
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
}

TEST_P(ExecutablesTest, LayoutMismatchDoesNotPoisonSharedCode) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  const auto commands = compilation_.commands;
  // The same code must still validate each request's complete declaration.
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Select("domains", true, 1));
  loom_serve_executables_release(executables_);
  EXPECT_EQ(compilation_.commands, commands);
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Select("target", false, 1));
  loom_serve_executables_release(executables_);
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
}

TEST_P(ExecutablesTest, SharedCodeChecksEachRequestsWorkspaceLimit) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("domains_workspace", false));
  ASSERT_NO_FATAL_FAILURE(Execute(870, 9));
  const auto commands = compilation_.commands;
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Select("domains_workspace", false, 2, 1, 1));
  loom_serve_executables_release(executables_);
  EXPECT_EQ(compilation_.commands, commands);
  IREE_ASSERT_OK(Select("domains_workspace", true, 2, 2, 4096));
  ASSERT_NO_FATAL_FAILURE(Execute(570, 9));
  EXPECT_EQ(compilation_.commands, commands);
}

INSTANTIATE_TEST_SUITE_P(Native, ExecutablesTest, ::testing::Bool());

}  // namespace loom_serve::testing
