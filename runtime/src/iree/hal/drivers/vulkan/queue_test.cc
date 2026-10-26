// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/queue.h"

#include <atomic>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {
namespace {

struct CompletionState {
  // Number of times the queue invoked the completion action.
  std::atomic<int> callback_count{0};
  // Status observed by the completion action.
  std::atomic<iree_status_code_t> status_code{IREE_STATUS_OK};
  // Number of action resources released by their final owner.
  std::atomic<int> destroy_count{0};
};

struct CompletionResource {
  // Ref-counted storage held by the caller and accepted submission.
  iree_hal_resource_t resource;
  // Borrowed observations that outlive submission completion.
  CompletionState* state;
};

static void DestroyCompletionResource(iree_hal_resource_t* base_resource) {
  auto* resource = reinterpret_cast<CompletionResource*>(base_resource);
  ++resource->state->destroy_count;
  delete resource;
}

static const iree_hal_resource_vtable_t kCompletionResourceVTable = {
    .destroy = DestroyCompletionResource,
};

static iree_hal_vulkan_queue_completion_action_t MakeCompletionAction(
    CompletionState* state) {
  auto* resource = new CompletionResource;
  iree_hal_resource_initialize(&kCompletionResourceVTable, &resource->resource);
  resource->state = state;
  return {
      .fn =
          [](void* user_data, const iree_status_t status) {
            auto* state = static_cast<CompletionState*>(user_data);
            state->status_code = iree_status_code(status);
            ++state->callback_count;
          },
      .user_data = state,
      .resource = &resource->resource,
  };
}

class VulkanQueueTest : public CtsTestBase<> {};

TEST_P(VulkanQueueTest, RejectedSubmissionDoesNotInvokeCompletion) {
  SemaphoreList waits(device_, {0}, {1});
  SemaphoreList signals(device_, {0}, {1});
  iree_hal_semaphore_fail(waits.semaphores[0],
                          iree_status_from_code(IREE_STATUS_ABORTED));
  CompletionState state;
  auto action = MakeCompletionAction(&state);
  Status status(iree_hal_vulkan_queue_submit_barrier_with_action(
      reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_), waits,
      signals, /*barriers=*/nullptr, action));
  iree_hal_resource_release(action.resource);

  EXPECT_EQ(status.code(), StatusCode::kAborted);
  EXPECT_EQ(state.callback_count.load(), 0);
  EXPECT_EQ(state.destroy_count.load(), 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(VulkanQueueTest, AcceptedSubmissionCompletesOnceAndReleasesAction) {
  for (iree_status_code_t completion_status :
       {IREE_STATUS_OK, IREE_STATUS_ABORTED}) {
    SCOPED_TRACE(completion_status);
    SemaphoreList waits(device_, {0}, {1});
    SemaphoreList signals(device_, {0}, {1});
    CompletionState state;
    auto action = MakeCompletionAction(&state);
    Status status(iree_hal_vulkan_queue_submit_barrier_with_action(
        reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_), waits,
        signals, /*barriers=*/nullptr, action));
    iree_hal_resource_release(action.resource);
    IREE_ASSERT_OK(status.release());
    EXPECT_EQ(state.callback_count.load(), 0);
    EXPECT_EQ(state.destroy_count.load(), 0);

    if (completion_status == IREE_STATUS_OK) {
      IREE_ASSERT_OK(iree_hal_semaphore_signal(waits.semaphores[0], 1,
                                               /*frontier=*/nullptr));
    } else {
      iree_hal_semaphore_fail(waits.semaphores[0],
                              iree_status_from_code(completion_status));
    }
    IREE_EXPECT_STATUS_IS(
        completion_status,
        iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(state.callback_count.load(), 1);
    EXPECT_EQ(state.status_code.load(), completion_status);
    EXPECT_EQ(state.destroy_count.load(), 1);
  }
}

CTS_REGISTER_TEST_SUITE(VulkanQueueTest);

}  // namespace
}  // namespace iree::hal::cts
