// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/iocp/proactor.h"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>

#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/semaphore.h"
#include "iree/async/socket.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

struct CompletionState {
  int call_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
};

struct MessageState {
  int call_count = 0;
  uint64_t value = 0;
};

struct ControlledAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  int allocations_before_failure = -1;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ControlledAllocator*>(self);
    const bool is_allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_CALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_REALLOC;
    if (is_allocation && allocator->allocations_before_failure == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "intentional allocation failure");
    }
    if (is_allocation && allocator->allocations_before_failure > 0) {
      --allocator->allocations_before_failure;
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t value() { return {.self = this, .ctl = Control}; }
};

static void RecordCompletion(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  auto* state = static_cast<CompletionState*>(user_data);
  ++state->call_count;
  state->status_code = iree_status_code(status);
  iree_status_free(status);
}

static void RecordMessage(iree_async_proactor_t* proactor,
                          uint64_t message_data, void* user_data) {
  (void)proactor;
  auto* state = static_cast<MessageState*>(user_data);
  ++state->call_count;
  state->value = message_data;
}

template <typename T>
static void InitializeOperation(T* operation, iree_async_operation_type_t type,
                                CompletionState* completion) {
  iree_async_operation_zero(&operation->base, sizeof(*operation));
  iree_async_operation_initialize(
      &operation->base, type, IREE_ASYNC_OPERATION_FLAG_NONE,
      completion ? RecordCompletion : nullptr, completion);
}

// Replaces the component's owned handle with NULL while preserving the same
// completion-port object through a duplicate. Synthetic posts then fail
// through the real PostQueuedCompletionStatus call. Restoring the duplicate
// returns ownership of a valid handle to the component.
static iree_status_t DisableCompletionPosting(
    iree_async_proactor_iocp_t* proactor, HANDLE* out_preserved_port) {
  *out_preserved_port = NULL;
  HANDLE completion_port = (HANDLE)proactor->completion_port.handle;
  HANDLE preserved_port = NULL;
  if (!DuplicateHandle(GetCurrentProcess(), completion_port,
                       GetCurrentProcess(), &preserved_port, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    DWORD error_code = GetLastError();
    return iree_make_status(iree_status_code_from_win32_error(error_code),
                            "DuplicateHandle failed (error %lu)",
                            (unsigned long)error_code);
  }
  if (!CloseHandle(completion_port)) {
    DWORD error_code = GetLastError();
    iree_status_t status = iree_make_status(
        iree_status_code_from_win32_error(error_code),
        "CloseHandle failed while disabling completion posts (error %lu)",
        (unsigned long)error_code);
    if (!CloseHandle(preserved_port)) {
      DWORD close_error_code = GetLastError();
      status = iree_status_join(
          status,
          iree_make_status(iree_status_code_from_win32_error(close_error_code),
                           "CloseHandle failed while unwinding duplicate "
                           "completion port (error %lu)",
                           (unsigned long)close_error_code));
    }
    return status;
  }
  proactor->completion_port.handle = 0;
  *out_preserved_port = preserved_port;
  return iree_ok_status();
}

static void RestoreCompletionPosting(iree_async_proactor_iocp_t* proactor,
                                     HANDLE preserved_port) {
  proactor->completion_port.handle = (uintptr_t)preserved_port;
}

static void WaitForFallbackCompletionReady(iree_async_iocp_carrier_t* carrier) {
  while (iree_atomic_load(&carrier->fallback_completion_state,
                          iree_memory_order_acquire) !=
         IREE_ASYNC_IOCP_FALLBACK_COMPLETION_READY) {
    std::this_thread::yield();
  }
}

class IocpProactorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_iocp(
        options, iree_allocator_system(), &proactor_));
  }

  void TearDown() override { iree_async_proactor_release(proactor_); }

  iree_async_proactor_iocp_t* iocp() {
    return iree_async_proactor_iocp_cast(proactor_);
  }

  iree_async_proactor_t* proactor_ = nullptr;
};

struct ReleaseIoCancellation {
  void operator()(iree_async_iocp_carrier_t* carrier) const {
    iree_async_proactor_iocp_release_io_cancel(carrier->proactor, carrier);
  }
};
using IoCancellationBorrow =
    std::unique_ptr<iree_async_iocp_carrier_t, ReleaseIoCancellation>;

struct FileIssueResultCase {
  // Native read or write whose admission returned this result.
  iree_async_operation_type_t type;
  // GetLastError result from an immediately completed native call.
  uint32_t error_code;
  // Public result of the one terminal completion.
  iree_status_code_t expected_status;
};

class IocpFileIssueResultTest
    : public IocpProactorTest,
      public ::testing::WithParamInterface<FileIssueResultCase> {};

TEST_P(IocpFileIssueResultTest, ImmediateResultReturnsCarrierExactlyOnce) {
  const auto test_case = GetParam();
  CompletionState completion;
  iree_async_file_read_operation_t read;
  iree_async_file_write_operation_t write;
  InitializeOperation(&read, IREE_ASYNC_OPERATION_TYPE_FILE_READ, &completion);
  InitializeOperation(&write, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
                      &completion);
  iree_async_operation_t* operation =
      test_case.type == IREE_ASYNC_OPERATION_TYPE_FILE_READ ? &read.base
                                                            : &write.base;
  // The native dependency returned without retaining an I/O borrow. Inject
  // that result at the real admission boundary using its reserved carrier.
  iree_async_iocp_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_iocp_acquire_carrier(
      iocp(), IREE_ASYNC_IOCP_CARRIER_FILE_IO, operation, 0, &carrier));
  operation->next = reinterpret_cast<iree_async_operation_t*>(carrier);
  IREE_ASSERT_OK(iree_async_proactor_iocp_finish_file_issue(
      iocp(), carrier, test_case.error_code));
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);
  while (completion.call_count == 0) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, test_case.expected_status);
  EXPECT_EQ(read.bytes_read, 0u);
  EXPECT_EQ(write.bytes_written, 0u);
  EXPECT_EQ(operation->next, nullptr);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr));
  EXPECT_EQ(completion.call_count, 1);
}

TEST_P(IocpFileIssueResultTest, ImmediateResultJoinsCancellationBeforeReuse) {
  const auto test_case = GetParam();
  CompletionState completion;
  iree_async_file_read_operation_t read;
  iree_async_file_write_operation_t write;
  InitializeOperation(&read, IREE_ASYNC_OPERATION_TYPE_FILE_READ, &completion);
  InitializeOperation(&write, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
                      &completion);
  iree_async_operation_t* operation =
      test_case.type == IREE_ASYNC_OPERATION_TYPE_FILE_READ ? &read.base
                                                            : &write.base;
  iree_async_iocp_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_iocp_acquire_carrier(
      iocp(), IREE_ASYNC_IOCP_CARRIER_FILE_IO, operation, 0, &carrier));
  operation->next = reinterpret_cast<iree_async_operation_t*>(carrier);
  IoCancellationBorrow cancellation(
      iree_async_proactor_iocp_acquire_io_cancel(operation));
  ASSERT_EQ(cancellation.get(), carrier);

  // Return an immediate native result while cancellation still holds the
  // OVERLAPPED identity. Posting the result must not sever that ownership.
  IREE_ASSERT_OK(iree_async_proactor_iocp_finish_file_issue(
      iocp(), carrier, test_case.error_code));
  while (!iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                           IREE_ASYNC_IOCP_INTERNAL_FLAG_IO_RETIRED)) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(operation->next,
            reinterpret_cast<iree_async_operation_t*>(carrier));
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);
  cancellation.reset();
  while (!completion.call_count) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, test_case.expected_status);
  EXPECT_EQ(operation->next, nullptr);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
}

INSTANTIATE_TEST_SUITE_P(
    NativeResults, IocpFileIssueResultTest,
    ::testing::Values(
        FileIssueResultCase{IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                            ERROR_HANDLE_EOF, IREE_STATUS_OK},
        FileIssueResultCase{IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                            ERROR_ACCESS_DENIED, IREE_STATUS_PERMISSION_DENIED},
        FileIssueResultCase{IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                            ERROR_OPERATION_ABORTED, IREE_STATUS_CANCELLED},
        FileIssueResultCase{IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
                            ERROR_ACCESS_DENIED, IREE_STATUS_PERMISSION_DENIED},
        FileIssueResultCase{IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
                            ERROR_HANDLE_EOF, IREE_STATUS_RESOURCE_EXHAUSTED}));

TEST_F(IocpProactorTest, FileOpenAndCloseRunOnlyFromPoll) {
  iree::testing::TempFilePath path("iree_iocp_file_owner");
  CompletionState opened;
  iree_async_file_open_operation_t open;
  InitializeOperation(&open, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN, &opened);
  open.path = path.path().c_str();
  open.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ |
                    IREE_ASYNC_FILE_OPEN_FLAG_WRITE |
                    IREE_ASYNC_FILE_OPEN_FLAG_CREATE;
  iree_status_t submit_status = iree_ok_status();
  std::jthread producer([&] {
    submit_status = iree_async_proactor_submit_one(proactor_, &open.base);
  });
  producer.join();
  IREE_ASSERT_OK(submit_status);
  EXPECT_EQ(open.opened_file, nullptr);
  EXPECT_EQ(opened.call_count, 0);
  while (!opened.call_count) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  ASSERT_EQ(opened.status_code, IREE_STATUS_OK);
  iree_async_file_t* file = open.opened_file;
  ASSERT_NE(file, nullptr);

  CompletionState closed;
  iree_async_file_close_operation_t close;
  InitializeOperation(&close, IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE, &closed);
  // Accepted close consumes one reference. Keep a separate observer reference
  // to verify that the native handle stays live until the owner polls.
  iree_async_file_retain(file);
  close.file = file;
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &close.base));
  EXPECT_NE(file->primitive.value.win32_handle, uintptr_t{0});
  EXPECT_EQ(GetFileType((HANDLE)file->primitive.value.win32_handle),
            FILE_TYPE_DISK);
  EXPECT_EQ(closed.call_count, 0);
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &close.base));
  while (!closed.call_count) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(closed.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(iree_atomic_ref_count_load(&file->ref_count), 1);
  EXPECT_EQ(GetFileType((HANDLE)file->primitive.value.win32_handle),
            FILE_TYPE_DISK);

  // Cancellation still consumes the reference transferred to close, but the
  // observer's reference keeps the native file usable for another close.
  closed = {};
  InitializeOperation(&close, IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE, &closed);
  iree_async_file_retain(file);
  close.file = file;
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &close.base));
  while (!closed.call_count) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(closed.status_code, IREE_STATUS_OK);
  EXPECT_EQ(file->primitive.value.win32_handle, uintptr_t{0});
  iree_async_file_release(file);
  EXPECT_TRUE(path.Remove());
}

TEST_F(IocpProactorTest, CancelledOpenNeverCreatesFile) {
  iree::testing::TempFilePath path("iree_iocp_cancelled_open");
  CompletionState completed;
  iree_async_file_open_operation_t open;
  InitializeOperation(&open, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN, &completed);
  open.path = path.path().c_str();
  open.open_flags =
      IREE_ASYNC_FILE_OPEN_FLAG_WRITE | IREE_ASYNC_FILE_OPEN_FLAG_CREATE;
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open.base));
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &open.base));
  while (!completed.call_count) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(completed.call_count, 1);
  EXPECT_EQ(completed.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(open.opened_file, nullptr);
  EXPECT_FALSE(path.Exists());
}

class IocpFileAdmissionTest
    : public IocpProactorTest,
      public ::testing::WithParamInterface<iree_async_operation_type_t> {
 protected:
  void SetUp() override {
    IocpProactorTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    CompletionState opened;
    iree_async_file_open_operation_t open;
    InitializeOperation(&open, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN, &opened);
    open.path = path_.path().c_str();
    open.open_flags =
        IREE_ASYNC_FILE_OPEN_FLAG_READ | IREE_ASYNC_FILE_OPEN_FLAG_WRITE |
        IREE_ASYNC_FILE_OPEN_FLAG_CREATE | IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open.base));
    PollUntil(opened);
    ASSERT_EQ(opened.status_code, IREE_STATUS_OK);
    file_ = open.opened_file;
    ASSERT_NE(file_, nullptr);
    CompletionState written;
    iree_async_file_write_operation_t write;
    InitializeOperation(&write, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE, &written);
    write.file = file_;
    write.buffer = iree_async_span_from_ptr(initial_.data(), initial_.size());
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write.base));
    PollUntil(written);
    ASSERT_EQ(written.status_code, IREE_STATUS_OK);
    ASSERT_EQ(write.bytes_written, initial_.size());
  }

  void TearDown() override {
    iree_async_file_release(file_);
    if (path_.Exists()) {
      EXPECT_TRUE(path_.Remove());
    }
    IocpProactorTest::TearDown();
  }

  void PollUntil(const CompletionState& completion) {
    while (!completion.call_count) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
  }

  iree_async_operation_t* Prepare(CompletionState* completion) {
    if (GetParam() == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
      InitializeOperation(&read_, GetParam(), completion);
      read_.file = file_;
      read_.buffer = iree_async_span_from_ptr(data_.data(), data_.size());
      return &read_.base;
    }
    InitializeOperation(&write_, GetParam(), completion);
    write_.file = file_;
    write_.buffer = iree_async_span_from_ptr(data_.data(), data_.size());
    return &write_.base;
  }

  void ExpectFileContents(const std::array<uint8_t, 4>& expected) {
    std::array<uint8_t, 4> actual = {};
    CompletionState completed;
    iree_async_file_read_operation_t read;
    InitializeOperation(&read, IREE_ASYNC_OPERATION_TYPE_FILE_READ, &completed);
    read.file = file_;
    read.buffer = iree_async_span_from_ptr(actual.data(), actual.size());
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read.base));
    PollUntil(completed);
    EXPECT_EQ(completed.status_code, IREE_STATUS_OK);
    EXPECT_EQ(actual, expected);
  }

  // Private file retained across all native requests in the case.
  iree_async_file_t* file_ = nullptr;
  // File pathname removed only after its native handle closes.
  iree::testing::TempFilePath path_{"iree_iocp_file_admission"};
  // Known initial contents for observing rejected or cancelled writes.
  std::array<uint8_t, 4> initial_ = {1, 2, 3, 4};
  // Stable read destination or write source through the terminal callback.
  std::array<uint8_t, 4> data_ = {5, 6, 7, 8};
  // Placed read storage used only by read cases.
  iree_async_file_read_operation_t read_ = {};
  // Placed write storage used only by write cases.
  iree_async_file_write_operation_t write_ = {};
};

TEST_P(IocpFileAdmissionTest, ProducerOnlyPublishesReservedOwnership) {
  CompletionState completed;
  iree_async_operation_t* operation = Prepare(&completed);
  iree_status_t status = iree_ok_status();
  std::jthread producer(
      [&] { status = iree_async_proactor_submit_one(proactor_, operation); });
  producer.join();
  IREE_ASSERT_OK(status);
  auto* carrier = reinterpret_cast<iree_async_iocp_carrier_t*>(operation->next);
  ASSERT_NE(carrier, nullptr);
  // Reservation captures the cancellation handle, but the producer only
  // queues admission. The poll owner has not issued the native file request.
  EXPECT_EQ(carrier->io_handle, file_->primitive.value.win32_handle);
  EXPECT_EQ(data_, (std::array<uint8_t, 4>{5, 6, 7, 8}));
  EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 2);
  EXPECT_EQ(completed.call_count, 0);
  PollUntil(completed);
  EXPECT_EQ(completed.status_code, IREE_STATUS_OK);
  EXPECT_EQ(completed.call_count, 1);
  EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 1);
  if (GetParam() == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
    EXPECT_EQ(data_, initial_);
  } else {
    ExpectFileContents(data_);
  }
}

TEST_P(IocpFileAdmissionTest, CancelBeforePollNeverIssuesNativeIo) {
  CompletionState completed;
  iree_async_operation_t* operation = Prepare(&completed);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, operation));
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, operation));
  EXPECT_EQ(completed.call_count, 0);
  PollUntil(completed);
  EXPECT_EQ(completed.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(completed.call_count, 1);
  EXPECT_EQ(data_, (std::array<uint8_t, 4>{5, 6, 7, 8}));
  EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 1);
  ExpectFileContents(initial_);
}

TEST_P(IocpFileAdmissionTest, CancellationHoldsOnlyItsOwnTerminalResult) {
  for (bool issue_before_cancel : {false, true}) {
    SCOPED_TRACE(issue_before_cancel);
    CompletionState completed;
    iree_async_operation_t* operation = Prepare(&completed);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, operation));
    if (issue_before_cancel) {
      iree_host_size_t completed_count = 0;
      IREE_ASSERT_OK(iree_async_proactor_iocp_drain_file_submissions(
          iocp(), &completed_count));
      EXPECT_EQ(completed_count, 0);
    }

    // Hold the actual native cancellation boundary across owner progression.
    // Two callers must both return before either the carrier or the retained
    // file can be reused. No native failure or race is injected.
    IoCancellationBorrow first(
        iree_async_proactor_iocp_acquire_io_cancel(operation));
    IoCancellationBorrow second(
        iree_async_proactor_iocp_acquire_io_cancel(operation));
    ASSERT_NE(first.get(), nullptr);
    ASSERT_EQ(first.get(), second.get());
    while (
        !iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                          IREE_ASYNC_IOCP_INTERNAL_FLAG_IO_RETIRED)) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
    EXPECT_EQ(completed.call_count, 0);
    EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 2);

    // A different request on this same native handle continues to completion.
    // Its carrier cannot alias the one still held by cancellation.
    CompletionState independent_completed;
    iree_async_file_read_operation_t independent;
    InitializeOperation(&independent, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                        &independent_completed);
    std::array<uint8_t, 4> independent_data = {};
    independent.file = file_;
    independent.buffer = iree_async_span_from_ptr(independent_data.data(),
                                                  independent_data.size());
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &independent.base));
    EXPECT_NE(independent.base.next, operation->next);
    PollUntil(independent_completed);
    EXPECT_EQ(independent_completed.status_code, IREE_STATUS_OK);
    EXPECT_EQ(completed.call_count, 0);
    EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 2);

    first.reset();
    // Cancelling a result already held for retirement cannot acquire another
    // native borrow or make the first result available prematurely.
    IREE_EXPECT_OK(iree_async_proactor_cancel(proactor_, operation));
    EXPECT_EQ(completed.call_count, 0);
    EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                               iree_memory_order_relaxed),
              1);
    std::jthread canceller([&] { second.reset(); });
    // Poll must be woken by the cancelling thread's return, without another
    // native completion or a timer to make the terminal result progress.
    PollUntil(completed);
    canceller.join();
    EXPECT_EQ(completed.call_count, 1);
    EXPECT_EQ(completed.status_code, IREE_STATUS_CANCELLED);
    EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 1);
    EXPECT_EQ(operation->next, nullptr);
    IREE_EXPECT_OK(iree_async_proactor_cancel(proactor_, operation));

    // Caller-owned storage is reusable after both completion and cancel return.
    completed = {};
    operation = Prepare(&completed);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, operation));
    PollUntil(completed);
    EXPECT_EQ(completed.call_count, 1);
    EXPECT_EQ(completed.status_code, IREE_STATUS_OK);
    EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                               iree_memory_order_relaxed),
              0);
  }
}

TEST_P(IocpFileAdmissionTest, RejectedBatchDoesNotRetainOrIssueFileIo) {
  CompletionState completed;
  iree_async_operation_t* operation = Prepare(&completed);
  iree_async_nop_operation_t malformed;
  InitializeOperation(&malformed, IREE_ASYNC_OPERATION_TYPE_NOP, nullptr);
  iree_async_operation_t* operations[] = {operation, &malformed.base};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_submit(proactor_,
                                 iree_async_operation_list_make(
                                     operations, IREE_ARRAYSIZE(operations))));
  EXPECT_EQ(completed.call_count, 0);
  EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 1);
  EXPECT_EQ(operation->next, nullptr);
  ExpectFileContents(initial_);
  EXPECT_EQ(completed.call_count, 0);
}

TEST_P(IocpFileAdmissionTest, CallbackCanResubmitSameFileOperation) {
  struct ReissueState {
    // Borrowed proactor owning all three executions.
    iree_async_proactor_t* proactor;
    // Number of terminal callbacks observed before resubmitting.
    int count = 0;
    // First I/O or admission failure ends the sequence.
    iree_status_code_t status_code = IREE_STATUS_OK;
  } state{proactor_};
  iree_async_operation_t* operation = Prepare(nullptr);
  operation->user_data = &state;
  operation->completion_fn =
      +[](void* user_data, iree_async_operation_t* operation,
          iree_status_t status, iree_async_completion_flags_t flags) {
        auto* state = static_cast<ReissueState*>(user_data);
        ++state->count;
        state->status_code = iree_status_code(status);
        iree_status_free(status);
        if (state->status_code == IREE_STATUS_OK && state->count < 3) {
          status = iree_async_proactor_submit_one(state->proactor, operation);
          state->status_code = iree_status_code(status);
          iree_status_free(status);
        }
      };
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, operation));
  while (state.count < 3 && state.status_code == IREE_STATUS_OK) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(state.count, 3);
  EXPECT_EQ(state.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 1);
  if (GetParam() == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
    EXPECT_EQ(data_, initial_);
  } else {
    ExpectFileContents(data_);
  }
}

TEST_P(IocpFileAdmissionTest, CancellationJoinsBothSidesOfNativeIssue) {
  for (bool cancel_before_issue : {true, false}) {
    SCOPED_TRACE(cancel_before_issue);
    CompletionState completed;
    iree_async_operation_t* operation = Prepare(&completed);
    HANDLE handle = (HANDLE)file_->primitive.value.win32_handle;
    iree_async_iocp_carrier_t* carrier = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_iocp_acquire_carrier(
        iocp(), IREE_ASYNC_IOCP_CARRIER_FILE_IO, operation, (uintptr_t)handle,
        &carrier));
    iree_async_operation_acquire_resources(operation);
    operation->next = reinterpret_cast<iree_async_operation_t*>(carrier);
    // Model the exact native boundary after the owner has selected a request.
    // A cancellation before ReadFile/WriteFile necessarily misses its key;
    // the real issue-result helper must join that intent after native issue.
    if (cancel_before_issue) {
      IREE_EXPECT_OK(iree_async_proactor_cancel(proactor_, operation));
    }
    BOOL accepted = GetParam() == IREE_ASYNC_OPERATION_TYPE_FILE_READ
                        ? ReadFile(handle, data_.data(), (DWORD)data_.size(),
                                   nullptr, &carrier->overlapped)
                        : WriteFile(handle, data_.data(), (DWORD)data_.size(),
                                    nullptr, &carrier->overlapped);
    const DWORD error = accepted ? ERROR_SUCCESS : GetLastError();
    EXPECT_TRUE(error == ERROR_SUCCESS || error == ERROR_IO_PENDING) << error;
    if (!cancel_before_issue) {
      IREE_EXPECT_OK(iree_async_proactor_cancel(proactor_, operation));
    }
    IREE_EXPECT_OK(
        iree_async_proactor_iocp_finish_file_issue(iocp(), carrier, error));
    EXPECT_EQ(completed.call_count, 0);
    EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 2);
    PollUntil(completed);
    EXPECT_EQ(completed.status_code, IREE_STATUS_CANCELLED);
    EXPECT_EQ(completed.call_count, 1);
    EXPECT_EQ(operation->next, nullptr);
    EXPECT_EQ(iree_atomic_ref_count_load(&file_->ref_count), 1);
    EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                               iree_memory_order_relaxed),
              0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    FileDirections, IocpFileAdmissionTest,
    ::testing::Values(IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                      IREE_ASYNC_OPERATION_TYPE_FILE_WRITE));

TEST_F(IocpProactorTest, MultishotRearmJoinsTerminalCancellationBorrow) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));
  std::unique_ptr<iree_async_socket_t, decltype(&iree_async_socket_release)>
      socket_owner(socket, iree_async_socket_release);
  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket, &address));
  IREE_ASSERT_OK(iree_async_socket_query_local_address(socket, &address));
  CompletionState connected;
  iree_async_socket_connect_operation_t connect;
  InitializeOperation(&connect, IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
                      &connected);
  connect.socket = socket;
  connect.address = address;
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect.base));
  while (!connected.call_count) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  ASSERT_EQ(connected.status_code, IREE_STATUS_OK);

  struct ReceiveState {
    // Number of complete datagrams delivered before cancellation.
    int more_count = 0;
    // Held across terminal delivery to model an outstanding native cancel.
    IoCancellationBorrow cancellation;
    // Terminal callback, distinct from the intermediate datagrams.
    CompletionState terminal;
  } received;
  uint8_t received_byte = 0;
  iree_async_span_t receive_span =
      iree_async_span_from_ptr(&received_byte, sizeof(received_byte));
  iree_async_socket_recv_operation_t receive;
  InitializeOperation(&receive, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV, nullptr);
  receive.base.flags = IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
  receive.base.user_data = &received;
  receive.base.completion_fn =
      +[](void* user_data, iree_async_operation_t* operation,
          iree_status_t status, iree_async_completion_flags_t flags) {
        auto* state = static_cast<ReceiveState*>(user_data);
        if (iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE)) {
          IREE_EXPECT_OK(status);
          if (++state->more_count == 2) {
            state->cancellation.reset(
                iree_async_proactor_iocp_acquire_io_cancel(operation));
          }
        } else {
          RecordCompletion(&state->terminal, operation, status, flags);
        }
      };
  receive.socket = socket;
  receive.buffers = iree_async_span_list_make(&receive_span, 1);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &receive.base));

  // The first packet rearms normally. Cancellation during the second MORE
  // callback cannot target that finished native issue, but must prevent a
  // third issue and join the terminal result before releasing the socket.
  for (uint8_t value : {1, 2}) {
    CompletionState sent;
    iree_async_socket_send_operation_t send;
    InitializeOperation(&send, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND, &sent);
    iree_async_span_t send_span =
        iree_async_span_from_ptr(&value, sizeof(value));
    send.socket = socket;
    send.buffers = iree_async_span_list_make(&send_span, 1);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send.base));
    while (!sent.call_count || received.more_count < value) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
    EXPECT_EQ(sent.status_code, IREE_STATUS_OK);
    EXPECT_EQ(received_byte, value);
  }
  EXPECT_NE(received.cancellation.get(), nullptr);
  EXPECT_EQ(received.terminal.call_count, 0);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 2);
  received.cancellation.reset();
  while (!received.terminal.call_count) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(received.more_count, 2);
  EXPECT_EQ(received.terminal.call_count, 1);
  EXPECT_EQ(received.terminal.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
}

TEST_F(IocpProactorTest, DirectPostFailureCompletesAcceptedSignal) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  ASSERT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 1);

  CompletionState completion;
  iree_async_notification_signal_operation_t signal_operation;
  InitializeOperation(&signal_operation,
                      IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,
                      &completion);
  signal_operation.notification = notification;
  signal_operation.wake_count = 1;

  uint32_t initial_epoch = iree_async_notification_query_epoch(notification);
  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &signal_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);

  EXPECT_NE(iree_async_notification_query_epoch(notification), initial_epoch);
  EXPECT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 2);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(signal_operation.base.next, nullptr);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 1);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);

  iree_async_notification_release(notification);
}

TEST_F(IocpProactorTest, DirectPostFailurePreservesRichStatus) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, /*initial_value=*/10,
      IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, iree_allocator_system(),
      &semaphore));
  iree_async_semaphore_t* semaphore_storage = semaphore;
  uint64_t non_monotonic_value = 10;

  CompletionState completion;
  iree_async_semaphore_signal_operation_t signal_operation;
  InitializeOperation(&signal_operation,
                      IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL, &completion);
  signal_operation.semaphores = &semaphore_storage;
  signal_operation.values = &non_monotonic_value;
  signal_operation.count = 1;

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &signal_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 10u);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(signal_operation.base.next, nullptr);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);

  iree_async_semaphore_release(semaphore);
}

TEST_F(IocpProactorTest, DirectPostFailureCompletesAcceptedSocketClose) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));
  iree_async_socket_retain(socket);

  CompletionState completion;
  iree_async_socket_close_operation_t close_operation;
  InitializeOperation(&close_operation, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                      &completion);
  close_operation.socket = socket;

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &close_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);

  EXPECT_EQ(socket->primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_EQ(socket->state, IREE_ASYNC_SOCKET_STATE_CLOSED);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 2);
  EXPECT_EQ(completion.call_count, 0);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);

  iree_async_socket_release(socket);
}

TEST_F(IocpProactorTest, DirectPostFailureCompletesPlatformSetupFailure) {
  CompletionState completion;
  iree_async_file_open_operation_t open_operation;
  InitializeOperation(&open_operation, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN,
                      &completion);
  open_operation.path = "iree-iocp-invalid-<>.bin";
  open_operation.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &open_operation.base);
  // File calls run on the poll owner. Keep posting disabled through native
  // admission so the setup failure exercises the carrier fallback queue.
  iree_host_size_t completed_count = 0;
  iree_status_t issue_status =
      iree_async_proactor_iocp_drain_file_submissions(iocp(), &completed_count);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);
  IREE_EXPECT_OK(issue_status);
  EXPECT_EQ(completed_count, 0);

  EXPECT_EQ(open_operation.opened_file, nullptr);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_NE(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
}

TEST_F(IocpProactorTest, MalformedTailDoesNotConsumeSocketClose) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));
  const SOCKET native_socket = (SOCKET)socket->primitive.value.win32_handle;

  CompletionState completion;
  iree_async_socket_close_operation_t close_operation;
  InitializeOperation(&close_operation, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                      &completion);
  close_operation.socket = socket;

  iree_async_nop_operation_t malformed_operation;
  InitializeOperation(&malformed_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                      /*completion=*/nullptr);
  iree_async_operation_t* operations[] = {&close_operation.base,
                                          &malformed_operation.base};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_submit(proactor_,
                                 iree_async_operation_list_make(
                                     operations, IREE_ARRAYSIZE(operations))));

  int socket_type = 0;
  int socket_type_length = sizeof(socket_type);
  EXPECT_EQ(getsockopt(native_socket, SOL_SOCKET, SO_TYPE, (char*)&socket_type,
                       &socket_type_length),
            0);
  EXPECT_EQ(socket->primitive.value.win32_handle, (uintptr_t)native_socket);
  EXPECT_EQ(socket->state, IREE_ASYNC_SOCKET_STATE_CREATED);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  EXPECT_EQ(completion.call_count, 0);

  iree_async_socket_retain(socket);
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &close_operation.base));
  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(socket->primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  iree_async_socket_release(socket);
}

TEST(IocpProactorSubmitTest, CarrierAllocationFailureRollsBackCloseAndMessage) {
  ControlledAllocator allocator;
  iree_async_proactor_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      iree_async_proactor_options_default(), allocator.value(), &source));
  iree_async_proactor_t* target = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      iree_async_proactor_options_default(), iree_allocator_system(), &target));

  MessageState message_state;
  iree_async_proactor_set_message_callback(
      target, {.fn = RecordMessage, .user_data = &message_state});

  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(source, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));
  const SOCKET native_socket = (SOCKET)socket->primitive.value.win32_handle;

  CompletionState completion;
  iree_async_socket_close_operation_t close_operation;
  InitializeOperation(&close_operation, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                      &completion);
  close_operation.socket = socket;

  iree_async_message_operation_t message_operation;
  InitializeOperation(&message_operation, IREE_ASYNC_OPERATION_TYPE_MESSAGE,
                      &completion);
  message_operation.target = target;
  message_operation.message_data = 0xC0FFEE;
  message_operation.message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;

  iree_async_operation_t* operations[] = {&close_operation.base,
                                          &message_operation.base};
  allocator.allocations_before_failure = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_submit(
          source, iree_async_operation_list_make(operations,
                                                 IREE_ARRAYSIZE(operations))));

  int socket_type = 0;
  int socket_type_length = sizeof(socket_type);
  EXPECT_EQ(getsockopt(native_socket, SOL_SOCKET, SO_TYPE, (char*)&socket_type,
                       &socket_type_length),
            0);
  EXPECT_EQ(socket->primitive.value.win32_handle, (uintptr_t)native_socket);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  EXPECT_EQ(message_operation.platform.software.reserved_entry, nullptr);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(message_state.call_count, 0);
  EXPECT_EQ(
      iree_atomic_load(
          &iree_async_proactor_iocp_cast(source)->outstanding_carrier_count,
          iree_memory_order_relaxed),
      0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(source, iree_immediate_timeout(), nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(target, iree_immediate_timeout(), nullptr));

  allocator.allocations_before_failure = -1;
  iree_async_socket_retain(socket);
  IREE_ASSERT_OK(iree_async_proactor_submit(
      source,
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations))));
  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(source, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 2);
  EXPECT_EQ(completion.call_count, 2);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(socket->primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);

  IREE_ASSERT_OK(iree_async_proactor_poll(target, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(message_state.call_count, 1);
  EXPECT_EQ(message_state.value, 0xC0FFEEu);

  iree_async_socket_release(socket);
  iree_async_proactor_release(target);
  iree_async_proactor_release(source);
}

TEST(IocpFileImportTest, AllocationFailureLeavesCompletionPortUnassociated) {
  ControlledAllocator allocator;
  iree_async_proactor_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      iree_async_proactor_options_default(), allocator.value(), &source));
  std::unique_ptr<iree_async_proactor_t, decltype(&iree_async_proactor_release)>
      source_owner(source, iree_async_proactor_release);
  iree_async_proactor_t* target = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      iree_async_proactor_options_default(), iree_allocator_system(), &target));
  std::unique_ptr<iree_async_proactor_t, decltype(&iree_async_proactor_release)>
      target_owner(target, iree_async_proactor_release);

  iree::testing::TempFilePath path("iree_async_import_rollback");
  HANDLE handle =
      CreateFileA(path.path().c_str(), GENERIC_READ | GENERIC_WRITE,
                  FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
                  FILE_FLAG_OVERLAPPED | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  std::unique_ptr<void, decltype(&CloseHandle)> handle_owner(handle,
                                                             CloseHandle);
  iree_async_primitive_t primitive = iree_async_primitive_from_win32_handle(
      reinterpret_cast<uintptr_t>(handle));

  allocator.allocations_before_failure = 0;
  iree_async_file_t* file = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_file_import(source, primitive, IREE_ASYNC_FILE_FLAG_NONE,
                             &file));
  EXPECT_EQ(file, nullptr);
  allocator.allocations_before_failure = -1;

  // A successful import into a different port proves that failed construction
  // did not irreversibly associate the caller's native handle with |source|.
  IREE_ASSERT_OK(iree_async_file_import(target, primitive,
                                        IREE_ASYNC_FILE_FLAG_NONE, &file));
  handle_owner.release();
  std::unique_ptr<iree_async_file_t, decltype(&iree_async_file_release)>
      file_owner(file, iree_async_file_release);
  CompletionState completion;
  uint8_t data = 0;
  iree_async_file_read_operation_t read;
  InitializeOperation(&read, IREE_ASYNC_OPERATION_TYPE_FILE_READ, &completion);
  read.file = file;
  read.offset = 0;
  read.buffer = iree_async_span_from_ptr(&data, 1);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(target, &read.base));
  while (completion.call_count == 0) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(target, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(read.bytes_read, 0);
}

TEST_F(IocpProactorTest, FailedWakePersistsUntilPoll) {
  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_async_proactor_wake(proactor_);

  EXPECT_EQ(iree_atomic_load(&iocp()->completion_port.fallback_wake_pending,
                             iree_memory_order_acquire),
            1);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));
  EXPECT_EQ(iree_atomic_load(&iocp()->completion_port.fallback_wake_pending,
                             iree_memory_order_acquire),
            0);
}

TEST_F(IocpProactorTest, FallbackApcInterruptsBlockedPoll) {
  std::atomic<iree_status_code_t> poll_status{IREE_STATUS_UNKNOWN};
  std::thread poll_thread([&]() {
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_infinite_timeout(), /*out_completed_count=*/nullptr);
    poll_status.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
  });

  // Waiting for the durable owner handle ensures wake() must use the APC path
  // instead of relying on a pre-poll pending flag.
  while (iree_atomic_load(&iocp()->completion_port.poll_thread_handle,
                          iree_memory_order_acquire) == 0) {
    std::this_thread::yield();
  }
  iree_async_iocp_completion_port_request_fallback_wake(
      &iocp()->completion_port);
  poll_thread.join();

  EXPECT_EQ(poll_status.load(std::memory_order_acquire), IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp()->completion_port.fallback_wake_pending,
                             iree_memory_order_acquire),
            0);
}

TEST(IocpEventSourceTest, RequiresWaitCompletionPackets) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  iree_async_event_source_callback_t callback = {
      +[](void*, iree_async_event_source_t*, iree_async_poll_events_t) {},
      nullptr,
  };
  iree_async_event_source_t* source = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_async_proactor_register_event_source(
          proactor, event->native.wait_primitive, callback, &source));
  EXPECT_EQ(source, nullptr);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
}

TEST(IocpLegacyEventWaitTest, FailedCallbackPostDispatchesFromPoll) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));
  iree_async_proactor_iocp_t* iocp = iree_async_proactor_iocp_cast(proactor);

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  CompletionState completion;
  iree_async_event_wait_operation_t wait_operation;
  std::memset(&wait_operation, 0, sizeof(wait_operation));
  wait_operation.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_operation.base.completion_fn = RecordCompletion;
  wait_operation.base.user_data = &completion;
  wait_operation.event = event;

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor, &wait_operation.base));
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));
  ASSERT_EQ(completion.call_count, 0);
  ASSERT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp, &preserved_port));
  iree_async_event_set(event);
  iree_async_iocp_carrier_t* carrier = iocp->active_carriers;
  ASSERT_NE(carrier, nullptr);
  WaitForFallbackCompletionReady(carrier);
  RestoreCompletionPosting(iocp, preserved_port);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                          &completed_count));

  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
  EXPECT_EQ(iree_atomic_ref_count_load(&event->ref_count), 1);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
}

TEST(IocpLegacyEventWaitTest, SuccessfulCallbackPostReleasesRegistration) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));
  iree_async_proactor_iocp_t* iocp = iree_async_proactor_iocp_cast(proactor);

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  CompletionState completion;
  iree_async_event_wait_operation_t wait_operation;
  std::memset(&wait_operation, 0, sizeof(wait_operation));
  wait_operation.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_operation.base.completion_fn = RecordCompletion;
  wait_operation.base.user_data = &completion;
  wait_operation.event = event;

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor, &wait_operation.base));
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));
  ASSERT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  iree_async_event_set(event);
  iree_host_size_t completed_count = 0;
  while (completion.call_count == 0) {
    iree_host_size_t poll_completed_count = 0;
    IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                            &poll_completed_count));
    completed_count += poll_completed_count;
  }

  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
  EXPECT_EQ(iree_atomic_ref_count_load(&event->ref_count), 1);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
}

TEST(IocpLegacyEventWaitTest, CancelledFallbackDispatchesExactlyOnce) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));
  iree_async_proactor_iocp_t* iocp = iree_async_proactor_iocp_cast(proactor);

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  CompletionState completion;
  iree_async_event_wait_operation_t wait_operation;
  std::memset(&wait_operation, 0, sizeof(wait_operation));
  wait_operation.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_operation.base.completion_fn = RecordCompletion;
  wait_operation.base.user_data = &completion;
  wait_operation.event = event;

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor, &wait_operation.base));
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp, &preserved_port));
  iree_async_event_set(event);
  iree_async_iocp_carrier_t* carrier = iocp->active_carriers;
  ASSERT_NE(carrier, nullptr);
  WaitForFallbackCompletionReady(carrier);
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor, &wait_operation.base));
  RestoreCompletionPosting(iocp, preserved_port);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                          &completed_count));

  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
  EXPECT_EQ(iree_atomic_ref_count_load(&event->ref_count), 1);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
}

}  // namespace
