// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_memory.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "gtest/gtest.h"

namespace {

// Inject only native outcomes; backing ownership and rollback use production
// code and real file descriptors/mappings.
struct NativeState {
  // Failure of the initial view map, or zero for a real mapping.
  int map_error = 0;
  // Failure before unmapping the persistent view, or zero for real unmapping.
  int unmap_error = 0;
  // Error reported after close has consumed the descriptor.
  int close_error = 0;
  // Real descriptor acquired by the production backing preparation.
  int descriptor = -1;
  // Real persistent mapping, if the production map succeeded.
  void* pointer = nullptr;
  // Number of source release attempts through the production implementation.
  uint32_t close_count = 0;
};

thread_local NativeState* native_state = nullptr;

}  // namespace

extern "C" void* __real_mmap(void*, size_t, int, int, int, off_t);
extern "C" int __real_munmap(void*, size_t);
extern "C" int __real_close(int);

extern "C" void* __wrap_mmap(void* address, size_t length, int protection,
                             int flags, int descriptor, off_t offset) {
  if (native_state == nullptr || (flags & MAP_SHARED) == 0 || descriptor < 0) {
    return __real_mmap(address, length, protection, flags, descriptor, offset);
  }
  native_state->descriptor = descriptor;
  if (native_state->map_error != 0) {
    errno = native_state->map_error;
    return MAP_FAILED;
  }
  void* pointer =
      __real_mmap(address, length, protection, flags, descriptor, offset);
  if (pointer != MAP_FAILED) {
    native_state->pointer = pointer;
  }
  return pointer;
}

extern "C" int __wrap_munmap(void* address, size_t length) {
  if (native_state != nullptr && address == native_state->pointer &&
      native_state->unmap_error != 0) {
    errno = native_state->unmap_error;
    return -1;
  }
  const int result = __real_munmap(address, length);
  if (native_state != nullptr && address == native_state->pointer &&
      result == 0) {
    native_state->pointer = nullptr;
  }
  return result;
}

extern "C" int __wrap_close(int descriptor) {
  if (native_state == nullptr || descriptor != native_state->descriptor) {
    return __real_close(descriptor);
  }
  ++native_state->close_count;
  const int result = __real_close(descriptor);
  if (result != 0) {
    return result;
  }
  native_state->descriptor = -1;
  if (native_state->close_error != 0) {
    errno = native_state->close_error;
    return -1;
  }
  return 0;
}

namespace {

class HostMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override { native_state = &native_; }

  void TearDown() override {
    // Test-owned native residue is independent of consumed library metadata.
    // Retire it directly, without retrying the production operation.
    native_state = nullptr;
    if (native_.pointer != nullptr) {
      EXPECT_EQ(__real_munmap(native_.pointer, 4096), 0);
    }
    if (native_.descriptor >= 0) {
      EXPECT_EQ(__real_close(native_.descriptor), 0);
    }
  }

  // Native dependency state retained through final assertions.
  NativeState native_;
};

TEST_F(HostMemoryTest, FailedPreparationRetainsItsSourceForOwnerCleanup) {
  native_.map_error = ENOMEM;
  amdf_platform_host_backing_t backing = {};
  EXPECT_EQ(amdf_platform_host_backing_prepare(4096, &backing),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  ASSERT_GE(native_.descriptor, 0);
  EXPECT_EQ(backing.handle, static_cast<uintptr_t>(native_.descriptor));
  EXPECT_EQ(backing.pointer, nullptr);
  EXPECT_EQ(fcntl(native_.descriptor, F_GETFD), FD_CLOEXEC);
  EXPECT_EQ(native_.close_count, 0u);
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096), AMDF_STATUS_OK);
  EXPECT_EQ(backing.handle, UINTPTR_MAX);
  EXPECT_EQ(native_.close_count, 1u);
}

TEST_F(HostMemoryTest, FailedViewRemovalPreservesBackingAndReleaseProgress) {
  amdf_platform_host_backing_t backing = {};
  ASSERT_EQ(amdf_platform_host_backing_prepare(4096, &backing), AMDF_STATUS_OK);
  auto* bytes = static_cast<uint8_t*>(backing.pointer);
  bytes[17] = 0xA5;
  native_.unmap_error = EIO;
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(backing.pointer, bytes);
  EXPECT_EQ(bytes[17], 0xA5);
  EXPECT_EQ(backing.handle, static_cast<uintptr_t>(native_.descriptor));
  EXPECT_EQ(fcntl(native_.descriptor, F_GETFD), FD_CLOEXEC);
  EXPECT_EQ(native_.close_count, 0u);
}

TEST_F(HostMemoryTest, CloseErrorPreservesTheFactThatTheSourceWasConsumed) {
  amdf_platform_host_backing_t backing = {};
  ASSERT_EQ(amdf_platform_host_backing_prepare(4096, &backing), AMDF_STATUS_OK);
  const int descriptor = native_.descriptor;
  native_.close_error = EIO;
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(backing.pointer, nullptr);
  EXPECT_EQ(backing.handle, UINTPTR_MAX);
  EXPECT_EQ(fcntl(descriptor, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
  EXPECT_EQ(native_.close_count, 1u);
}

}  // namespace
