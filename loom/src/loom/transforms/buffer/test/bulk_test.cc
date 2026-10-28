// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"

// Linked from the same source compiled for each native width policy. Guard
// pages observe machine-code reads as well as writes, including code that is
// outside the host compiler's sanitizer instrumentation.
extern "C" void bulk_copy(const void* source, uint64_t source_offset,
                          void* target, uint64_t target_offset,
                          uint64_t length);

namespace {

class GuardedPage {
 public:
  GuardedPage() {
    mapping_ = mmap(nullptr, 3 * page_size_, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_ != MAP_FAILED) {
      ready_ = mprotect(data(), page_size_, PROT_READ | PROT_WRITE) == 0;
    }
  }
  ~GuardedPage() {
    if (mapping_ != MAP_FAILED) {
      munmap(mapping_, 3 * page_size_);
    }
  }
  bool ready() const { return ready_; }
  uint8_t* data() const { return static_cast<uint8_t*>(mapping_) + page_size_; }
  size_t size() const { return page_size_; }

 private:
  // Host page size used by both accessible payload and inaccessible guards.
  size_t page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  // Owned mapping: one accessible page between two inaccessible pages.
  void* mapping_ = MAP_FAILED;
  // Whether allocation and protection setup both succeeded.
  bool ready_ = false;
};

class BulkMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if LOOM_TEST_VECTOR_BITS == 512
    if (!__builtin_cpu_supports("avx512f") ||
        !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") ||
        !__builtin_cpu_supports("avx512vl")) {
      GTEST_SKIP() << "AVX512 execution requires supporting hardware";
    }
#elif LOOM_TEST_VECTOR_BITS == 256
    if (!__builtin_cpu_supports("avx2")) {
      GTEST_SKIP() << "AVX2 execution requires supporting hardware";
    }
#endif
  }
};

TEST_F(BulkMemoryTest, CopyExactRange) {
  alignas(64) std::array<uint8_t, 256> source;
  alignas(64) std::array<uint8_t, 256> target;
  std::array<uint8_t, 256> expected;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = (i * 29 + 17) & 255;
  }
  for (size_t length = 0; length <= 129; ++length) {
    for (size_t source_offset = 0; source_offset < 64; ++source_offset) {
      const size_t target_offset = (3 * source_offset + length + 7) % 64;
      target.fill(0xA5);
      expected.fill(0xA5);
      std::memcpy(expected.data() + target_offset,
                  source.data() + source_offset, length);
      bulk_copy(source.data(), source_offset, target.data(), target_offset,
                length);
      ASSERT_EQ(target, expected)
          << "length=" << length << "source offset=" << source_offset
          << "target offset=" << target_offset;
    }
  }
}

TEST_F(BulkMemoryTest, CopyDoesNotAccessOutsideRange) {
  GuardedPage source;
  GuardedPage target;
  ASSERT_TRUE(source.ready());
  ASSERT_TRUE(target.ready());
  for (size_t i = 0; i < source.size(); ++i) {
    source.data()[i] = i & 255;
  }
  bulk_copy(nullptr, 0, nullptr, 0, 0);
  for (size_t length = 0; length <= 129; ++length) {
    const size_t source_offset = source.size() - length;
    const size_t target_offset = target.size() - length;
    std::memset(target.data(), 0xA5, target.size());
    bulk_copy(source.data(), source_offset, target.data(), target_offset,
              length);
    EXPECT_EQ(std::memcmp(source.data() + source_offset,
                          target.data() + target_offset, length),
              0);
    EXPECT_EQ(target.data()[target_offset - 1], 0xA5);
  }
}

}  // namespace
