// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU sanitizer executable load validation coverage.

#include <string>
#include <string_view>

#include "iree/hal/cts/sanitizer/sanitizer_test_util.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;
using ::testing::HasSubstr;

typedef struct SanitizerExecutableLoadExpectation {
  std::string_view executable_file;
  std::string_view global_name;
  std::string_view capability_name;
} SanitizerExecutableLoadExpectation;

static SanitizerExecutableLoadExpectation GetSanitizerExecutableLoadExpectation(
    const BackendInfo& backend) {
  const std::string_view backend_name(backend.name);
  if (backend_name.find("asan_executable") != std::string_view::npos) {
    return {
        /*.executable_file=*/"asan_executable_test.bin",
        /*.global_name=*/IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME,
        /*.capability_name=*/"AMDGPU ASAN shadow memory",
    };
  }
  if (backend_name.find("tsan_executable") != std::string_view::npos) {
    return {
        /*.executable_file=*/"tsan_executable_test.bin",
        /*.global_name=*/IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME,
        /*.capability_name=*/"AMDGPU TSAN shadow memory",
    };
  }
  return {
      /*.executable_file=*/"feedback_executable_test.bin",
      /*.global_name=*/IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME,
      /*.capability_name=*/"the AMDGPU feedback channel",
  };
}

class SanitizerExecutableLoadTest
    : public ::testing::TestWithParam<BackendInfo> {
 protected:
  void SetUp() override {
    iree_status_t status = device_.Initialize(GetParam(), "none");
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' unavailable on this system";
    }
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device(), iree_make_cstring_view("default"), executable_cache_.out()));
  }

  iree_hal_device_t* device() const { return device_.device(); }

  SanitizerCachedBackendDevice device_;
  Ref<iree_hal_executable_cache_t> executable_cache_;
};

TEST_P(SanitizerExecutableLoadTest, RejectsDisabledRuntimeFeatureGlobal) {
  const SanitizerExecutableLoadExpectation expectation =
      GetSanitizerExecutableLoadExpectation(GetParam());

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format =
      iree_make_cstring_view(GetParam().executable_format);
  executable_params.executable_data = GetParam().executable_data(
      iree_make_string_view(expectation.executable_file.data(),
                            expectation.executable_file.size()));

  Ref<iree_hal_executable_t> executable;
  iree::Status status(iree_hal_executable_cache_prepare_executable(
      executable_cache_, &executable_params, executable.out()));
  if (status.code() == StatusCode::kIncompatible) {
    GTEST_SKIP() << "Executable format '" << GetParam().executable_format
                 << "' is incompatible with CTS backend/device '"
                 << GetParam().name << "'";
  }

  EXPECT_THAT(status, StatusIs(StatusCode::kFailedPrecondition));
  const std::string status_text = status.ToString();
  EXPECT_THAT(status_text, HasSubstr(expectation.global_name));
  EXPECT_THAT(status_text, HasSubstr(expectation.capability_name));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(SanitizerExecutableLoadTest);

}  // namespace iree::hal::cts
