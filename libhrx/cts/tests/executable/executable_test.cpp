// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cctype>
#include <string>

#include "build_tools/amdgpu/target_map.h"
#include "hrx_test_fixture.hpp"
#include "libhrx/cts/amdgpu_executable_noop_kernels.h"

namespace {

std::string gpu_architecture(hrx_device_t device) {
  std::array<char, 64> arch = {};
  REQUIRE_OK(hrx().device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       arch.data(), arch.size()));
  std::string value(arch.data());
  for (char c : value) {
    REQUIRE((std::isalnum(static_cast<unsigned char>(c)) || c == '_'));
  }
  return value;
}

const iree_file_toc_t* find_noop_hsaco_for_target(const std::string& target) {
  char fragment[64] = {};
  if (!iree_amdgpu_target_label_fragment(target.c_str(), fragment,
                                         sizeof(fragment))) {
    return nullptr;
  }
  const std::string filename =
      std::string("hrx_cts_noop_kernel_") + fragment + ".so";
  const iree_file_toc_t* toc = hrx_cts_amdgpu_executable_noop_kernels_create();
  for (size_t i = 0; i < hrx_cts_amdgpu_executable_noop_kernels_size(); ++i) {
    if (filename == toc[i].name) {
      return &toc[i];
    }
  }
  return nullptr;
}

const iree_file_toc_t* find_noop_hsaco(const std::string& arch) {
  if (const iree_file_toc_t* file = find_noop_hsaco_for_target(arch)) {
    return file;
  }
  const char* code_object_target =
      iree_amdgpu_code_object_target_for_exact(arch.c_str());
  if (code_object_target && arch != code_object_target) {
    return find_noop_hsaco_for_target(code_object_target);
  }
  return nullptr;
}

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "executable_load_lookup_dispatch_noop") {
  if (!is_gpu()) {
    SUCCEED("Native executable dispatch is GPU-only");
    return;
  }

  std::string arch = gpu_architecture(device_);
  if (arch.empty() || arch == "unknown") {
    SUCCEED("Skipping native executable CTS: unknown GPU architecture");
    return;
  }

  const iree_file_toc_t* hsaco = find_noop_hsaco(arch);
  if (!hsaco) {
    SUCCEED("Skipping native executable CTS: no build-time HSACO test asset");
    return;
  }

  hrx_executable_t executable = nullptr;
  REQUIRE_OK(hrx().executable_load_data(device_, hsaco->data, hsaco->size,
                                        nullptr, &executable));
  REQUIRE(executable != nullptr);

  hrx().executable_retain(executable);
  hrx().executable_release(executable);

  size_t export_count = 0;
  REQUIRE_OK(hrx().executable_export_count(executable, &export_count));
  REQUIRE(export_count == 1);

  uint32_t ordinal = UINT32_MAX;
  REQUIRE_OK(
      hrx().executable_lookup_export_by_name(executable, "hrx_noop", &ordinal));
  REQUIRE(ordinal == 0);

  hrx_executable_export_info_t info = {};
  REQUIRE_OK(hrx().executable_export_info(executable, ordinal, &info));
  REQUIRE(info.name != nullptr);
  REQUIRE(std::string(info.name) == "hrx_noop");
  REQUIRE(info.constant_count == 0);
  REQUIRE(info.binding_count == 0);
  REQUIRE(info.workgroup_size[0] >= 1);
  REQUIRE(info.workgroup_size[1] >= 1);
  REQUIRE(info.workgroup_size[2] >= 1);

  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(device_, 0, &stream));
  REQUIRE(stream != nullptr);

  hrx_dispatch_config_t config = {
      /* .workgroup_count = */ {1, 1, 1},
      /* .workgroup_size = */
      {
          info.workgroup_size[0],
          info.workgroup_size[1],
          info.workgroup_size[2],
      },
      /* .subgroup_size = */ 0,
  };
  REQUIRE_OK(hrx().stream_dispatch(stream, executable, ordinal, &config,
                                   /*constants=*/nullptr, /*constants_size=*/0,
                                   /*bindings=*/nullptr, /*binding_count=*/0,
                                   HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  hrx().stream_release(stream);
  hrx().executable_release(executable);
}
