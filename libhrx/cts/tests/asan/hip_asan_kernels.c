// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#define HRX_AMDGPU_ATTRIBUTE_KERNEL \
  [[clang::amdgpu_kernel, gnu::visibility("protected"), gnu::used]]

typedef __UINT8_TYPE__ uint8_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT64_TYPE__ uint64_t;

extern void __asan_load1(uint64_t address);

HRX_AMDGPU_ATTRIBUTE_KERNEL void hrx_asan_plain_in_bounds(uint8_t* data,
                                                          uint32_t* output) {
  output[0] = (uint32_t)data[0] + 1u;
}

HRX_AMDGPU_ATTRIBUTE_KERNEL void hrx_asan_in_bounds(uint8_t* data,
                                                    uint32_t* output) {
  __asan_load1((uint64_t)data);
  output[0] = (uint32_t)data[0] + 1u;
}

HRX_AMDGPU_ATTRIBUTE_KERNEL void hrx_asan_left_redzone(uint64_t address,
                                                       uint32_t* output) {
  __asan_load1(address - 1u);
  output[0] = 0xA5A5A5A5u;
}

HRX_AMDGPU_ATTRIBUTE_KERNEL void hrx_asan_check_only(uint64_t address,
                                                     uint32_t* output) {
  __asan_load1(address);
  output[0] = 0x5A5A5A5Au;
}
