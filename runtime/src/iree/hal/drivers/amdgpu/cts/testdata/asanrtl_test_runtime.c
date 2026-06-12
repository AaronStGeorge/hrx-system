// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/device/support/asan.h"

#if defined(__clang__)
// The replacement ASAN runtime is linked as bitcode into instrumented programs.
// Its ABI globals and helper functions must keep their exact layout and must
// not recursively call back into the sanitizer hooks they implement.
#pragma clang attribute push(                           \
    __attribute__((disable_sanitizer_instrumentation)), \
    apply_to = any(function, variable(is_global)))
#endif  // defined(__clang__)

__attribute__((no_sanitize("address")))
[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_asan_config_t iree_asan_config = {0};

__attribute__((no_sanitize("address")))
[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_feedback_config_t iree_feedback_config = {0};

static uint64_t iree_asan_shadow_address(uint64_t address) {
  uint64_t shadow_address =
      iree_asan_config.shadow_base +
      ((address - iree_asan_config.application_window_base) >>
       iree_asan_config.shadow_scale_shift);
  return shadow_address;
}

static void iree_asan_report(iree_hal_amdgpu_asan_access_kind_t access_kind,
                             uint64_t address, uint64_t access_size) {
  uint64_t shadow_address = iree_asan_shadow_address(address);
  (void)iree_hal_amdgpu_asan_report_access(&iree_feedback_config, access_kind,
                                           address, access_size, 0,
                                           shadow_address, 0);
}

static void iree_asan_load(uint64_t address, uint64_t access_size) {
  iree_asan_report(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ, address, access_size);
}

static void iree_asan_store(uint64_t address, uint64_t access_size) {
  iree_asan_report(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE, address,
                   access_size);
}

void __asan_load1(uint64_t address) { iree_asan_load(address, 1); }
void __asan_load2(uint64_t address) { iree_asan_load(address, 2); }
void __asan_load4(uint64_t address) { iree_asan_load(address, 4); }
void __asan_load8(uint64_t address) { iree_asan_load(address, 8); }
void __asan_load16(uint64_t address) { iree_asan_load(address, 16); }
void __asan_loadN(uint64_t address, uint64_t size) {
  iree_asan_load(address, size);
}

void __asan_load1_noabort(uint64_t address) { __asan_load1(address); }
void __asan_load2_noabort(uint64_t address) { __asan_load2(address); }
void __asan_load4_noabort(uint64_t address) { __asan_load4(address); }
void __asan_load8_noabort(uint64_t address) { __asan_load8(address); }
void __asan_load16_noabort(uint64_t address) { __asan_load16(address); }
void __asan_loadN_noabort(uint64_t address, uint64_t size) {
  __asan_loadN(address, size);
}

void __asan_store1(uint64_t address) { iree_asan_store(address, 1); }
void __asan_store2(uint64_t address) { iree_asan_store(address, 2); }
void __asan_store4(uint64_t address) { iree_asan_store(address, 4); }
void __asan_store8(uint64_t address) { iree_asan_store(address, 8); }
void __asan_store16(uint64_t address) { iree_asan_store(address, 16); }
void __asan_storeN(uint64_t address, uint64_t size) {
  iree_asan_store(address, size);
}

void __asan_store1_noabort(uint64_t address) { __asan_store1(address); }
void __asan_store2_noabort(uint64_t address) { __asan_store2(address); }
void __asan_store4_noabort(uint64_t address) { __asan_store4(address); }
void __asan_store8_noabort(uint64_t address) { __asan_store8(address); }
void __asan_store16_noabort(uint64_t address) { __asan_store16(address); }
void __asan_storeN_noabort(uint64_t address, uint64_t size) {
  __asan_storeN(address, size);
}

void __asan_report_load1(uint64_t address) { __asan_load1(address); }
void __asan_report_load2(uint64_t address) { __asan_load2(address); }
void __asan_report_load4(uint64_t address) { __asan_load4(address); }
void __asan_report_load8(uint64_t address) { __asan_load8(address); }
void __asan_report_load16(uint64_t address) { __asan_load16(address); }
void __asan_report_load_n(uint64_t address, uint64_t size) {
  __asan_loadN(address, size);
}

void __asan_report_load1_noabort(uint64_t address) { __asan_load1(address); }
void __asan_report_load2_noabort(uint64_t address) { __asan_load2(address); }
void __asan_report_load4_noabort(uint64_t address) { __asan_load4(address); }
void __asan_report_load8_noabort(uint64_t address) { __asan_load8(address); }
void __asan_report_load16_noabort(uint64_t address) { __asan_load16(address); }
void __asan_report_load_n_noabort(uint64_t address, uint64_t size) {
  __asan_loadN(address, size);
}

void __asan_report_store1(uint64_t address) { __asan_store1(address); }
void __asan_report_store2(uint64_t address) { __asan_store2(address); }
void __asan_report_store4(uint64_t address) { __asan_store4(address); }
void __asan_report_store8(uint64_t address) { __asan_store8(address); }
void __asan_report_store16(uint64_t address) { __asan_store16(address); }
void __asan_report_store_n(uint64_t address, uint64_t size) {
  __asan_storeN(address, size);
}

void __asan_report_store1_noabort(uint64_t address) { __asan_store1(address); }
void __asan_report_store2_noabort(uint64_t address) { __asan_store2(address); }
void __asan_report_store4_noabort(uint64_t address) { __asan_store4(address); }
void __asan_report_store8_noabort(uint64_t address) { __asan_store8(address); }
void __asan_report_store16_noabort(uint64_t address) {
  __asan_store16(address);
}
void __asan_report_store_n_noabort(uint64_t address, uint64_t size) {
  __asan_storeN(address, size);
}

uint64_t __asan_region_is_poisoned(uint64_t address, uint64_t size) {
  (void)address;
  (void)size;
  return 0;
}

void __asan_poison_region(uint64_t address, uint64_t size) {
  (void)address;
  (void)size;
}

void* __asan_memcpy(void* target, const void* source, uint64_t size) {
  __asan_loadN((uint64_t)source, size);
  __asan_storeN((uint64_t)target, size);
  uint8_t* target_bytes = (uint8_t*)target;
  const uint8_t* source_bytes = (const uint8_t*)source;
  for (uint64_t i = 0; i < size; ++i) {
    target_bytes[i] = source_bytes[i];
  }
  return target;
}

void* __asan_memmove(void* target, const void* source, uint64_t size) {
  __asan_loadN((uint64_t)source, size);
  __asan_storeN((uint64_t)target, size);
  uint8_t* target_bytes = (uint8_t*)target;
  const uint8_t* source_bytes = (const uint8_t*)source;
  if (target_bytes <= source_bytes) {
    for (uint64_t i = 0; i < size; ++i) {
      target_bytes[i] = source_bytes[i];
    }
  } else {
    for (uint64_t i = size; i > 0; --i) {
      target_bytes[i - 1] = source_bytes[i - 1];
    }
  }
  return target;
}

void* __asan_memset(void* target, int value, uint64_t size) {
  __asan_storeN((uint64_t)target, size);
  uint8_t* target_bytes = (uint8_t*)target;
  for (uint64_t i = 0; i < size; ++i) {
    target_bytes[i] = (uint8_t)value;
  }
  return target;
}

uint64_t __asan_load_cxx_array_cookie(uint64_t address) {
  __asan_load8(address);
  return *(uint64_t*)address;
}

void __asan_poison_cxx_array_cookie(uint64_t address) {
  __asan_store8(address);
}

uint64_t __asan_malloc_impl(uint64_t size, uint64_t alignment) {
  (void)size;
  (void)alignment;
  return 0;
}

void __asan_free_impl(uint64_t address, uint64_t size) {
  (void)address;
  (void)size;
}

void __asan_register_globals(uint64_t globals, uint64_t size) {
  (void)globals;
  (void)size;
}

void __asan_unregister_globals(uint64_t globals, uint64_t size) {
  (void)globals;
  (void)size;
}

void __asan_register_elf_globals(uint64_t flag, uint64_t start, uint64_t stop) {
  (void)flag;
  (void)start;
  (void)stop;
}

void __asan_unregister_elf_globals(uint64_t flag, uint64_t start,
                                   uint64_t stop) {
  (void)flag;
  (void)start;
  (void)stop;
}

void __asan_register_image_globals(uint64_t image_id) { (void)image_id; }
void __asan_unregister_image_globals(uint64_t image_id) { (void)image_id; }
void __asan_after_dynamic_init(void) {}
void __asan_before_dynamic_init(uint64_t module_name) { (void)module_name; }
void __asan_handle_no_return(void) {}
void __asan_init(void) {}
void __asan_version_mismatch_check_v8(void) {}

#if defined(__clang__)
#pragma clang attribute pop
#endif  // defined(__clang__)
