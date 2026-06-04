// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_RCCL_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_DRIVERS_HIP_RCCL_DYNAMIC_SYMBOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"

#ifndef IREE_HAL_HIP_ENABLE_RCCL
#define IREE_HAL_HIP_ENABLE_RCCL 0
#endif  // IREE_HAL_HIP_ENABLE_RCCL

#if IREE_HAL_HIP_ENABLE_RCCL
#include "iree/hal/drivers/hip/rccl_headers.h"
#endif  // IREE_HAL_HIP_ENABLE_RCCL

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if IREE_HAL_HIP_ENABLE_RCCL

// iree_dynamic_library_t allows dynamically loading a subset of the NCCL API.
// We load all the symbols in `nccl_dynamic_symbol_table.h` and fail if any of
// the symbol is not available. The functions signatures are matching the
// declarations in `nccl.h`.

// NCCL API dynamic symbols.
typedef struct iree_hal_hip_nccl_dynamic_symbols_t {
  // The dynamic library handle.
  iree_dynamic_library_t* dylib;

  // Concrete NCCL symbols defined by including the `dynamic_symbol_tables.h`.
#define IREE_NCCL_PFN_DECL(ncclSymbolName, ...) \
  ncclResult_t (*ncclSymbolName)(__VA_ARGS__);
#define IREE_NCCL_PFN_DECL_STR_RETURN(ncclSymbolName, ...) \
  const char* (*ncclSymbolName)(__VA_ARGS__);
#include "iree/hal/drivers/hip/rccl_dynamic_symbol_table.h"  // IWYU pragma: export
#undef IREE_NCCL_PFN_DECL
#undef IREE_NCCL_PFN_DECL_STR_RETURN
} iree_hal_hip_nccl_dynamic_symbols_t;

#else

// Stub NCCL dynamic symbol state used when RCCL support is not compiled in.
typedef struct iree_hal_hip_nccl_dynamic_symbols_t {
  // Non-empty placeholder for C compatibility.
  int reserved;
} iree_hal_hip_nccl_dynamic_symbols_t;

#endif  // IREE_HAL_HIP_ENABLE_RCCL

// Initializes |out_syms| in-place with dynamically loaded NCCL symbols.
// iree_hal_hip_dynamic_symbols_deinitialize must be used to release the
// library resources.
#if IREE_HAL_HIP_ENABLE_RCCL
iree_status_t iree_hal_hip_nccl_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    const iree_hal_hip_dynamic_symbols_t* hip_library,
    iree_hal_hip_nccl_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_hip_nccl_dynamic_symbols_deinitialize(
    iree_hal_hip_nccl_dynamic_symbols_t* syms);
#else
static inline iree_status_t iree_hal_hip_nccl_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    const iree_hal_hip_dynamic_symbols_t* hip_library,
    iree_hal_hip_nccl_dynamic_symbols_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  out_syms->reserved = 0;
  return iree_ok_status();
}

static inline void iree_hal_hip_nccl_dynamic_symbols_deinitialize(
    iree_hal_hip_nccl_dynamic_symbols_t* syms) {}
#endif  // IREE_HAL_HIP_ENABLE_RCCL

static inline bool iree_hal_hip_nccl_dynamic_symbols_is_available(
    const iree_hal_hip_nccl_dynamic_symbols_t* syms) {
#if IREE_HAL_HIP_ENABLE_RCCL
  return syms && syms->dylib;
#else
  return false;
#endif  // IREE_HAL_HIP_ENABLE_RCCL
}

static inline iree_status_t
iree_hal_hip_nccl_dynamic_symbols_unavailable_status(void) {
#if IREE_HAL_HIP_ENABLE_RCCL
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "RCCL runtime library version %d.%d and greater not available; "
      "ensure installed and the shared library (rccl.dll/librccl.so) "
      "is on your PATH/LD_LIBRARY_PATH.",
      NCCL_MAJOR, NCCL_MINOR);
#else
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "RCCL support is not enabled in this HIP HAL build");
#endif  // IREE_HAL_HIP_ENABLE_RCCL
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_RCCL_DYNAMIC_SYMBOLS_H_
