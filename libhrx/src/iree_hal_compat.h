// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_IREE_HAL_COMPAT_H_
#define LIBHRX_SRC_IREE_HAL_COMPAT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// HRX was imported from a tree that used the older HAL "export" terminology.
// The reduced runtime has the newer "executable function" API. Keep this shim
// local to libhrx so the imported runtime code can be adapted incrementally
// without leaking the old HAL names into public headers.
typedef uint32_t iree_hal_executable_export_ordinal_t;
typedef iree_hal_executable_function_info_t iree_hal_executable_export_info_t;
typedef iree_hal_executable_function_parameter_t
    iree_hal_executable_export_parameter_t;

#define IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_CONSTANT \
  IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT
#define IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING \
  IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING
#define IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BUFFER_PTR \
  IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR

static inline iree_host_size_t iree_hal_executable_export_count(
    iree_hal_executable_t* executable) {
  return iree_hal_executable_function_count(executable);
}

static inline iree_status_t iree_hal_executable_export_info(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t ordinal,
    iree_hal_executable_export_info_t* out_info) {
  return iree_hal_executable_function_info(
      executable, iree_hal_executable_function_from_index(ordinal), out_info);
}

static inline iree_status_t iree_hal_executable_export_parameters(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t ordinal, iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  return iree_hal_executable_function_parameters(
      executable, iree_hal_executable_function_from_index(ordinal), capacity,
      out_parameters);
}

static inline iree_status_t iree_hal_executable_lookup_export_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_ordinal) {
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_lookup_function_by_name(executable, name, &function));
  if (function.value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "executable function is not a dense ordinal");
  }
  *out_ordinal = (uint32_t)function.value;
  return iree_ok_status();
}

#endif  // LIBHRX_SRC_IREE_HAL_COMPAT_H_
