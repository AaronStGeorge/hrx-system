// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typedef int iree_status_t;

#define IREE_STATUS_OK 0

iree_status_t iree_clang_tidy_trace_status_source(void);

#define IREE_TRACE_ZONE_BEGIN(zone_id) int zone_id = 0
#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) int zone_id = 0
#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length) \
  int zone_id = 0
#define IREE_TRACE_ZONE_END(zone_id) (void)(zone_id)

#define IREE_RETURN_IF_ERROR(expr)                           \
  do {                                                       \
    iree_status_t status_macro = (expr);                     \
    if (status_macro != IREE_STATUS_OK) return status_macro; \
  } while (0)

#define IREE_RETURN_AND_END_ZONE_IF_ERROR(zone_id, expr) \
  IREE_RETURN_IF_ERROR(expr)

#define IREE_RETURN_AND_END_ZONE(zone_id, expr) return (expr)

#define HRX_TRACE_ZONE_BEGIN(zone_id, name_literal) int zone_id = 0
#define HRX_TRACE_ZONE_END(zone_id) (void)(zone_id)
#define HRX_RETURN_IF_IREE_ERROR(expr)                       \
  do {                                                       \
    iree_status_t status_macro = (expr);                     \
    if (status_macro != IREE_STATUS_OK) return status_macro; \
  } while (0)
#define HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(zone_id, expr) \
  do {                                                       \
    iree_status_t status_macro = (expr);                     \
    if (status_macro != IREE_STATUS_OK) {                    \
      HRX_TRACE_ZONE_END(zone_id);                           \
      return status_macro;                                   \
    }                                                        \
  } while (0)
#define HRX_RETURN_AND_END_ZONE(zone_id, expr) \
  do {                                         \
    HRX_TRACE_ZONE_END(zone_id);               \
    return (expr);                             \
  } while (0)

#define HIP_RETURN_ERROR(error) \
  do {                          \
    int error_macro = (error);  \
    return error_macro;         \
  } while (0)

iree_status_t iree_clang_tidy_trace_zone_balanced(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_nested_balanced(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_BEGIN_NAMED(z1, "inner");
  IREE_TRACE_ZONE_END(z1);
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_cleanup_label(int fail) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = IREE_STATUS_OK;
  if (fail) {
    status = iree_clang_tidy_trace_status_source();
    goto cleanup;
  }
cleanup:
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_clang_tidy_trace_zone_branch_return_balanced(int fail) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (fail) {
    IREE_TRACE_ZONE_END(z0);
    return iree_clang_tidy_trace_status_source();
  }
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

int iree_clang_tidy_trace_zone_branch_macro_return_balanced(int fail) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (fail) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(1);
  }
  IREE_TRACE_ZONE_END(z0);
  return 0;
}

iree_status_t iree_clang_tidy_trace_zone_branch_local_zone(int flag) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (flag) {
    IREE_TRACE_ZONE_BEGIN(z1);
    IREE_TRACE_ZONE_END(z1);
  }
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_return_and_end_if_error(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_clang_tidy_trace_status_source());
  IREE_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_return_and_end(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE(z0, iree_clang_tidy_trace_status_source());
}

iree_status_t iree_clang_tidy_trace_zone_hrx_return_and_end_if_error(void) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx");
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(z0,
                                        iree_clang_tidy_trace_status_source());
  HRX_TRACE_ZONE_END(z0);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_plain_return_if_error(void) {
  IREE_TRACE_ZONE_BEGIN(plain_return_if_error_zone);
  IREE_RETURN_IF_ERROR(iree_clang_tidy_trace_status_source());
  IREE_TRACE_ZONE_END(plain_return_if_error_zone);
  return IREE_STATUS_OK;
}

iree_status_t iree_clang_tidy_trace_zone_hrx_plain_return_if_error(void) {
  HRX_TRACE_ZONE_BEGIN(hrx_plain_return_zone, "hrx");
  HRX_RETURN_IF_IREE_ERROR(iree_clang_tidy_trace_status_source());
  HRX_TRACE_ZONE_END(hrx_plain_return_zone);
  return IREE_STATUS_OK;
}
