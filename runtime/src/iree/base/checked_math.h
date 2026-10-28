// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_CHECKED_MATH_H_
#define IREE_BASE_CHECKED_MATH_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Checked arithmetic helpers
//===----------------------------------------------------------------------===//

// Performs a checked unsigned 64-bit addition of |a| and |b|, storing the
// result in |out_result|. Returns true if the addition succeeded without
// overflow, false if overflow occurred (|out_result| is undefined on
// overflow).
static inline bool iree_checked_add_u64(uint64_t a, uint64_t b,
                                        uint64_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out_result);
#else
  if (a > UINT64_MAX - b) {
    return false;
  }
  *out_result = a + b;
  return true;
#endif
}

// Performs a checked unsigned 64-bit multiplication of |a| and |b|, storing
// the result in |out_result|. Returns true if the multiplication succeeded
// without overflow, false if overflow occurred (|out_result| is undefined on
// overflow).
static inline bool iree_checked_mul_u64(uint64_t a, uint64_t b,
                                        uint64_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out_result);
#else
  if (b != 0 && a > UINT64_MAX / b) {
    return false;
  }
  *out_result = a * b;
  return true;
#endif
}

// Aligns |value| up to |alignment| with overflow checking.
// |alignment| must be a power of two.
// Returns true if the alignment succeeded without overflow, false if overflow
// occurred (|out_aligned| is undefined on overflow).
static inline bool iree_checked_align_u64(uint64_t value, uint64_t alignment,
                                          uint64_t* out_aligned) {
  uint64_t padded = 0;
  if (!iree_checked_add_u64(value, alignment - 1, &padded)) {
    return false;
  }
  *out_aligned = padded & ~(alignment - 1);
  return true;
}

// Performs a checked signed 32-bit addition of |a| and |b|, storing the result
// in |out_result|. Returns true if the addition succeeded without overflow,
// false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_checked_add_i32(int32_t a, int32_t b,
                                        int32_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out_result);
#else
  if ((b > 0 && a > INT32_MAX - b) || (b < 0 && a < INT32_MIN - b)) {
    return false;
  }
  *out_result = a + b;
  return true;
#endif
}

// Performs a checked signed 32-bit multiplication of |a| and |b|, storing the
// result in |out_result|. Returns true if the multiplication succeeded without
// overflow, false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_checked_mul_i32(int32_t a, int32_t b,
                                        int32_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out_result);
#else
  int64_t result = (int64_t)a * (int64_t)b;
  if (result < INT32_MIN || result > INT32_MAX) {
    return false;
  }
  *out_result = (int32_t)result;
  return true;
#endif
}

// Performs checked computation of |base| + |lhs| * |rhs|, storing the result in
// |out_result|. Returns true if the computation succeeded without overflow,
// false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_checked_mul_add_i32(int32_t base, int32_t lhs,
                                            int32_t rhs, int32_t* out_result) {
  int32_t product = 0;
  if (!iree_checked_mul_i32(lhs, rhs, &product)) {
    return false;
  }
  return iree_checked_add_i32(base, product, out_result);
}

// Performs a checked signed 64-bit addition of |a| and |b|, storing the result
// in |out_result|. Returns true if the addition succeeded without overflow,
// false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_checked_add_i64(int64_t a, int64_t b,
                                        int64_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out_result);
#else
  if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
    return false;
  }
  *out_result = a + b;
  return true;
#endif
}

// Performs a checked signed 64-bit subtraction of |b| from |a|, storing the
// result in |out_result|. Returns true if the subtraction succeeded without
// overflow, false if overflow occurred (|out_result| is undefined on
// overflow).
static inline bool iree_checked_sub_i64(int64_t a, int64_t b,
                                        int64_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_sub_overflow)
  return !__builtin_sub_overflow(a, b, out_result);
#else
  if ((b > 0 && a < INT64_MIN + b) || (b < 0 && a > INT64_MAX + b)) {
    return false;
  }
  *out_result = a - b;
  return true;
#endif
}

// Performs a checked signed 64-bit multiplication of |a| and |b|, storing the
// result in |out_result|. Returns true if the multiplication succeeded without
// overflow, false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_checked_mul_i64(int64_t a, int64_t b,
                                        int64_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out_result);
#else
  if (a == 0 || b == 0) {
    *out_result = 0;
    return true;
  }
  if (a == -1 && b == INT64_MIN) {
    return false;
  }
  if (b == -1 && a == INT64_MIN) {
    return false;
  }
  if (a > 0) {
    if (b > 0) {
      if (a > INT64_MAX / b) {
        return false;
      }
    } else if (b < INT64_MIN / a) {
      return false;
    }
  } else if (b > 0) {
    if (a < INT64_MIN / b) {
      return false;
    }
  } else if (a < INT64_MAX / b) {
    return false;
  }
  *out_result = a * b;
  return true;
#endif
}

// Performs checked computation of |base| + |lhs| * |rhs|, storing the result in
// |out_result|. Returns true if the computation succeeded without overflow,
// false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_checked_mul_add_i64(int64_t base, int64_t lhs,
                                            int64_t rhs, int64_t* out_result) {
  int64_t product = 0;
  if (!iree_checked_mul_i64(lhs, rhs, &product)) {
    return false;
  }
  return iree_checked_add_i64(base, product, out_result);
}

// Performs a checked addition of |a| and |b|, storing the result in
// |out_result|. Returns true if the addition succeeded without overflow, false
// if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_host_size_checked_add(iree_host_size_t a,
                                              iree_host_size_t b,
                                              iree_host_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out_result);
#else
  if (a > IREE_HOST_SIZE_MAX - b) {
    return false;
  }
  *out_result = a + b;
  return true;
#endif
}

// Performs a checked multiplication of |a| and |b|, storing the result in
// |out_result|. Returns true if the multiplication succeeded without overflow,
// false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_host_size_checked_mul(iree_host_size_t a,
                                              iree_host_size_t b,
                                              iree_host_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out_result);
#else
  if (b != 0 && a > IREE_HOST_SIZE_MAX / b) {
    return false;
  }
  *out_result = a * b;
  return true;
#endif
}

// Performs checked computation of |base| + |count| * |element_size|, storing
// the result in |out_result|. Returns true if the computation succeeded without
// overflow, false if overflow occurred (|out_result| is undefined on overflow).
// This is the common pattern for computing allocation sizes with a header.
static inline bool iree_host_size_checked_mul_add(
    iree_host_size_t base, iree_host_size_t count,
    iree_host_size_t element_size, iree_host_size_t* out_result) {
  iree_host_size_t product = 0;
  if (!iree_host_size_checked_mul(count, element_size, &product)) {
    return false;
  }
  return iree_host_size_checked_add(base, product, out_result);
}

// Device size variants for remote device allocations.

static inline bool iree_device_size_checked_add(
    iree_device_size_t a, iree_device_size_t b,
    iree_device_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out_result);
#else
  if (a > IREE_DEVICE_SIZE_MAX - b) {
    return false;
  }
  *out_result = a + b;
  return true;
#endif
}

static inline bool iree_device_size_checked_mul(
    iree_device_size_t a, iree_device_size_t b,
    iree_device_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out_result);
#else
  if (b != 0 && a > IREE_DEVICE_SIZE_MAX / b) {
    return false;
  }
  *out_result = a * b;
  return true;
#endif
}

static inline bool iree_device_size_checked_mul_add(
    iree_device_size_t base, iree_device_size_t count,
    iree_device_size_t element_size, iree_device_size_t* out_result) {
  iree_device_size_t product = 0;
  if (!iree_device_size_checked_mul(count, element_size, &product)) {
    return false;
  }
  return iree_device_size_checked_add(base, product, out_result);
}

// Aligns |value| up to |alignment| with overflow checking.
// |alignment| must be a power of two.
// Returns true if the alignment succeeded without overflow, false if overflow
// occurred (|out_aligned| is undefined on overflow).
static inline bool iree_host_size_checked_align(iree_host_size_t value,
                                                iree_host_size_t alignment,
                                                iree_host_size_t* out_aligned) {
  iree_host_size_t padded = 0;
  if (!iree_host_size_checked_add(value, alignment - 1, &padded)) {
    return false;
  }
  *out_aligned = padded & ~(alignment - 1);
  return true;
}

static inline bool iree_device_size_checked_align(
    iree_device_size_t value, iree_device_size_t alignment,
    iree_device_size_t* out_aligned) {
  iree_device_size_t padded = 0;
  if (!iree_device_size_checked_add(value, alignment - 1, &padded)) {
    return false;
  }
  *out_aligned = padded & ~(alignment - 1);
  return true;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_CHECKED_MATH_H_
