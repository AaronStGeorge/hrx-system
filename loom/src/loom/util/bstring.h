// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// B-strings: length-prefixed byte strings for compile-time .rodata tables.
//
// A B-string is stored as [length_byte][char_data...]. The first byte
// encodes the string length (0-255 characters). The character data
// follows immediately with no required NUL terminator, though compile-
// time literals include one from C string concatenation.
//
// B-strings are the standard representation for compile-time-known
// strings in loom .rodata tables: op names, keyword strings, enum case
// names, type names, instance flag names. They encode length without
// runtime strlen, cost 1 byte of overhead per string, and support
// direct comparison against iree_string_view_t parser tokens.
//
// Typical usage in generated tables:
//
//   static const uint8_t my_keyword[] = "\x05" "hello";
//   if (loom_bstring_equal(my_keyword, token.text)) { ... }
//
// The "\xNN" prefix is the hex-encoded string length. The Python
// generator computes this at code-generation time.

#ifndef LOOM_UTIL_BSTRING_H_
#define LOOM_UTIL_BSTRING_H_

#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to a length-prefixed byte string: [length][data...].
typedef const uint8_t* loom_bstring_t;

// Returns the character count of a B-string.
static inline uint8_t loom_bstring_length(loom_bstring_t bstring) {
  return bstring[0];
}

// Returns a pointer to the character data (past the length byte).
static inline const char* loom_bstring_data(loom_bstring_t bstring) {
  return (const char*)(bstring + 1);
}

// Converts a B-string to an iree_string_view_t. Zero-cost: no strlen,
// just struct construction from the embedded length.
static inline iree_string_view_t loom_bstring_view(loom_bstring_t bstring) {
  return iree_make_string_view((const char*)(bstring + 1), bstring[0]);
}

// Returns true if a B-string equals an iree_string_view_t.
// Checks length first (single byte comparison), then memcmp only
// when lengths match. This is the hot-path replacement for
// iree_string_view_equal(view, iree_make_cstring_view(cstring))
// which would call strlen on every comparison.
static inline bool loom_bstring_equal(loom_bstring_t bstring,
                                      iree_string_view_t view) {
  return view.size == bstring[0] &&
         memcmp(view.data, bstring + 1, bstring[0]) == 0;
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_BSTRING_H_
