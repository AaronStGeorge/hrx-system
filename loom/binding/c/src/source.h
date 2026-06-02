// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_SOURCE_STORAGE_H_
#define LOOMC_SOURCE_STORAGE_H_

#include "loomc/source.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copies string into allocator-owned storage.
LOOMC_API_PRIVATE loomc_status_t loomc_source_copy_string(
    loomc_string_view_t source, loomc_allocator_t allocator,
    loomc_string_view_t* out_string);

// Creates a source that takes ownership of allocator-backed contents.
LOOMC_API_PRIVATE loomc_status_t loomc_source_create_take_contents(
    loomc_source_format_t format, loomc_string_view_t identifier,
    loomc_byte_span_t contents, loomc_allocator_t allocator,
    loomc_source_t** out_source);

// Transfers allocator-owned source contents out of a source.
LOOMC_API_PRIVATE loomc_status_t loomc_source_take_contents(
    loomc_source_t* source, loomc_byte_span_t* out_contents);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_SOURCE_STORAGE_H_
