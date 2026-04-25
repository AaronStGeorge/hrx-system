// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_DESCRIPTORS_MANIFEST_H_
#define LOOM_CODEGEN_LOW_DESCRIPTORS_MANIFEST_H_

#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Appends a compact JSON manifest for |descriptor_set| to |builder|. The JSON
// is a diagnostic/tooling format, not the runtime representation.
iree_status_t loom_low_descriptor_set_format_manifest_json(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // LOOM_CODEGEN_LOW_DESCRIPTORS_MANIFEST_H_
