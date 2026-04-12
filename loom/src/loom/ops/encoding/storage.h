// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical storage composition helpers for encoding values.
//
// `#physical_storage` composes an address layout with an encoded storage
// schema in the existing shaped-type encoding attachment slot:
//
//   %storage = encoding.define #physical_storage {
//     layout = %layout : encoding,
//     schema = %schema : encoding
//   } : encoding
//
// Memory operations still see physical storage. These helpers recover the
// address-layout part for address arithmetic without making loads/stores decode
// schema bytes into logical elements.

#ifndef LOOM_OPS_ENCODING_STORAGE_H_
#define LOOM_OPS_ENCODING_STORAGE_H_

#include "loom/ir/encoding.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Built-in encoding family verifier for `#physical_storage`.
extern const loom_encoding_vtable_t loom_encoding_physical_storage_vtable;

// Resolves |value_id| to the dynamic address-layout op that memory address
// arithmetic should use. Direct `encoding.layout.*` values resolve to their
// defining op. `#physical_storage` values resolve through their `layout`
// parameter. Returns false when the value is not a known dynamic address layout
// or has only a static nested layout.
bool loom_encoding_resolve_address_layout_op(const loom_module_t* module,
                                             loom_value_id_t value_id,
                                             const loom_op_t** out_layout_op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_STORAGE_H_
