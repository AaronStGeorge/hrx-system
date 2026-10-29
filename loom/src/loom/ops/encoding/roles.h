// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Semantic role queries for encoding values.

#ifndef LOOM_OPS_ENCODING_ROLES_H_
#define LOOM_OPS_ENCODING_ROLES_H_

#include "loom/ir/encoding.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_fact_context_t loom_fact_context_t;

// Returns a diagnostic-facing description of |role|.
iree_string_view_t loom_encoding_role_description(loom_encoding_role_t role);

// Returns the semantic role of a static encoding instance.
loom_encoding_role_t loom_encoding_static_role(const loom_module_t* module,
                                               const loom_encoding_t* encoding);

// Returns the semantic role of an SSA value with type `encoding`.
loom_encoding_role_t loom_encoding_value_role(const loom_module_t* module,
                                              loom_value_id_t value_id);

// Returns the semantic role expressed directly by a shaped type's encoding
// attachment. An absent attachment is the contextual dense address layout.
// Static attachments resolve through their family descriptor; SSA attachments
// use only the referenced value's declared encoding role and never inspect its
// producer.
loom_encoding_role_t loom_encoding_type_role(const loom_module_t* module,
                                             loom_type_t type);

// Returns the semantic role of a shaped type's encoding attachment, resolving
// an otherwise generic SSA encoding through its indexed facts. This never
// inspects the encoding value's producer.
loom_encoding_role_t loom_encoding_query_type_role(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_ROLES_H_
