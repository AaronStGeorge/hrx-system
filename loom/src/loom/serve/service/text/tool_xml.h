// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_TOOL_XML_H_
#define LOOM_SERVE_SERVICE_TEXT_TOOL_XML_H_

#include "iree/base/api.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates the optional tools.parse_xml(schemas, text) -> buffer capability.
// Both inputs are UTF-8 buffers. The result is a JSON array of standard
// function tool calls, with JSON-encoded argument strings and no assigned IDs.
// Parameter type hints distinguish literal strings from JSON values; they do
// not constrain generated names or arguments. Unknown arguments remain literal
// strings, and duplicate members retain their original order. The harness owns
// required-field, schema, permission, and execution validation.
// Source explicitly selects this grammar by importing it; native chat handling
// consumes only the returned function records. The environment outlives the
// module and every returned buffer.
iree_status_t loom_serve_text_tool_xml_module_create(
    iree_vm_environment_t* environment, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_TOOL_XML_H_
