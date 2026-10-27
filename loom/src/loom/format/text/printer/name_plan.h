// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_NAME_PLAN_H_
#define LOOM_FORMAT_TEXT_PRINTER_NAME_PLAN_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/printer.h"
#include "loom/ir/ir.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonical SSA name resolutions shared by one print invocation.
//
// Construction indexes explicit names by parser scope and resolves all
// same-scope duplicates, shadowed captures, and generated-name collisions once.
// A scope walk checks ordinary and embedded type/attribute references against
// indexed bindings; captured values receive unambiguous spellings. Generated
// candidates have disjoint value-ID suffixes and cannot equal explicit names in
// any scope. Emission is then an O(1) value-id lookup. The plan owns only the
// resolutions in a scoped arena backed by the module block pool; its
// construction index is discarded from that arena before preparation
// returns.
//
// Prepares canonical SSA name resolutions if not already prepared.
iree_status_t loom_print_name_plan_prepare(loom_text_printer_t* printer);

// Releases name storage, leaving the borrowed module and lazy state intact.
void loom_print_name_plan_discard(loom_text_printer_t* printer);

// Prints |value_id|'s canonical SSA reference.
//
// The printer is prepared lazily on the first valid reference and must be
// deinitialized by the caller. All references in a composite type or attribute
// share that plan; scalar atoms require no module traversal or allocation.
iree_status_t loom_print_name_plan_write_value_ref(loom_text_printer_t* printer,
                                                   loom_output_stream_t* stream,
                                                   loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_NAME_PLAN_H_
