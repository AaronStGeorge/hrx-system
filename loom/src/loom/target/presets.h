// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target preset expansion.
//
// target.preset is compact production authoring IR for command-line-style
// target selection. It must be expanded early into concrete target records so
// legality, scheduling, allocation, emission, and bytecode permanence consume
// the same explicit target.snapshot/export/config/bundle records.

#ifndef LOOM_TARGET_PRESETS_H_
#define LOOM_TARGET_PRESETS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/preset_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Expands all target.preset records in |module| into concrete target records.
//
// Each preset keeps its user-facing symbol: target.preset @name becomes
// target.bundle @name. Generated sibling records use @name__snapshot,
// @name__export, and @name__config and fail on collisions instead of silently
// uniquing. The selected preset registry supplies the record payloads, and the
// preset's source symbol supplies the concrete target.export source plus the
// default exported symbol name.
iree_status_t loom_target_expand_presets(
    loom_module_t* module, const loom_target_preset_registry_t* registry,
    iree_host_size_t* out_expanded_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PRESETS_H_
