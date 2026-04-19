// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Materialization helpers for target record IR ops.
//
// Target dialect ops are the textual/bytecode representation of target
// records. These helpers convert verified target.* ops into the compact
// target-neutral structs consumed by LLVMIR, target-low, and future emitters.

#ifndef LOOM_TARGET_IR_RECORDS_H_
#define LOOM_TARGET_IR_RECORDS_H_

#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_ir_bundle_storage_t {
  // Materialized snapshot record owned by this storage object.
  loom_target_snapshot_t snapshot;
  // Materialized export plan record owned by this storage object.
  loom_target_export_plan_t export_plan;
  // Materialized target config record owned by this storage object.
  loom_target_config_t config;
  // Bundle view pointing at snapshot, export_plan, and config above.
  loom_target_bundle_t bundle;
} loom_target_ir_bundle_storage_t;

// Rebinds |storage->bundle| to the records embedded in |storage|. Call this
// after copying a storage object by value; the embedded bundle is a view and
// its pointers must refer to the same enclosing storage object.
static inline void loom_target_ir_bundle_storage_rebind(
    loom_target_ir_bundle_storage_t* storage) {
  IREE_ASSERT_ARGUMENT(storage);
  storage->bundle.snapshot = &storage->snapshot;
  storage->bundle.export_plan = &storage->export_plan;
  storage->bundle.config = &storage->config;
}

// Converts a target.snapshot op into a target snapshot record. Borrowed strings
// in |out_snapshot| point into |module|.
iree_status_t loom_target_ir_snapshot_from_op(
    const loom_module_t* module, const loom_op_t* snapshot_op,
    loom_target_snapshot_t* out_snapshot);

// Converts a target.export op into a target export-plan record. Borrowed
// strings in |out_export_plan| point into |module|.
iree_status_t loom_target_ir_export_plan_from_op(
    const loom_module_t* module, const loom_op_t* export_op,
    loom_target_export_plan_t* out_export_plan);

// Converts a target.config op into a target config record. Borrowed strings in
// |out_config| point into |module|.
iree_status_t loom_target_ir_config_from_op(const loom_module_t* module,
                                            const loom_op_t* config_op,
                                            loom_target_config_t* out_config);

// Converts the resolved ops referenced by a target.bundle into shared target
// records. Symbol resolution and kind checking are owned by the caller so this
// helper can stay diagnostic-policy-free.
iree_status_t loom_target_ir_bundle_from_ops(
    const loom_module_t* module, const loom_op_t* bundle_op,
    const loom_op_t* snapshot_op, const loom_op_t* export_op,
    const loom_op_t* config_op, loom_target_ir_bundle_storage_t* out_storage);

// Resolves a module-local target.bundle symbol by name and materializes the
// target records it references. |bundle_name| does not include a leading '@'.
// Borrowed strings in |out_storage| point into |module|.
iree_status_t loom_target_ir_bundle_from_symbol_name(
    const loom_module_t* module, iree_string_view_t bundle_name,
    loom_target_ir_bundle_storage_t* out_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_IR_RECORDS_H_
