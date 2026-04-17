// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared target record helpers.
//
// These helpers operate only on target-neutral records from types.h. Target
// families own the records themselves; this file only defines generic mechanics
// such as compact equality fingerprints for transient cache keys and
// diagnostics. These fingerprints are not durable artifact identity.

#ifndef LOOM_TARGET_RECORDS_H_
#define LOOM_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Computes a content fingerprint for |snapshot|. The fingerprint covers
// every v0 snapshot field in declaration order and is suitable for transient
// cache-key inputs, not for durable module identity or adversarial hashing.
iree_status_t loom_target_snapshot_fingerprint(
    const loom_target_snapshot_t* snapshot, uint64_t* out_fingerprint);

// Computes a content fingerprint for |export_plan|. The fingerprint
// covers every v0 export-plan field in declaration order and is suitable for
// transient cache-key inputs, not for durable module identity or adversarial
// hashing.
iree_status_t loom_target_export_plan_fingerprint(
    const loom_target_export_plan_t* export_plan, uint64_t* out_fingerprint);

// Computes a content fingerprint for |config|. The fingerprint covers
// every v0 config field in declaration order and is suitable for cache-key
// inputs, not for durable module identity or adversarial hashing.
iree_status_t loom_target_config_fingerprint(const loom_target_config_t* config,
                                             uint64_t* out_fingerprint);

// Computes a content fingerprint for |bundle| from its name and the
// content fingerprints of the referenced snapshot, export plan, and config.
iree_status_t loom_target_bundle_fingerprint(const loom_target_bundle_t* bundle,
                                             uint64_t* out_fingerprint);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_RECORDS_H_
