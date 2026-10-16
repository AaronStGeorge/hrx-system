// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_EXECUTABLES_H_
#define LOOM_SERVE_SERVICE_FINITE_EXECUTABLES_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/service/execution_reconfiguration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Code ownership for one serialized finite model. Management selects a source
// generation for future batches; a prepared batch retains its own generation.
// Device results have no dependency on this owner after producer retirement.
typedef struct loom_serve_finite_executables_t loom_serve_finite_executables_t;

typedef struct loom_serve_finite_executable_options_t {
  // Shared device and compiler host, borrowed through destruction.
  loom_serve_device_t* device;
  // Original indexed source generation; its report outlives this owner.
  loom_serve_jit_t* jit;
  // Immutable cold declarations, borrowed through every replacement job.
  const loom_serve_preparation_t* preparation;
  // Original checkpoint placement, or NULL for a parameter-free direct model.
  const loom_cmd_program_t* layout;
  // Original descriptors in preparation order, borrowed through destruction.
  const loom_serve_stage_t* stages;
  // Immutable checkpoint roots, borrowed through recording and destruction.
  iree_hal_buffer_t* const* parameters;
  // Command recording policy shared by original and replacement code.
  iree_hal_command_buffer_mode_t command_mode;
  // Borrowed runner, initially bound to stages and outliving this owner.
  iree_vm_module_t* runner;
  // Bounds reusable request products without restricting one active request.
  loom_serve_variant_cache_options_t cache;
} loom_serve_finite_executable_options_t;

iree_status_t loom_serve_finite_executables_allocate(
    const loom_serve_finite_executable_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_finite_executables_t** out_executables);

// All accepted compilation and batch execution has retired. Management payloads
// have been disposed before destruction; a captured batch may remain after a
// terminal submission failure. NULL is accepted.
void loom_serve_finite_executables_free(
    loom_serve_finite_executables_t* executables);

loom_serve_execution_update_t loom_serve_finite_executables_update(
    loom_serve_finite_executables_t* executables);

// Begins the sole batch's code borrow before source preparation. The previous
// batch has been released. Serialized with management selection.
void loom_serve_finite_executables_capture(
    loom_serve_finite_executables_t* executables);

// Transfers source declarations after CPU preparation, including NULL for a
// cold-table invocation. Serialized with owner advance and selection.
void loom_serve_finite_executables_declare(
    loom_serve_finite_executables_t* executables,
    loom_serve_preparation_t* preparation);

// Installs a nonblocking owner wake before accepting compilations. Detachment
// follows terminal advance of every accepted request compilation.
void loom_serve_finite_executables_set_observer(
    loom_serve_finite_executables_t* executables,
    loom_serve_wake_callback_t observer);

// Resolves request declarations and binds the captured table before parameter
// admission. A miss submits compilation without waiting; completion schedules
// another owner turn. Cancellation skips unstarted work but joins accepted
// producers. Terminal failure retains the batch until release. No checkpoint
// residency or device work is acquired here.
iree_status_t loom_serve_finite_executables_advance(
    loom_serve_finite_executables_t* executables, bool cancel,
    bool* out_complete);

// Releases an unsubmitted or fully retired batch; no compiler/device wait.
void loom_serve_finite_executables_release(
    loom_serve_finite_executables_t* executables);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_FINITE_EXECUTABLES_H_
