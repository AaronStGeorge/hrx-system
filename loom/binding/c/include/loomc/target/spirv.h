// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_H_
#define LOOMC_TARGET_SPIRV_H_

#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/workspace.h"

/// @file
/// SPIR-V target provider and binary emission.
///
/// This optional header is implemented by the SPIR-V target binding package.
/// Embedders that do not link SPIR-V support do not need to include or ship
/// this leaf. Embedders that do link it can create a SPIR-V target environment,
/// lower modules with a target pipeline, and explicitly emit SPIR-V binaries
/// from prepared target-low IR.
///
/// @par Example
/// Emit a prepared target-low module to an in-memory SPIR-V artifact:
///
/// @code{.c}
/// loomc_spirv_emit_options_t emit_options = {
///     .type = LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
///     .structure_size = sizeof(loomc_spirv_emit_options_t),
///     .identifier = loomc_make_cstring_view("kernel.spv"),
/// };
/// loomc_result_t* emit_result = NULL;
/// loomc_status_t status = loomc_spirv_emit_module(
///     target_environment, workspace, module, &emit_options,
///     loomc_allocator_system(), &emit_result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (loomc_result_succeeded(emit_result)) {
///   const loomc_artifact_t* artifact = loomc_result_artifact_at(emit_result,
///   0);
///   // Load or cache artifact->contents as SPIR-V binary bytes.
/// }
/// loomc_result_release(emit_result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// SPIR-V binary artifact format.
#define LOOMC_ARTIFACT_FORMAT_SPIRV "spirv"

/// SPIR-V binary emission options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_spirv_emit_options_t)`, and fill the requested fields.
typedef struct loomc_spirv_emit_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future SPIR-V emission options.
  const void* next;

  /// Artifact identifier for the returned SPIR-V binary. Empty selects
  /// `module.spv`.
  loomc_string_view_t identifier;
} loomc_spirv_emit_options_t;

/// Creates a target environment containing the SPIR-V target package.
///
/// @param allocator Host allocator used for target-environment storage.
/// @param out_target_environment Receives one retained target environment on
/// success.
/// @return OK when the target environment was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_target_environment_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_spirv(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

/// Emits one SPIR-V binary artifact from prepared target-low IR.
///
/// The module must already contain SPIR-V target-low function definitions, such
/// as IR produced by a prepared-low target pipeline. This function does not run
/// source-to-low lowering or target preparation passes.
///
/// @param target_environment SPIR-V target environment used to resolve
/// target-low descriptor rows.
/// @param workspace Invocation-local scratch workspace.
/// @param module Mutable module containing prepared SPIR-V target-low IR.
/// @param options Emission options, or `NULL` for defaults.
/// @param allocator Host allocator used for result-owned storage.
/// @param out_result Receives a retained result containing diagnostics and, on
/// success, one SPIR-V binary artifact.
/// @return OK when the emission operation completed far enough to report a
/// result. Non-OK statuses represent API misuse or infrastructure failures
/// before a result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`.
///
/// @lifetime
/// Returned artifacts do not borrow from `workspace` or `module` and remain
/// valid until the result is released.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_emit_module(
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module,
    const loomc_spirv_emit_options_t* options, loomc_allocator_t allocator,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_H_
