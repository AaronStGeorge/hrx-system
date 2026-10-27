// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/elf_compiler_provider.h"

#include "loom/target/arch/x86/facts.h"
#include "loom/target/arch/x86/provider.h"
#include "loom/target/emit/native/x86/module.h"

static iree_status_t loom_x86_compiler_emit_module(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  return loom_x86_module_emit(request, &loom_x86_target_fact_type,
                              LOOM_NATIVE_MODULE_FORMAT_OBJECT, out_emitted,
                              out_artifact);
}

static iree_status_t loom_x86_compiler_emit_image(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  return loom_x86_module_emit(request, &loom_x86_target_fact_type,
                              LOOM_NATIVE_MODULE_FORMAT_SHARED, out_emitted,
                              out_artifact);
}

static iree_status_t loom_x86_compiler_emit_hal_library(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  return loom_x86_module_emit(request, &loom_x86_target_fact_type,
                              LOOM_NATIVE_MODULE_FORMAT_HAL_LIBRARY,
                              out_emitted, out_artifact);
}

static const loom_target_emitter_t loom_x86_hal_library_emitter = {
    .name = IREE_SVL("x86-hal"),
    .public_artifact_format = IREE_SVL("x86-hal"),
    .default_identifier = IREE_SVL("module.so"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pipeline_options =
        {
            .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
        },
    .emit = loom_x86_compiler_emit_hal_library,
};

static const loom_target_emitter_t loom_x86_image_emitter = {
    .name = IREE_SVL("x86-elf-shared"),
    .public_artifact_format = IREE_SVL("x86-elf-shared"),
    .default_identifier = IREE_SVL("module.so"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pipeline_options =
        {
            .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
        },
    .emit = loom_x86_compiler_emit_image,
};

static const loom_target_emitter_t loom_x86_module_emitter = {
    .name = IREE_SVL("x86-elf"),
    .public_artifact_format = IREE_SVL("x86-elf"),
    .default_identifier = IREE_SVL("module.o"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pipeline_options =
        {
            .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
        },
    .emit = loom_x86_compiler_emit_module,
};

const loom_target_provider_t loom_x86_elf_compiler_provider = {
    .emitter_list =
        {
            .values =
                (const loom_target_emitter_t* const[]){
                    &loom_x86_module_emitter, &loom_x86_image_emitter,
                    &loom_x86_hal_library_emitter},
            .count = 3,
        },
    .canonical_module_emitter = &loom_x86_module_emitter,
    .canonical_module_fact_type = &loom_x86_target_fact_type,
    .canonical_kernel_emitter = &loom_x86_hal_library_emitter,
    .canonical_kernel_fact_type = &loom_x86_target_fact_type,
};

static const loom_target_provider_t* const kLoomX86ElfCompilerProviders[] = {
    &loom_x86_target_provider,
    &loom_x86_elf_compiler_provider,
};

const loom_target_provider_set_t loom_x86_elf_compiler_provider_set = {
    .providers = kLoomX86ElfCompilerProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomX86ElfCompilerProviders),
};
