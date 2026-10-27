// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/aie2p/elf_compiler_provider.h"

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/aie2p/provider.h"

static const loom_target_emitter_t* const kAie2pXdnaArtifactEmitters[] = {
    &loom_aie2p_xdna_artifact_emitter,
};

const loom_target_provider_t loom_aie2p_xdna_artifact_provider = {
    .emitter_list =
        {
            .values = kAie2pXdnaArtifactEmitters,
            .count = IREE_ARRAYSIZE(kAie2pXdnaArtifactEmitters),
        },
    .canonical_kernel_emitter = &loom_aie2p_xdna_artifact_emitter,
    .canonical_kernel_fact_type = &loom_aie2p_target_fact_type,
};

static const loom_target_provider_t* const kAie2pCompilerProviders[] = {
    &loom_aie2p_target_provider,
    &loom_aie2p_xdna_artifact_provider,
};

const loom_target_provider_set_t loom_aie2p_compiler_provider_set = {
    .providers = kAie2pCompilerProviders,
    .provider_count = IREE_ARRAYSIZE(kAie2pCompilerProviders),
};
