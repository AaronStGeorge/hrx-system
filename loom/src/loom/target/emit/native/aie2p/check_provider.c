// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/aie2p/check_provider.h"

#include "loom/target/arch/amd/xdna/aie2p/check/artifact.h"

static const loom_check_emit_provider_t* const
    kAie2pArtifactCheckEmitProviders[] = {
        &loom_aie2p_artifact_check_emit_provider,
};

const loom_check_provider_t loom_aie2p_artifact_check_provider = {
    .name = IREE_SVL("amd-xdna-aie2p-artifact"),
    .emit_providers = kAie2pArtifactCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kAie2pArtifactCheckEmitProviders),
};
