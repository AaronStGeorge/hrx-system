// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check/provider.h"

#include "loom/target/arch/amdgpu/check/occupancy.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/packet_diagnostics.h"
#include "loom/target/emit/native/amdgpu/check/loom_check.h"

static const loom_check_emit_provider_t* const kLoomAmdgpuCheckEmitProviders[] =
    {
        &loom_amdgpu_occupancy_loom_check_emit_provider,
        &loom_amdgpu_native_loom_check_emit_provider,
};

static const loom_target_low_legality_provider_t* const
    kLoomAmdgpuLowLegalityProviders[] = {
        &loom_amdgpu_low_legality_provider_storage,
};

static const loom_target_low_packet_diagnostic_provider_t* const
    kLoomAmdgpuLowPacketDiagnosticProviders[] = {
        &loom_amdgpu_low_packet_diagnostic_provider_storage,
};

const loom_check_provider_t loom_amdgpu_check_provider = {
    .name = IREE_SVL("amdgpu"),
    .register_context = loom_amdgpu_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_amdgpu_low_lower_policy_registry_initialize,
    .low_legality_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowLegalityProviders),
            .values = kLoomAmdgpuLowLegalityProviders,
        },
    .low_packet_diagnostic_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowPacketDiagnosticProviders),
            .values = kLoomAmdgpuLowPacketDiagnosticProviders,
        },
    .emit_providers = kLoomAmdgpuCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckEmitProviders),
};
