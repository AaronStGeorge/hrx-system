// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/coverage.h"

#define LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES                                 \
  (LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |                                     \
   LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |                                      \
   LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER |                                   \
   LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE | \
   LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT |                                      \
   LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |                                  \
   LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP)

#define LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES \
  (LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES |        \
   LOOM_TARGET_COVERAGE_PHASE_INSPECT | LOOM_TARGET_COVERAGE_PHASE_RUN)

#define LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES                                   \
  (LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |                                     \
   LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |                                      \
   LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE | \
   LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT)

#define LOOM_AMDGPU_AUTHORED_LOW_GFX11_PHASES \
  (LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES |     \
   LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE)

#define LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES \
  (LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES |  \
   LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER)

#define LOOM_AMDGPU_NATIVE_LOW_GFX11_PHASES   \
  (LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES |       \
   LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE | \
   LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP)

#define LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES \
  LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES

#define LOOM_AMDGPU_COVERAGE_ROW(target, descriptor_set, category_, semantic, \
                                 expected, supported, gap)                    \
  {                                                                           \
      .target_key = IREE_SVL(target),                                         \
      .descriptor_set_key = IREE_SVL(descriptor_set),                         \
      .category = IREE_SVL(category_),                                        \
      .semantic_tag = IREE_SVL(semantic),                                     \
      .expected_phases = (expected),                                          \
      .supported_phases = (supported),                                        \
      .gap_key = IREE_SVL(gap),                                               \
  }

#define LOOM_AMDGPU_GFX11_ROW(category, semantic, expected, supported, gap) \
  LOOM_AMDGPU_COVERAGE_ROW("amdgpu-gfx11", "amdgpu.gfx11.core", category,   \
                           semantic, expected, supported, gap)

#define LOOM_AMDGPU_GFX12_ROW(category, semantic, expected, supported, gap) \
  LOOM_AMDGPU_COVERAGE_ROW("amdgpu-gfx12", "amdgpu.gfx12.core", category,   \
                           semantic, expected, supported, gap)

#define LOOM_AMDGPU_GFX1250_ROW(category, semantic, expected, supported, gap) \
  LOOM_AMDGPU_COVERAGE_ROW("amdgpu-gfx1250", "amdgpu.gfx1250.core", category, \
                           semantic, expected, supported, gap)

#define LOOM_AMDGPU_GFX950_ROW(category, semantic, expected, supported, gap) \
  LOOM_AMDGPU_COVERAGE_ROW("amdgpu-gfx950", "amdgpu.gfx950.core", category,  \
                           semantic, expected, supported, gap)

static const loom_target_coverage_row_t kLoomAmdgpuTargetCoverageRows[] = {
    LOOM_AMDGPU_GFX11_ROW("vector.arithmetic", "i32-f32-vgpr",
                          LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                          LOOM_AMDGPU_NATIVE_LOW_GFX11_PHASES,
                          "validation-run-provider-optional"),
    LOOM_AMDGPU_GFX12_ROW("vector.arithmetic", "i32-f32-vgpr",
                          LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                          LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                          "descriptor-packet-encoding-missing"),
    LOOM_AMDGPU_GFX1250_ROW("vector.arithmetic", "i32-f32-vgpr",
                            LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                            LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                            "descriptor-packet-encoding-missing"),
    LOOM_AMDGPU_GFX950_ROW("vector.arithmetic", "i32-f32-vgpr",
                           LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                           LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                           "descriptor-packet-encoding-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.scalar", "kernarg-s-load-s-buffer-resource",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_NATIVE_LOW_GFX11_PHASES, ""),
    LOOM_AMDGPU_GFX12_ROW("memory.scalar", "kernarg-s-load-s-buffer-resource",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                          "descriptor-packet-encoding-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.scalar", "kernarg-s-load-s-buffer-resource",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                            "descriptor-packet-encoding-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.scalar", "kernarg-s-load-s-buffer-resource",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                           "descriptor-packet-encoding-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.global", "buffer-b32-b64-b128-load-store",
                          LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                          LOOM_AMDGPU_NATIVE_LOW_GFX11_PHASES,
                          "validation-run-provider-optional"),
    LOOM_AMDGPU_GFX12_ROW("memory.global", "buffer-b32-b64-b128-load-store",
                          LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                          LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                          "descriptor-packet-encoding-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.global", "buffer-b32-b64-b128-load-store",
                            LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                            LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                            "descriptor-packet-encoding-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.global", "buffer-b32-b64-b128-load-store",
                           LOOM_AMDGPU_EXPECTED_EXECUTABLE_LOW_PHASES,
                           LOOM_AMDGPU_NATIVE_LOW_TEXT_PHASES,
                           "descriptor-packet-encoding-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.global", "global-flat-scratch-generic",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-global-flat-scratch-missing"),
    LOOM_AMDGPU_GFX12_ROW("memory.global", "global-flat-scratch-generic",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-global-flat-scratch-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.global", "global-flat-scratch-generic",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_TARGET_COVERAGE_PHASE_NONE,
                            "descriptor-global-flat-scratch-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.global", "global-flat-scratch-generic",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_TARGET_COVERAGE_PHASE_NONE,
                           "descriptor-global-flat-scratch-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.workgroup", "ds-b32-b128-load-store",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_AUTHORED_LOW_GFX11_PHASES,
                          "source-lower-workgroup-memory-missing"),
    LOOM_AMDGPU_GFX12_ROW("memory.workgroup", "ds-b32-b128-load-store",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                          "source-lower-binary-encode-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.workgroup", "ds-b32-b128-load-store",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                            "source-lower-binary-encode-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.workgroup", "ds-b32-b128-load-store",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                           "source-lower-binary-encode-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.workgroup", "ds-b64-b96-read-write",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_AUTHORED_LOW_GFX11_PHASES,
                          "source-lower-workgroup-memory-missing"),
    LOOM_AMDGPU_GFX12_ROW("memory.workgroup", "ds-b64-b96-read-write",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                          "source-lower-binary-encode-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.workgroup", "ds-b64-b96-read-write",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                            "source-lower-binary-encode-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.workgroup", "ds-b64-b96-read-write",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                           "source-lower-binary-encode-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.workgroup", "ds-read2-write2-stride64-addtid",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-workgroup-addressing-missing"),
    LOOM_AMDGPU_GFX12_ROW("memory.workgroup", "ds-read2-write2-stride64-addtid",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-workgroup-addressing-missing"),
    LOOM_AMDGPU_GFX1250_ROW(
        "memory.workgroup", "ds-read2-write2-stride64-addtid",
        LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES, LOOM_TARGET_COVERAGE_PHASE_NONE,
        "descriptor-workgroup-addressing-missing"),
    LOOM_AMDGPU_GFX950_ROW(
        "memory.workgroup", "ds-read2-write2-stride64-addtid",
        LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES, LOOM_TARGET_COVERAGE_PHASE_NONE,
        "descriptor-workgroup-addressing-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.crosslane", "ds-permute-bpermute-swizzle",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-crosslane-memory-missing"),
    LOOM_AMDGPU_GFX12_ROW("memory.crosslane", "ds-permute-bpermute-swizzle",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-crosslane-memory-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.crosslane", "ds-permute-bpermute-swizzle",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_TARGET_COVERAGE_PHASE_NONE,
                            "descriptor-crosslane-memory-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.crosslane", "ds-permute-bpermute-swizzle",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_TARGET_COVERAGE_PHASE_NONE,
                           "descriptor-crosslane-memory-missing"),

    LOOM_AMDGPU_GFX950_ROW("memory.async", "global-load-lds-async-tdm",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_TARGET_COVERAGE_PHASE_NONE,
                           "descriptor-async-global-to-lds-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.transpose", "ds-read-tr",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_TARGET_COVERAGE_PHASE_NONE,
                           "descriptor-transpose-load-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.cache", "cache-swizzle-prefetch-invalidate",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-cache-policy-missing"),
    LOOM_AMDGPU_GFX12_ROW("memory.cache", "cache-swizzle-prefetch-invalidate",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-cache-policy-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.cache", "cache-swizzle-prefetch-invalidate",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_TARGET_COVERAGE_PHASE_NONE,
                            "descriptor-cache-policy-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.cache", "cache-swizzle-prefetch-invalidate",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_TARGET_COVERAGE_PHASE_NONE,
                           "descriptor-cache-policy-missing"),

    LOOM_AMDGPU_GFX11_ROW("memory.atomic", "buffer-global-flat-ds-atomics",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-atomics-missing"),
    LOOM_AMDGPU_GFX12_ROW("memory.atomic", "buffer-global-flat-ds-atomics",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_TARGET_COVERAGE_PHASE_NONE,
                          "descriptor-atomics-missing"),
    LOOM_AMDGPU_GFX1250_ROW("memory.atomic", "buffer-global-flat-ds-atomics",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_TARGET_COVERAGE_PHASE_NONE,
                            "descriptor-atomics-missing"),
    LOOM_AMDGPU_GFX950_ROW("memory.atomic", "buffer-global-flat-ds-atomics",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_TARGET_COVERAGE_PHASE_NONE,
                           "descriptor-atomics-missing"),

    LOOM_AMDGPU_GFX11_ROW(
        "synchronization", "combined-waitcnt-depctr-workgroup-barrier",
        LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
        LOOM_AMDGPU_AUTHORED_LOW_GFX11_PHASES, "source-lower-barrier-missing"),
    LOOM_AMDGPU_GFX12_ROW("synchronization",
                          "split-loadcnt-storecnt-alu-workgroup-barrier",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                          "source-lower-binary-encode-missing"),
    LOOM_AMDGPU_GFX1250_ROW("synchronization",
                            "split-loadcnt-storecnt-alu-workgroup-barrier",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                            "source-lower-binary-encode-missing"),
    LOOM_AMDGPU_GFX950_ROW("synchronization",
                           "combined-waitcnt-workgroup-barrier",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_AMDGPU_AUTHORED_LOW_TEXT_PHASES,
                           "source-lower-binary-encode-missing"),

    LOOM_AMDGPU_GFX11_ROW("matrix.contract", "wmma-mfma",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES,
                          "source-lower-matrix-encoding-missing"),
    LOOM_AMDGPU_GFX12_ROW("matrix.contract", "wmma",
                          LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                          LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES,
                          "source-lower-matrix-encoding-missing"),
    LOOM_AMDGPU_GFX1250_ROW("matrix.contract", "wmma-swmmac-scale",
                            LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                            LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES,
                            "source-lower-matrix-encoding-missing"),
    LOOM_AMDGPU_GFX950_ROW("matrix.contract", "mfma",
                           LOOM_AMDGPU_EXPECTED_NATIVE_LOW_PHASES,
                           LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES,
                           "source-lower-matrix-encoding-missing"),
    {
        .target_key = IREE_SVL("amdgpu"),
        .descriptor_set_key = IREE_SVL("amdgpu.gfx11.core"),
        .category = IREE_SVL("artifact"),
        .semantic_tag = IREE_SVL("hsaco"),
        .expected_phases = LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |
                           LOOM_TARGET_COVERAGE_PHASE_INSPECT |
                           LOOM_TARGET_COVERAGE_PHASE_RUN,
        .supported_phases = LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP,
        .gap_key = IREE_SVL("validation-run-provider-optional"),
    },
};

const loom_target_coverage_provider_t loom_amdgpu_target_coverage_provider = {
    .name = IREE_SVL("amdgpu"),
    .rows = kLoomAmdgpuTargetCoverageRows,
    .row_count = IREE_ARRAYSIZE(kLoomAmdgpuTargetCoverageRows),
};
