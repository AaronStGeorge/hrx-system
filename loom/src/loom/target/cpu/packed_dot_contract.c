// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/cpu/packed_dot_contract.h"

#define PACKED_DOT_SHAPE(vector_bits_value, input_lanes_value, \
                         result_lanes_value, group_size_value) \
  (loom_cpu_packed_dot_shape_t) {                              \
    .vector_bit_width = (vector_bits_value),                   \
    .input_lane_count = (input_lanes_value),                   \
    .result_lane_count = (result_lanes_value),                 \
    .reduction_group_size = (group_size_value),                \
  }

#define PACKED_DOT_DESCRIPTOR(                                             \
    name_value, intrinsic_name_value, mnemonic_value, family_value,        \
    feature_bits_value, flags_value, vector_bits_value, input_lanes_value, \
    result_lanes_value, group_size_value, lhs_type_value, rhs_type_value,  \
    accumulator_type_value, result_type_value)                             \
  {                                                                        \
      .name = IREE_SVL(name_value),                                        \
      .llvm_intrinsic_name = IREE_SVL(intrinsic_name_value),               \
      .instruction_mnemonic = IREE_SVL(mnemonic_value),                    \
      .family = (family_value),                                            \
      .required_feature_bits = (feature_bits_value),                       \
      .flags = (flags_value),                                              \
      .shape = PACKED_DOT_SHAPE(vector_bits_value, input_lanes_value,      \
                                result_lanes_value, group_size_value),     \
      .lhs_numeric_type = (lhs_type_value),                                \
      .rhs_numeric_type = (rhs_type_value),                                \
      .accumulator_numeric_type = (accumulator_type_value),                \
      .result_numeric_type = (result_type_value),                          \
  }

#define AVX512_VNNI_FEATURES_512 (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_VNNI)
#define AVX512_VNNI_FEATURES_128_256             \
  (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_VNNI | \
   LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_VL)
#define AVX512_BF16_FEATURES_512 (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_BF16)
#define AVX512_BF16_FEATURES_128_256             \
  (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_BF16 | \
   LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_VL)
#define AVX_VNNI_FEATURES (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI)
#define AVX_VNNI_INT8_FEATURES (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI_INT8)
#define AVX_VNNI_INT16_FEATURES (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI_INT16)
#define AVX10_2_FEATURES (LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX10_2)
#define SATURATING_FLAGS (LOOM_CPU_PACKED_DOT_CONTRACT_FLAG_SATURATING)

static const loom_cpu_packed_dot_descriptor_t kPackedDotDescriptors[] = {
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpbusd.128", "llvm.x86.avx512.vpdpbusd.128",
        "vpdpbusd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, 0, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpbusd.256", "llvm.x86.avx512.vpdpbusd.256",
        "vpdpbusd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, 0, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpbusd.512", "llvm.x86.avx512.vpdpbusd.512",
        "vpdpbusd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_512, 0, 512, 64, 16, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpbusds.128", "llvm.x86.avx512.vpdpbusds.128",
        "vpdpbusds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, SATURATING_FLAGS, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpbusds.256", "llvm.x86.avx512.vpdpbusds.256",
        "vpdpbusds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, SATURATING_FLAGS, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpbusds.512", "llvm.x86.avx512.vpdpbusds.512",
        "vpdpbusds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_512, SATURATING_FLAGS, 512, 64, 16, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpwssd.128", "llvm.x86.avx512.vpdpwssd.128",
        "vpdpwssd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, 0, 128, 8, 4, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpwssd.256", "llvm.x86.avx512.vpdpwssd.256",
        "vpdpwssd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, 0, 256, 16, 8, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpwssd.512", "llvm.x86.avx512.vpdpwssd.512",
        "vpdpwssd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_512, 0, 512, 32, 16, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpwssds.128", "llvm.x86.avx512.vpdpwssds.128",
        "vpdpwssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, SATURATING_FLAGS, 128, 8, 4, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpwssds.256", "llvm.x86.avx512.vpdpwssds.256",
        "vpdpwssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_128_256, SATURATING_FLAGS, 256, 16, 8, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-vnni.vpdpwssds.512", "llvm.x86.avx512.vpdpwssds.512",
        "vpdpwssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
        AVX512_VNNI_FEATURES_512, SATURATING_FLAGS, 512, 32, 16, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),

    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpbusd.128", "llvm.x86.avx512.vpdpbusd.128", "vpdpbusd",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES, 0, 128, 16,
        4, 4, LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpbusd.256", "llvm.x86.avx512.vpdpbusd.256", "vpdpbusd",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES, 0, 256, 32,
        8, 4, LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpbusds.128", "llvm.x86.avx512.vpdpbusds.128",
        "vpdpbusds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES,
        SATURATING_FLAGS, 128, 16, 4, 4, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I32,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpbusds.256", "llvm.x86.avx512.vpdpbusds.256",
        "vpdpbusds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES,
        SATURATING_FLAGS, 256, 32, 8, 4, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I32,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpwssd.128", "llvm.x86.avx512.vpdpwssd.128", "vpdpwssd",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES, 0, 128, 8,
        4, 2, LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpwssd.256", "llvm.x86.avx512.vpdpwssd.256", "vpdpwssd",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES, 0, 256, 16,
        8, 2, LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpwssds.128", "llvm.x86.avx512.vpdpwssds.128",
        "vpdpwssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES,
        SATURATING_FLAGS, 128, 8, 4, 2, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I32,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni.vpdpwssds.256", "llvm.x86.avx512.vpdpwssds.256",
        "vpdpwssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI, AVX_VNNI_FEATURES,
        SATURATING_FLAGS, 256, 16, 8, 2, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_I32,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32),

    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbssd.128", "llvm.x86.avx2.vpdpbssd.128",
        "vpdpbssd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, 0, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbssd.256", "llvm.x86.avx2.vpdpbssd.256",
        "vpdpbssd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, 0, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbssds.128", "llvm.x86.avx2.vpdpbssds.128",
        "vpdpbssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, SATURATING_FLAGS, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbssds.256", "llvm.x86.avx2.vpdpbssds.256",
        "vpdpbssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, SATURATING_FLAGS, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbsud.128", "llvm.x86.avx2.vpdpbsud.128",
        "vpdpbsud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, 0, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbsud.256", "llvm.x86.avx2.vpdpbsud.256",
        "vpdpbsud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, 0, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbsuds.128", "llvm.x86.avx2.vpdpbsuds.128",
        "vpdpbsuds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, SATURATING_FLAGS, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbsuds.256", "llvm.x86.avx2.vpdpbsuds.256",
        "vpdpbsuds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, SATURATING_FLAGS, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbuud.128", "llvm.x86.avx2.vpdpbuud.128",
        "vpdpbuud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, 0, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbuud.256", "llvm.x86.avx2.vpdpbuud.256",
        "vpdpbuud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, 0, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbuuds.128", "llvm.x86.avx2.vpdpbuuds.128",
        "vpdpbuuds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, SATURATING_FLAGS, 128, 16, 4, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int8.vpdpbuuds.256", "llvm.x86.avx2.vpdpbuuds.256",
        "vpdpbuuds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
        AVX_VNNI_INT8_FEATURES, SATURATING_FLAGS, 256, 32, 8, 4,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),

    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int16.vpdpwsud.128", "llvm.x86.avx2.vpdpwsud.128",
        "vpdpwsud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16,
        AVX_VNNI_INT16_FEATURES, 0, 128, 8, 4, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_U16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int16.vpdpwsud.256", "llvm.x86.avx2.vpdpwsud.256",
        "vpdpwsud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16,
        AVX_VNNI_INT16_FEATURES, 0, 256, 16, 8, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_U16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int16.vpdpwusd.128", "llvm.x86.avx2.vpdpwusd.128",
        "vpdpwusd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16,
        AVX_VNNI_INT16_FEATURES, 0, 128, 8, 4, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_U16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int16.vpdpwusd.256", "llvm.x86.avx2.vpdpwusd.256",
        "vpdpwusd", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16,
        AVX_VNNI_INT16_FEATURES, 0, 256, 16, 8, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_U16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int16.vpdpwuud.128", "llvm.x86.avx2.vpdpwuud.128",
        "vpdpwuud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16,
        AVX_VNNI_INT16_FEATURES, 0, 128, 8, 4, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_U16, LOOM_CPU_PACKED_DOT_NUMERIC_U16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx-vnni-int16.vpdpwuud.256", "llvm.x86.avx2.vpdpwuud.256",
        "vpdpwuud", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16,
        AVX_VNNI_INT16_FEATURES, 0, 256, 16, 8, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_U16, LOOM_CPU_PACKED_DOT_NUMERIC_U16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),

    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpbssd.512", "llvm.x86.avx10.vpdpbssd.512", "vpdpbssd",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 512, 64,
        16, 4, LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpbsud.512", "llvm.x86.avx10.vpdpbsud.512", "vpdpbsud",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 512, 64,
        16, 4, LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpbuud.512", "llvm.x86.avx10.vpdpbuud.512", "vpdpbuud",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 512, 64,
        16, 4, LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpbssds.512", "llvm.x86.avx10.vpdpbssds.512",
        "vpdpbssds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES,
        SATURATING_FLAGS, 512, 64, 16, 4, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_I8, LOOM_CPU_PACKED_DOT_NUMERIC_I32,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpbsuds.512", "llvm.x86.avx10.vpdpbsuds.512",
        "vpdpbsuds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES,
        SATURATING_FLAGS, 512, 64, 16, 4, LOOM_CPU_PACKED_DOT_NUMERIC_I8,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I32,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpbuuds.512", "llvm.x86.avx10.vpdpbuuds.512",
        "vpdpbuuds", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES,
        SATURATING_FLAGS, 512, 64, 16, 4, LOOM_CPU_PACKED_DOT_NUMERIC_U8,
        LOOM_CPU_PACKED_DOT_NUMERIC_U8, LOOM_CPU_PACKED_DOT_NUMERIC_I32,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpwsud.512", "llvm.x86.avx10.vpdpwsud.512", "vpdpwsud",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 512, 32,
        16, 2, LOOM_CPU_PACKED_DOT_NUMERIC_I16, LOOM_CPU_PACKED_DOT_NUMERIC_U16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpwusd.512", "llvm.x86.avx10.vpdpwusd.512", "vpdpwusd",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 512, 32,
        16, 2, LOOM_CPU_PACKED_DOT_NUMERIC_U16, LOOM_CPU_PACKED_DOT_NUMERIC_I16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vpdpwuud.512", "llvm.x86.avx10.vpdpwuud.512", "vpdpwuud",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 512, 32,
        16, 2, LOOM_CPU_PACKED_DOT_NUMERIC_U16, LOOM_CPU_PACKED_DOT_NUMERIC_U16,
        LOOM_CPU_PACKED_DOT_NUMERIC_I32, LOOM_CPU_PACKED_DOT_NUMERIC_I32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vdpphps.128", "llvm.x86.avx10.vdpphps.128", "vdpphps",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 128, 8, 4,
        2, LOOM_CPU_PACKED_DOT_NUMERIC_F16, LOOM_CPU_PACKED_DOT_NUMERIC_F16,
        LOOM_CPU_PACKED_DOT_NUMERIC_F32, LOOM_CPU_PACKED_DOT_NUMERIC_F32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vdpphps.256", "llvm.x86.avx10.vdpphps.256", "vdpphps",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 256, 16, 8,
        2, LOOM_CPU_PACKED_DOT_NUMERIC_F16, LOOM_CPU_PACKED_DOT_NUMERIC_F16,
        LOOM_CPU_PACKED_DOT_NUMERIC_F32, LOOM_CPU_PACKED_DOT_NUMERIC_F32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx10.2.vdpphps.512", "llvm.x86.avx10.vdpphps.512", "vdpphps",
        LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2, AVX10_2_FEATURES, 0, 512, 32,
        16, 2, LOOM_CPU_PACKED_DOT_NUMERIC_F16, LOOM_CPU_PACKED_DOT_NUMERIC_F16,
        LOOM_CPU_PACKED_DOT_NUMERIC_F32, LOOM_CPU_PACKED_DOT_NUMERIC_F32),

    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-bf16.vdpbf16ps.128", "llvm.x86.avx512bf16.dpbf16ps.128",
        "vdpbf16ps", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_BF16,
        AVX512_BF16_FEATURES_128_256, 0, 128, 8, 4, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_BF16, LOOM_CPU_PACKED_DOT_NUMERIC_BF16,
        LOOM_CPU_PACKED_DOT_NUMERIC_F32, LOOM_CPU_PACKED_DOT_NUMERIC_F32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-bf16.vdpbf16ps.256", "llvm.x86.avx512bf16.dpbf16ps.256",
        "vdpbf16ps", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_BF16,
        AVX512_BF16_FEATURES_128_256, 0, 256, 16, 8, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_BF16, LOOM_CPU_PACKED_DOT_NUMERIC_BF16,
        LOOM_CPU_PACKED_DOT_NUMERIC_F32, LOOM_CPU_PACKED_DOT_NUMERIC_F32),
    PACKED_DOT_DESCRIPTOR(
        "x86.avx512-bf16.vdpbf16ps.512", "llvm.x86.avx512bf16.dpbf16ps.512",
        "vdpbf16ps", LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_BF16,
        AVX512_BF16_FEATURES_512, 0, 512, 32, 16, 2,
        LOOM_CPU_PACKED_DOT_NUMERIC_BF16, LOOM_CPU_PACKED_DOT_NUMERIC_BF16,
        LOOM_CPU_PACKED_DOT_NUMERIC_F32, LOOM_CPU_PACKED_DOT_NUMERIC_F32),
};

iree_string_view_t loom_cpu_packed_dot_family_name(
    loom_cpu_packed_dot_family_t family) {
  switch (family) {
    case LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI:
      return IREE_SV("x86-avx512-vnni");
    case LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI:
      return IREE_SV("x86-avx-vnni");
    case LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8:
      return IREE_SV("x86-avx-vnni-int8");
    case LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16:
      return IREE_SV("x86-avx-vnni-int16");
    case LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2:
      return IREE_SV("x86-avx10.2");
    case LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_BF16:
      return IREE_SV("x86-avx512-bf16");
    case LOOM_CPU_PACKED_DOT_FAMILY_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_cpu_packed_dot_numeric_type_name(
    loom_cpu_packed_dot_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_CPU_PACKED_DOT_NUMERIC_I8:
      return IREE_SV("i8");
    case LOOM_CPU_PACKED_DOT_NUMERIC_U8:
      return IREE_SV("u8");
    case LOOM_CPU_PACKED_DOT_NUMERIC_I16:
      return IREE_SV("i16");
    case LOOM_CPU_PACKED_DOT_NUMERIC_U16:
      return IREE_SV("u16");
    case LOOM_CPU_PACKED_DOT_NUMERIC_F16:
      return IREE_SV("f16");
    case LOOM_CPU_PACKED_DOT_NUMERIC_BF16:
      return IREE_SV("bf16");
    case LOOM_CPU_PACKED_DOT_NUMERIC_I32:
      return IREE_SV("i32");
    case LOOM_CPU_PACKED_DOT_NUMERIC_F32:
      return IREE_SV("f32");
    case LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_cpu_packed_dot_feature_bits_for_name(
    iree_string_view_t name,
    loom_cpu_packed_dot_feature_bits_t* out_feature_bits) {
  if (out_feature_bits == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_feature_bits must not be NULL");
  }
  *out_feature_bits = 0;
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-vnni"))) {
    *out_feature_bits = LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_VNNI;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-vnni-vl"))) {
    *out_feature_bits = AVX512_VNNI_FEATURES_128_256;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx-vnni"))) {
    *out_feature_bits = LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx-vnni-int8"))) {
    *out_feature_bits = LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI_INT8;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx-vnni-int16"))) {
    *out_feature_bits = LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI_INT16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx10.2"))) {
    *out_feature_bits = LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX10_2;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-bf16"))) {
    *out_feature_bits = LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_BF16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-bf16-vl"))) {
    *out_feature_bits = AVX512_BF16_FEATURES_128_256;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown CPU packed-dot feature set '%.*s'",
                          (int)name.size, name.data);
}

iree_host_size_t loom_cpu_packed_dot_descriptor_count(void) {
  return IREE_ARRAYSIZE(kPackedDotDescriptors);
}

const loom_cpu_packed_dot_descriptor_t* loom_cpu_packed_dot_descriptor_at(
    iree_host_size_t index) {
  if (index >= IREE_ARRAYSIZE(kPackedDotDescriptors)) return NULL;
  return &kPackedDotDescriptors[index];
}

const loom_cpu_packed_dot_descriptor_t* loom_cpu_packed_dot_find_by_name(
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kPackedDotDescriptors); ++i) {
    const loom_cpu_packed_dot_descriptor_t* descriptor =
        &kPackedDotDescriptors[i];
    if (iree_string_view_equal(name, descriptor->name)) return descriptor;
  }
  return NULL;
}

bool loom_cpu_packed_dot_is_available(
    const loom_cpu_packed_dot_descriptor_t* descriptor,
    loom_cpu_packed_dot_feature_bits_t feature_bits) {
  if (descriptor == NULL) return false;
  return (feature_bits & descriptor->required_feature_bits) ==
         descriptor->required_feature_bits;
}

static bool loom_cpu_packed_dot_family_matches(
    const loom_cpu_packed_dot_descriptor_t* descriptor,
    const loom_cpu_packed_dot_match_request_t* request) {
  return request->family == LOOM_CPU_PACKED_DOT_FAMILY_UNKNOWN ||
         descriptor->family == request->family;
}

static bool loom_cpu_packed_dot_shape_matches(
    const loom_cpu_packed_dot_descriptor_t* descriptor,
    const loom_cpu_packed_dot_match_request_t* request) {
  return descriptor->shape.vector_bit_width ==
             request->shape.vector_bit_width &&
         descriptor->shape.input_lane_count ==
             request->shape.input_lane_count &&
         descriptor->shape.result_lane_count ==
             request->shape.result_lane_count &&
         descriptor->shape.reduction_group_size ==
             request->shape.reduction_group_size;
}

static loom_cpu_packed_dot_rejection_bits_t
loom_cpu_packed_dot_payload_rejection_bits(
    const loom_cpu_packed_dot_descriptor_t* descriptor,
    const loom_cpu_packed_dot_match_request_t* request) {
  loom_cpu_packed_dot_rejection_bits_t rejection_bits =
      LOOM_CPU_PACKED_DOT_REJECTION_NONE;
  if (request->lhs_numeric_type == LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->lhs_numeric_type != request->lhs_numeric_type) {
    rejection_bits |= LOOM_CPU_PACKED_DOT_REJECTION_PAYLOAD;
  }
  if (request->rhs_numeric_type == LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->rhs_numeric_type != request->rhs_numeric_type) {
    rejection_bits |= LOOM_CPU_PACKED_DOT_REJECTION_PAYLOAD;
  }
  if (request->accumulator_numeric_type ==
          LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->accumulator_numeric_type !=
          request->accumulator_numeric_type) {
    rejection_bits |= LOOM_CPU_PACKED_DOT_REJECTION_PAYLOAD;
  }
  if (request->result_numeric_type == LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->result_numeric_type != request->result_numeric_type) {
    rejection_bits |= LOOM_CPU_PACKED_DOT_REJECTION_PAYLOAD;
  }
  return rejection_bits;
}

static bool loom_cpu_packed_dot_flags_match(
    const loom_cpu_packed_dot_descriptor_t* descriptor,
    const loom_cpu_packed_dot_match_request_t* request) {
  return (request->required_flags & ~descriptor->flags) == 0;
}

const loom_cpu_packed_dot_descriptor_t* loom_cpu_packed_dot_select(
    const loom_cpu_packed_dot_match_request_t* request,
    loom_cpu_packed_dot_match_diagnostic_t* out_diagnostic) {
  loom_cpu_packed_dot_match_diagnostic_t diagnostic = {
      .descriptor_count = IREE_ARRAYSIZE(kPackedDotDescriptors),
  };
  if (request == NULL) {
    diagnostic.rejection_bits = LOOM_CPU_PACKED_DOT_REJECTION_INVALID_REQUEST;
    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return NULL;
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kPackedDotDescriptors); ++i) {
    const loom_cpu_packed_dot_descriptor_t* descriptor =
        &kPackedDotDescriptors[i];
    if (!loom_cpu_packed_dot_family_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.family_candidate_count;

    if (!loom_cpu_packed_dot_shape_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.shape_candidate_count;

    if (loom_cpu_packed_dot_payload_rejection_bits(descriptor, request) !=
        LOOM_CPU_PACKED_DOT_REJECTION_NONE) {
      continue;
    }
    ++diagnostic.payload_candidate_count;

    if (!loom_cpu_packed_dot_flags_match(descriptor, request)) {
      continue;
    }
    ++diagnostic.flag_candidate_count;

    if (!loom_cpu_packed_dot_is_available(descriptor, request->feature_bits)) {
      continue;
    }
    ++diagnostic.feature_candidate_count;

    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return descriptor;
  }

  if (diagnostic.family_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_CPU_PACKED_DOT_REJECTION_FAMILY;
  } else if (diagnostic.shape_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_CPU_PACKED_DOT_REJECTION_SHAPE;
  } else if (diagnostic.payload_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_CPU_PACKED_DOT_REJECTION_PAYLOAD;
  } else if (diagnostic.flag_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_CPU_PACKED_DOT_REJECTION_FLAGS;
  } else if (diagnostic.feature_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_CPU_PACKED_DOT_REJECTION_FEATURES;
  }
  if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
  return NULL;
}

#undef PACKED_DOT_DESCRIPTOR
#undef PACKED_DOT_SHAPE
#undef AVX512_VNNI_FEATURES_512
#undef AVX512_VNNI_FEATURES_128_256
#undef AVX512_BF16_FEATURES_512
#undef AVX512_BF16_FEATURES_128_256
#undef AVX_VNNI_FEATURES
#undef AVX_VNNI_INT8_FEATURES
#undef AVX_VNNI_INT16_FEATURES
#undef AVX10_2_FEATURES
#undef SATURATING_FLAGS
