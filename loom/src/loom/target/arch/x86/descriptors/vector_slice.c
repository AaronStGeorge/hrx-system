// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/descriptors/vector_slice.h"

#include "loom/target/arch/x86/descriptors/avx2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx2_features_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_features_descriptors.h"

#define LOOM_X86_VECTOR_SLICE_SIMD_CONVERSION(prefix)           \
  {                                                             \
      .widen =                                                  \
          {                                                     \
              {prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBW_XMM_XMM,  \
               prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBW_YMM_XMM}, \
              {prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBD_XMM_XMM,  \
               prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBD_YMM_XMM}, \
              {prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBQ_XMM_XMM,  \
               prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBQ_YMM_XMM}, \
              {prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXWD_XMM_XMM,  \
               prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXWD_YMM_XMM}, \
              {prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXWQ_XMM_XMM,  \
               prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXWQ_YMM_XMM}, \
              {prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXDQ_XMM_XMM,  \
               prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXDQ_YMM_XMM}, \
          },                                                    \
      .pack =                                                   \
          {                                                     \
              prefix##_DESCRIPTOR_REF_AVX2_VSHUFPS_XMM,         \
              prefix##_DESCRIPTOR_REF_AVX2_VPACKSSDW_XMM,       \
              prefix##_DESCRIPTOR_REF_AVX2_VPACKSSWB_XMM,       \
          },                                                    \
  }

#define LOOM_X86_VECTOR_SLICE_NONE_CLASSES                             \
  {LOOM_LOW_DESCRIPTOR_ORDINAL_NONE, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE, \
   LOOM_LOW_DESCRIPTOR_ORDINAL_NONE}

static const loom_x86_vector_slice_descriptors_t kAvx2VectorSliceDescriptors = {
    .maximum_vector_bit_width = 256,
    .supports_mask = false,
    .shift_xmm = X86_AVX2_CORE_DESCRIPTOR_REF_AVX2_VPSRLDQ_XMM,
    .align_xmm = X86_AVX2_CORE_DESCRIPTOR_REF_AVX2_VPALIGNR_XMM,
    .extract_xmm_from_ymm =
        X86_AVX2_CORE_DESCRIPTOR_REF_AVX2_VEXTRACTF128_XMM_YMM,
    .insert_xmm_into_ymm =
        X86_AVX2_CORE_DESCRIPTOR_REF_AVX2_VINSERTF128_YMM_XMM,
    .align_dword_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
    .align_qword_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
    .extract_xmm_from_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
    .extract_ymm_from_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
    .simd_conversion = LOOM_X86_VECTOR_SLICE_SIMD_CONVERSION(X86_AVX2_CORE),
    .mask_conversion =
        {
            .shift = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
            .compress =
                {
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                },
            .expand =
                {
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                    LOOM_X86_VECTOR_SLICE_NONE_CLASSES,
                },
            .zero_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
            .insert_xmm_into_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
            .insert_ymm_into_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
        },
};

static const loom_x86_vector_slice_descriptors_t kAvx512VectorSliceDescriptors =
    {
        .maximum_vector_bit_width = 512,
        .supports_mask = true,
        .shift_xmm = X86_AVX512_CORE_DESCRIPTOR_REF_AVX2_VPSRLDQ_XMM,
        .align_xmm = X86_AVX512_CORE_DESCRIPTOR_REF_AVX2_VPALIGNR_XMM,
        .extract_xmm_from_ymm =
            X86_AVX512_CORE_DESCRIPTOR_REF_AVX2_VEXTRACTF128_XMM_YMM,
        .insert_xmm_into_ymm =
            X86_AVX512_CORE_DESCRIPTOR_REF_AVX2_VINSERTF128_YMM_XMM,
        .align_dword_zmm = X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VALIGND_ZMM,
        .align_qword_zmm = X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VALIGNQ_ZMM,
        .extract_xmm_from_zmm =
            X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VEXTRACTF32X4_XMM_ZMM,
        .extract_ymm_from_zmm =
            X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VEXTRACTI64X4_YMM_ZMM,
        .simd_conversion =
            LOOM_X86_VECTOR_SLICE_SIMD_CONVERSION(X86_AVX512_CORE),
        .mask_conversion =
            {
                .shift = X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_KSHIFTRQ,
                .compress =
                    {
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVB2M_K_XMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVB2M_K_YMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVB2M_K_ZMM},
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVW2M_K_XMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVW2M_K_YMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVW2M_K_ZMM},
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVD2M_K_XMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVD2M_K_YMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVD2M_K_ZMM},
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVQ2M_K_XMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVQ2M_K_YMM,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVQ2M_K_ZMM},
                    },
                .expand =
                    {
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2B_XMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2B_YMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2B_ZMM_K},
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2W_XMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2W_YMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2W_ZMM_K},
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2D_XMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2D_YMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2D_ZMM_K},
                        {X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2Q_XMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2Q_YMM_K,
                         X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VPMOVM2Q_ZMM_K},
                    },
                .zero_zmm =
                    X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VXORPS_ZERO_ZMM,
                .insert_xmm_into_zmm =
                    X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VINSERTF32X4_ZMM_XMM,
                .insert_ymm_into_zmm =
                    X86_AVX512_CORE_DESCRIPTOR_REF_AVX512_VINSERTI64X4_ZMM_YMM,
            },
};

static const loom_x86_vector_slice_descriptors_t* const
    kVectorSliceDescriptors[] = {
        [X86_AVX2_CORE_DESCRIPTOR_SET_ORDINAL] = &kAvx2VectorSliceDescriptors,
        [X86_AVX2_FEATURES_CORE_DESCRIPTOR_SET_ORDINAL] =
            &kAvx2VectorSliceDescriptors,
        [X86_AVX512_CORE_DESCRIPTOR_SET_ORDINAL] =
            &kAvx512VectorSliceDescriptors,
        [X86_AVX512_FEATURES_CORE_DESCRIPTOR_SET_ORDINAL] =
            &kAvx512VectorSliceDescriptors,
};
#undef LOOM_X86_VECTOR_SLICE_NONE_CLASSES
#undef LOOM_X86_VECTOR_SLICE_SIMD_CONVERSION

const loom_x86_vector_slice_descriptors_t* loom_x86_vector_slice_descriptors(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set->descriptor_set_ordinal <
                 IREE_ARRAYSIZE(kVectorSliceDescriptors)
             ? kVectorSliceDescriptors[descriptor_set->descriptor_set_ordinal]
             : NULL;
}
