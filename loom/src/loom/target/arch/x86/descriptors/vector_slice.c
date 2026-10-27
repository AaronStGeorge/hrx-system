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

#define LOOM_X86_VECTOR_SLICE_AVX2(prefix)                                     \
  [prefix##_DESCRIPTOR_SET_ORDINAL] =                                          \
      &(const loom_x86_vector_slice_descriptors_t) {                           \
    .maximum_vector_bit_width = 256,                                           \
    .shift_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPSRLDQ_XMM,                     \
    .align_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPALIGNR_XMM,                    \
    .extract_xmm_from_ymm = prefix##_DESCRIPTOR_REF_AVX2_VEXTRACTF128_XMM_YMM, \
    .insert_xmm_into_ymm = prefix##_DESCRIPTOR_REF_AVX2_VINSERTF128_YMM_XMM,   \
    .align_dword_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,                       \
    .align_qword_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,                       \
    .extract_xmm_from_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,                  \
    .extract_ymm_from_zmm = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,                  \
  }

#define LOOM_X86_VECTOR_SLICE_AVX512(prefix)                                   \
  [prefix##_DESCRIPTOR_SET_ORDINAL] =                                          \
      &(const loom_x86_vector_slice_descriptors_t) {                           \
    .maximum_vector_bit_width = 512,                                           \
    .shift_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPSRLDQ_XMM,                     \
    .align_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPALIGNR_XMM,                    \
    .extract_xmm_from_ymm = prefix##_DESCRIPTOR_REF_AVX2_VEXTRACTF128_XMM_YMM, \
    .insert_xmm_into_ymm = prefix##_DESCRIPTOR_REF_AVX2_VINSERTF128_YMM_XMM,   \
    .align_dword_zmm = prefix##_DESCRIPTOR_REF_AVX512_VALIGND_ZMM,             \
    .align_qword_zmm = prefix##_DESCRIPTOR_REF_AVX512_VALIGNQ_ZMM,             \
    .extract_xmm_from_zmm =                                                    \
        prefix##_DESCRIPTOR_REF_AVX512_VEXTRACTF32X4_XMM_ZMM,                  \
    .extract_ymm_from_zmm =                                                    \
        prefix##_DESCRIPTOR_REF_AVX512_VEXTRACTI64X4_YMM_ZMM,                  \
  }

static const loom_x86_vector_slice_descriptors_t* const
    kVectorSliceDescriptors[] = {
        LOOM_X86_VECTOR_SLICE_AVX2(X86_AVX2_CORE),
        LOOM_X86_VECTOR_SLICE_AVX2(X86_AVX2_FEATURES_CORE),
        LOOM_X86_VECTOR_SLICE_AVX512(X86_AVX512_CORE),
        LOOM_X86_VECTOR_SLICE_AVX512(X86_AVX512_FEATURES_CORE),
};
#undef LOOM_X86_VECTOR_SLICE_AVX2
#undef LOOM_X86_VECTOR_SLICE_AVX512

const loom_x86_vector_slice_descriptors_t* loom_x86_vector_slice_descriptors(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set->descriptor_set_ordinal <
                 IREE_ARRAYSIZE(kVectorSliceDescriptors)
             ? kVectorSliceDescriptors[descriptor_set->descriptor_set_ordinal]
             : NULL;
}
