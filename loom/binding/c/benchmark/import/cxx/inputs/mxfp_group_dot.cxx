// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/encoding.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>
#include <loomcxx/vector.h>

#include <stdfloat>

using Words4 = unsigned __attribute__((ext_vector_type(4)));
using Words8 = unsigned __attribute__((ext_vector_type(8)));
using ScaleWord = unsigned __attribute__((ext_vector_type(1)));
using Fp8x32 = loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(32)));
using Fp8x64 = loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(64)));
using BFloat32 = std::bfloat16_t __attribute__((ext_vector_type(32)));
using BFloat64 = std::bfloat16_t __attribute__((ext_vector_type(64)));
using Float16 = float __attribute__((ext_vector_type(16)));
using Bits32 = unsigned __attribute__((ext_vector_type(32)));

struct Scales {
  // Four packed E8M0 bytes; each group consumes its corresponding byte.
  ScaleWord scale;
};

float mxfp4_group_dot(Words4 payload, ScaleWord scale, BFloat32 activations) {
  auto schema = loom::encoding::define<loom::encoding::f4e2m1{}>();
  auto weights = loom::vector::decode<BFloat32>(payload, schema, Scales{scale});
  Float16 partial = {};
  partial = loom::vector::dot2f(weights, activations, partial);
  return loom::vector::reduce::addf(partial, 0.0f);
}

float mxfp8_group_dot(Fp8x32 payload, ScaleWord scale, BFloat32 activations) {
  auto schema = loom::encoding::define<loom::encoding::f8e4m3fn{}>();
  auto weights = loom::vector::decode<BFloat32>(payload, schema, Scales{scale});
  Float16 partial = {};
  partial = loom::vector::dot2f(weights, activations, partial);
  return loom::vector::reduce::addf(partial, 0.0f);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp4_decode_dot(
    [[loom::noalias, loom::assume_aligned(64)]] const Words4* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat32* activations,
    [[loom::noalias]] float* output) {
  *output = mxfp4_group_dot(*payload, *scale, *activations);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp8_decode_dot(
    [[loom::noalias, loom::assume_aligned(64)]] const Fp8x32* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat32* activations,
    [[loom::noalias]] float* output) {
  *output = mxfp8_group_dot(*payload, *scale, *activations);
}

// Two groups in one vector retain their separate scale bytes through decoding.
[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp4_group_bits(
    [[loom::noalias, loom::assume_aligned(64)]] const Words8* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] Bits32* output) {
  auto schema =
      loom::encoding::define<loom::encoding::f4e2m1{.payload_elements = 64}>();
  auto weights =
      loom::vector::decode<BFloat64>(*payload, schema, Scales{*scale});
  *output = __builtin_bit_cast(Bits32, weights);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void mxfp8_group_bits(
    [[loom::noalias, loom::assume_aligned(64)]] const Fp8x64* payload,
    [[loom::noalias, loom::assume_aligned(64)]] const ScaleWord* scale,
    [[loom::noalias, loom::assume_aligned(64)]] Bits32* output) {
  auto schema = loom::encoding::define<loom::encoding::f8e4m3fn{
      .payload_elements = 64,
  }>();
  auto weights =
      loom::vector::decode<BFloat64>(*payload, schema, Scales{*scale});
  *output = __builtin_bit_cast(Bits32, weights);
}
