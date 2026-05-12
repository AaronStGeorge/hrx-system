// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/register_classes.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/avx512_packed_dot_descriptors.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/arch/x86/scalar_descriptors.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

void ExpectDescriptorClass(const loom_low_descriptor_set_t* descriptor_set,
                           loom_x86_register_class_t register_class) {
  uint16_t descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_ASSERT_OK(loom_x86_descriptor_set_register_class_id(
      descriptor_set, register_class, &descriptor_reg_class_id));

  loom_x86_register_class_t resolved_register_class =
      LOOM_X86_REGISTER_CLASS_GPR32;
  IREE_ASSERT_OK(loom_x86_descriptor_set_logical_register_class(
      descriptor_set, descriptor_reg_class_id, &resolved_register_class));
  EXPECT_EQ(resolved_register_class, register_class);

  iree_string_view_t expected_name = iree_string_view_empty();
  IREE_ASSERT_OK(loom_x86_register_class_name(register_class, &expected_name));
  ASSERT_LT(descriptor_reg_class_id, descriptor_set->reg_class_count);
  const loom_low_reg_class_t& descriptor_reg_class =
      descriptor_set->reg_classes[descriptor_reg_class_id];
  EXPECT_EQ(ToString(loom_low_descriptor_set_string(
                descriptor_set, descriptor_reg_class.name_string_offset)),
            ToString(expected_name));
}

TEST(X86RegisterClassesTest, SharedScalarClassesAcrossViews) {
  const loom_low_descriptor_set_t* scalar_descriptor_set =
      loom_x86_scalar_core_descriptor_set();
  const loom_low_descriptor_set_t* avx512_descriptor_set =
      loom_x86_avx512_core_descriptor_set();

  ExpectDescriptorClass(scalar_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR32);
  ExpectDescriptorClass(scalar_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR32);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
}

TEST(X86RegisterClassesTest, VectorClassesAcrossProfileViews) {
  const loom_low_descriptor_set_t* avx512_descriptor_set =
      loom_x86_avx512_core_descriptor_set();
  const loom_low_descriptor_set_t* packed_dot_descriptor_set =
      loom_x86_packed_dot_core_descriptor_set();
  const loom_low_descriptor_set_t* avx512_packed_dot_descriptor_set =
      loom_x86_avx512_packed_dot_core_descriptor_set();

  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_ZMM);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_K);

  ExpectDescriptorClass(packed_dot_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(packed_dot_descriptor_set, LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorClass(packed_dot_descriptor_set, LOOM_X86_REGISTER_CLASS_ZMM);

  ExpectDescriptorClass(avx512_packed_dot_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx512_packed_dot_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorClass(avx512_packed_dot_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_ZMM);
}

TEST(X86RegisterClassesTest, VectorWidthProjection) {
  loom_x86_register_class_t register_class = LOOM_X86_REGISTER_CLASS_GPR32;
  EXPECT_TRUE(
      loom_x86_register_class_for_vector_bit_width(128, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_XMM);
  EXPECT_TRUE(
      loom_x86_register_class_for_vector_bit_width(256, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_YMM);
  EXPECT_TRUE(
      loom_x86_register_class_for_vector_bit_width(512, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_ZMM);
  EXPECT_FALSE(
      loom_x86_register_class_for_vector_bit_width(64, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_GPR32);
}

TEST(X86RegisterClassesTest, InvalidProjectionFailsLoudly) {
  iree_string_view_t register_class_name = iree_string_view_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_x86_register_class_name((loom_x86_register_class_t)99,
                                   &register_class_name));

  loom_x86_register_class_t register_class = LOOM_X86_REGISTER_CLASS_GPR32;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_x86_descriptor_set_logical_register_class(
          loom_x86_scalar_core_descriptor_set(), UINT16_MAX, &register_class));
}

}  // namespace
}  // namespace loom
