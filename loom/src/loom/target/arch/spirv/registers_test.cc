// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/spirv/descriptors.h"
#include "loom/target/arch/spirv/target_records.h"

namespace {

const loom_low_reg_class_t* LookupRegisterClass(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t register_class_name) {
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = nullptr;
  EXPECT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set, register_class_name, &descriptor_register_class_id,
      &reg_class))
      << std::string(register_class_name.data, register_class_name.size);
  return reg_class;
}

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t descriptor_key) {
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, descriptor_key);
  EXPECT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
      << std::string(descriptor_key.data, descriptor_key.size);
  return loom_low_descriptor_set_descriptor_at(descriptor_set,
                                               descriptor_ordinal);
}

TEST(SpirvRegistersTest, LogicalIdsAndPointerValuesUseSeparateWidths) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_spirv_logical_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);

  const loom_low_reg_class_t* id_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.id"));
  ASSERT_NE(id_class, nullptr);
  EXPECT_EQ(id_class->alloc_unit_bits, 32);
  EXPECT_EQ(id_class->allocatable_count, 0);
  EXPECT_TRUE(
      iree_all_bits_set(id_class->flags, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_EQ(id_class->spill_slot_space, LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);

  const loom_low_reg_class_t* function_ptr_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.function"));
  ASSERT_NE(function_ptr_class, nullptr);
  EXPECT_EQ(function_ptr_class->alloc_unit_bits, 32);
  EXPECT_TRUE(iree_all_bits_set(function_ptr_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_EQ(function_ptr_class->spill_slot_space,
            LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);

  const loom_low_reg_class_t* workgroup_ptr_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.workgroup"));
  ASSERT_NE(workgroup_ptr_class, nullptr);
  EXPECT_EQ(workgroup_ptr_class->alloc_unit_bits, 32);
  EXPECT_TRUE(iree_all_bits_set(workgroup_ptr_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_TRUE(iree_all_bits_set(workgroup_ptr_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE));

  const loom_low_reg_class_t* storage_buffer_ptr_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.storage_buffer"));
  ASSERT_NE(storage_buffer_ptr_class, nullptr);
  EXPECT_EQ(storage_buffer_ptr_class->alloc_unit_bits, 64);
  EXPECT_TRUE(iree_all_bits_set(storage_buffer_ptr_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_EQ(storage_buffer_ptr_class->spill_slot_space,
            LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);
}

TEST(SpirvRegistersTest, VulkanTargetUsesBufferDeviceAddressWidths) {
  const loom_target_bundle_t& target = loom_spirv_low_target_bundle_vulkan1_3;
  ASSERT_NE(target.snapshot, nullptr);
  EXPECT_EQ(target.snapshot->default_pointer_bitwidth, 64u);
  EXPECT_EQ(target.snapshot->index_bitwidth, 32u);
  EXPECT_EQ(target.snapshot->offset_bitwidth, 64u);
  EXPECT_EQ(target.snapshot->memory_spaces.descriptor,
            LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER);
}

TEST(SpirvRegistersTest, StorageBufferEffectsStayDescriptorLocal) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_spirv_logical_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);

  const loom_low_descriptor_t* load_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("spirv.op_load.storage_buffer.i32"));
  ASSERT_NE(load_descriptor, nullptr);
  ASSERT_EQ(load_descriptor->effect_count, 1);
  const loom_low_effect_t& load_effect =
      descriptor_set->effects[load_descriptor->effect_start];
  EXPECT_EQ(load_effect.kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(load_effect.memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_TRUE(
      iree_all_bits_set(load_effect.flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY));

  const loom_low_descriptor_t* store_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("spirv.op_store.storage_buffer.i32"));
  ASSERT_NE(store_descriptor, nullptr);
  ASSERT_EQ(store_descriptor->effect_count, 1);
  const loom_low_effect_t& store_effect =
      descriptor_set->effects[store_descriptor->effect_start];
  EXPECT_EQ(store_effect.kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(store_effect.memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_TRUE(
      iree_all_bits_set(store_effect.flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY));
}

}  // namespace
