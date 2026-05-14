// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/cdna3_descriptors.h"
#include "loom/target/arch/amdgpu/cdna4_descriptors.h"
#include "loom/target/arch/amdgpu/occupancy_tables.h"
#include "loom/target/arch/amdgpu/rdna3_descriptors.h"
#include "loom/target/arch/amdgpu/rdna4_descriptors.h"
#include "loom/target/arch/amdgpu/rdna4_gfx125x_descriptors.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

void ExpectRegisterClass(const loom_low_descriptor_set_t* descriptor_set,
                         uint16_t reg_class_id,
                         iree_string_view_t expected_name,
                         uint16_t expected_alloc_unit_bits,
                         uint16_t expected_allocatable_count,
                         loom_low_reg_class_flags_t expected_flags) {
  EXPECT_LT(reg_class_id, descriptor_set->reg_class_count)
      << ToString(expected_name);
  if (reg_class_id >= descriptor_set->reg_class_count) {
    return;
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_id];
  EXPECT_EQ(ToString(loom_low_descriptor_set_string(
                descriptor_set, reg_class->name_string_offset)),
            ToString(expected_name));
  EXPECT_EQ(reg_class->alloc_unit_bits, expected_alloc_unit_bits)
      << ToString(expected_name);
  EXPECT_EQ(reg_class->allocatable_count, expected_allocatable_count)
      << ToString(expected_name);
  EXPECT_TRUE(iree_all_bits_set(reg_class->flags, expected_flags))
      << ToString(expected_name);
}

void ExpectRegisterClassMissing(const loom_low_descriptor_set_t* descriptor_set,
                                iree_string_view_t register_class_name) {
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = nullptr;
  EXPECT_FALSE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set, register_class_name, &descriptor_register_class_id,
      &reg_class))
      << ToString(register_class_name);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE)
      << ToString(register_class_name);
  EXPECT_EQ(reg_class, nullptr) << ToString(register_class_name);
}

void ExpectOccupancyRegisterClass(const loom_amdgpu_occupancy_model_t* model,
                                  iree_host_size_t index,
                                  iree_string_view_t expected_name,
                                  uint32_t expected_pool_units,
                                  uint32_t expected_allocation_granularity) {
  ASSERT_LT(index, model->register_class_count) << ToString(expected_name);
  const loom_amdgpu_occupancy_register_class_model_t* reg_class =
      &model->register_classes[index];
  EXPECT_EQ(ToString(reg_class->register_class), ToString(expected_name));
  EXPECT_EQ(reg_class->pool_units, expected_pool_units)
      << ToString(expected_name);
  EXPECT_EQ(reg_class->allocation_granularity, expected_allocation_granularity)
      << ToString(expected_name);
}

TEST(AmdgpuRegistersTest, DescriptorSetsExposeAddressableRegisterNamespaces) {
  struct DescriptorSetCase {
    // Generated descriptor set under test.
    const loom_low_descriptor_set_t* descriptor_set;
    // Descriptor-set-local SGPR register-class ID.
    uint16_t sgpr_id;
    // Compiler-visible SGPR allocation namespace size.
    uint16_t sgpr_allocatable_count;
    // Descriptor-set-local VGPR register-class ID.
    uint16_t vgpr_id;
    // Compiler-visible VGPR allocation namespace size.
    uint16_t vgpr_allocatable_count;
    // Descriptor-set-local SCC register-class ID.
    uint16_t scc_id;
    // Descriptor-set-local EXEC register-class ID.
    uint16_t exec_id;
    // Descriptor-set-local AGPR register-class ID, or NONE when absent.
    uint16_t agpr_id;
    // Descriptor-set-local MODE register-class ID, or NONE when absent.
    uint16_t mode_id;
  };
  const DescriptorSetCase cases[] = {
      {
          loom_amdgpu_cdna3_core_descriptor_set(),
          AMDGPU_CDNA3_CORE_REG_CLASS_ID_SGPR,
          102,
          AMDGPU_CDNA3_CORE_REG_CLASS_ID_VGPR,
          256,
          AMDGPU_CDNA3_CORE_REG_CLASS_ID_SCC,
          AMDGPU_CDNA3_CORE_REG_CLASS_ID_EXEC,
          AMDGPU_CDNA3_CORE_REG_CLASS_ID_AGPR,
          LOOM_LOW_REG_CLASS_NONE,
      },
      {
          loom_amdgpu_cdna4_core_descriptor_set(),
          AMDGPU_CDNA4_CORE_REG_CLASS_ID_SGPR,
          106,
          AMDGPU_CDNA4_CORE_REG_CLASS_ID_VGPR,
          256,
          AMDGPU_CDNA4_CORE_REG_CLASS_ID_SCC,
          AMDGPU_CDNA4_CORE_REG_CLASS_ID_EXEC,
          AMDGPU_CDNA4_CORE_REG_CLASS_ID_AGPR,
          LOOM_LOW_REG_CLASS_NONE,
      },
      {
          loom_amdgpu_rdna3_core_descriptor_set(),
          AMDGPU_RDNA3_CORE_REG_CLASS_ID_SGPR,
          106,
          AMDGPU_RDNA3_CORE_REG_CLASS_ID_VGPR,
          256,
          AMDGPU_RDNA3_CORE_REG_CLASS_ID_SCC,
          AMDGPU_RDNA3_CORE_REG_CLASS_ID_EXEC,
          LOOM_LOW_REG_CLASS_NONE,
          LOOM_LOW_REG_CLASS_NONE,
      },
      {
          loom_amdgpu_rdna4_core_descriptor_set(),
          AMDGPU_RDNA4_CORE_REG_CLASS_ID_SGPR,
          106,
          AMDGPU_RDNA4_CORE_REG_CLASS_ID_VGPR,
          256,
          AMDGPU_RDNA4_CORE_REG_CLASS_ID_SCC,
          AMDGPU_RDNA4_CORE_REG_CLASS_ID_EXEC,
          LOOM_LOW_REG_CLASS_NONE,
          LOOM_LOW_REG_CLASS_NONE,
      },
      {
          loom_amdgpu_rdna4_gfx125x_core_descriptor_set(),
          AMDGPU_RDNA4_GFX125X_CORE_REG_CLASS_ID_SGPR,
          106,
          AMDGPU_RDNA4_GFX125X_CORE_REG_CLASS_ID_VGPR,
          1024,
          AMDGPU_RDNA4_GFX125X_CORE_REG_CLASS_ID_SCC,
          AMDGPU_RDNA4_GFX125X_CORE_REG_CLASS_ID_EXEC,
          LOOM_LOW_REG_CLASS_NONE,
          AMDGPU_RDNA4_GFX125X_CORE_REG_CLASS_ID_MODE,
      },
  };

  for (const DescriptorSetCase& c : cases) {
    ASSERT_NE(c.descriptor_set, nullptr);
    ExpectRegisterClass(c.descriptor_set, c.sgpr_id, IREE_SV("amdgpu.sgpr"), 32,
                        c.sgpr_allocatable_count,
                        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
    ExpectRegisterClass(c.descriptor_set, c.vgpr_id, IREE_SV("amdgpu.vgpr"), 32,
                        c.vgpr_allocatable_count,
                        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
    ExpectRegisterClass(
        c.descriptor_set, c.scc_id, IREE_SV("amdgpu.scc"), 1, 1,
        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL | LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
    ExpectRegisterClass(
        c.descriptor_set, c.exec_id, IREE_SV("amdgpu.exec"), 64, 1,
        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL | LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);

    if (c.agpr_id == LOOM_LOW_REG_CLASS_NONE) {
      ExpectRegisterClassMissing(c.descriptor_set, IREE_SV("amdgpu.agpr"));
    } else {
      ExpectRegisterClass(c.descriptor_set, c.agpr_id, IREE_SV("amdgpu.agpr"),
                          32, 256, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
    }

    if (c.mode_id == LOOM_LOW_REG_CLASS_NONE) {
      ExpectRegisterClassMissing(c.descriptor_set, IREE_SV("amdgpu.mode"));
    } else {
      ExpectRegisterClass(c.descriptor_set, c.mode_id, IREE_SV("amdgpu.mode"),
                          32, 1,
                          LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                              LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
    }
  }
}

TEST(AmdgpuRegistersTest, OccupancyPoolsStaySeparateFromAddressability) {
  struct OccupancyCase {
    // Generated descriptor-set ordinal for the occupancy model.
    uint16_t descriptor_set_ordinal;
    // SGPR register-file pool available to resident waves.
    uint32_t sgpr_pool_units;
    // SGPR allocation granularity used by occupancy calculations.
    uint32_t sgpr_allocation_granularity;
    // VGPR register-file pool available to resident waves.
    uint32_t vgpr_pool_units;
    // VGPR allocation granularity used by occupancy calculations.
    uint32_t vgpr_allocation_granularity;
    // AGPR register-file pool available to resident waves, or zero when absent.
    uint32_t agpr_pool_units;
    // AGPR allocation granularity used by occupancy calculations.
    uint32_t agpr_allocation_granularity;
  };
  const OccupancyCase cases[] = {
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3,
          800,
          16,
          512,
          8,
          256,
          4,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4,
          800,
          16,
          1024,
          4,
          256,
          4,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3,
          800,
          16,
          1024,
          4,
          0,
          0,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4,
          800,
          16,
          1024,
          4,
          0,
          0,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
          800,
          16,
          1024,
          4,
          0,
          0,
      },
  };

  for (const OccupancyCase& c : cases) {
    const loom_amdgpu_occupancy_model_t* model =
        loom_amdgpu_occupancy_model_for_descriptor_set_ordinal(
            c.descriptor_set_ordinal);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->descriptor_set_ordinal, c.descriptor_set_ordinal);
    ExpectOccupancyRegisterClass(model, 0, IREE_SV("amdgpu.sgpr"),
                                 c.sgpr_pool_units,
                                 c.sgpr_allocation_granularity);
    ExpectOccupancyRegisterClass(model, 1, IREE_SV("amdgpu.vgpr"),
                                 c.vgpr_pool_units,
                                 c.vgpr_allocation_granularity);
    if (c.agpr_pool_units == 0) {
      EXPECT_EQ(model->register_class_count, 2u);
    } else {
      ASSERT_EQ(model->register_class_count, 3u);
      ExpectOccupancyRegisterClass(model, 2, IREE_SV("amdgpu.agpr"),
                                   c.agpr_pool_units,
                                   c.agpr_allocation_granularity);
    }
  }
}

}  // namespace
