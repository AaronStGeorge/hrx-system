// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/system_memory.h"

#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

namespace {

constexpr int64_t kSystemCacheScope = LOOM_CACHE_SCOPE_SYSTEM;

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

enum class SystemMemoryAttrKind {
  kLoad,
  kReleaseStore,
  kNoReturnAtomic,
  kReturnAtomic,
};

struct ExpectedAttr {
  // Expected descriptor attr name.
  iree_string_view_t name;
  // Expected integer attr value.
  int64_t value;
};

class AmdgpuSystemMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    UseDescriptorSet(IREE_SV("amdgpu.rdna3.core"));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    BuildFunctionBody();
  }

  void TearDown() override {
    if (module_ != NULL) {
      loom_module_free(module_);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder_, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
  }

  void UseDescriptorSet(iree_string_view_t key) {
    descriptor_set_ =
        loom_low_descriptor_registry_lookup(&low_registry_.registry, key);
    if (descriptor_set_ == NULL) {
      GTEST_SKIP() << "AMDGPU descriptor set is not linked: " << ToString(key);
    }
  }

  void BuildFunctionBody() {
    loom_symbol_ref_t target = AddSymbol(IREE_SV("gfx_target"));
    loom_symbol_ref_t callee = AddSymbol(IREE_SV("test_fn"));
    loom_op_t* function_op = NULL;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        /*purity=*/0, /*allocation=*/0, /*schedule=*/0, target, /*abi=*/0,
        loom_make_named_attr_slice(NULL, 0),
        loom_make_named_attr_slice(NULL, 0),
        /*export_symbol=*/LOOM_STRING_ID_INVALID,
        loom_make_named_attr_slice(NULL, 0), callee, /*arg_types=*/NULL,
        /*arg_types_count=*/0, /*result_types=*/NULL, /*result_count=*/0,
        /*tied_results=*/NULL, /*tied_result_count=*/0, /*predicates=*/NULL,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op));
    body_block_ = loom_region_entry_block(loom_low_func_def_body(function_op));
    loom_builder_initialize(module_, &module_->arena, body_block_, &builder_);
    builder_.ip.parent_op = function_op;
  }

  std::vector<loom_op_t*> Ops() const {
    std::vector<loom_op_t*> ops;
    loom_op_t* op = NULL;
    loom_block_for_each_op(body_block_, op) { ops.push_back(op); }
    return ops;
  }

  iree_string_view_t String(loom_string_id_t string_id) const {
    return module_->strings.entries[string_id];
  }

  const loom_low_descriptor_t* DescriptorForRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    const uint32_t descriptor_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set_, descriptor_ref);
    return loom_low_descriptor_set_descriptor_at(descriptor_set_,
                                                 descriptor_ordinal);
  }

  void ExpectLowOpDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_op_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_op_descriptor_ordinal(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
    EXPECT_EQ(ToString(String(loom_low_op_opcode(op))),
              ToString(loom_low_descriptor_set_string(
                  descriptor_set_, descriptor->key_string_offset)));
  }

  void ExpectAttrI64(loom_named_attr_slice_t attrs, iree_string_view_t name,
                     int64_t expected_value) const {
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (iree_string_view_equal(String(attrs.entries[i].name_id), name)) {
        EXPECT_EQ(loom_attr_as_i64(attrs.entries[i].value), expected_value);
        return;
      }
    }
    FAIL() << "expected attr not found: " << ToString(name);
  }

  void ExpectAttrs(loom_named_attr_slice_t attrs,
                   std::initializer_list<ExpectedAttr> expected_attrs) const {
    ASSERT_EQ(attrs.count, expected_attrs.size());
    for (const ExpectedAttr& expected_attr : expected_attrs) {
      ExpectAttrI64(attrs, expected_attr.name, expected_attr.value);
    }
  }

  void ExpectAppendedAttrs(iree_string_view_t descriptor_set_key,
                           SystemMemoryAttrKind attr_kind,
                           std::initializer_list<ExpectedAttr> expected_attrs) {
    UseDescriptorSet(descriptor_set_key);
    loom_named_attr_t attrs[4] = {};
    iree_host_size_t attr_count = 0;
    iree_status_t status = iree_ok_status();
    switch (attr_kind) {
      case SystemMemoryAttrKind::kLoad:
        status = loom_amdgpu_system_memory_append_load_attrs(
            &builder_, descriptor_set_, attrs, IREE_ARRAYSIZE(attrs),
            &attr_count);
        break;
      case SystemMemoryAttrKind::kReleaseStore:
        status = loom_amdgpu_system_memory_append_release_store_attrs(
            &builder_, descriptor_set_, attrs, IREE_ARRAYSIZE(attrs),
            &attr_count);
        break;
      case SystemMemoryAttrKind::kNoReturnAtomic:
        status = loom_amdgpu_system_memory_append_no_return_atomic_attrs(
            &builder_, descriptor_set_, attrs, IREE_ARRAYSIZE(attrs),
            &attr_count);
        break;
      case SystemMemoryAttrKind::kReturnAtomic:
        status = loom_amdgpu_system_memory_append_return_atomic_attrs(
            &builder_, descriptor_set_, attrs, IREE_ARRAYSIZE(attrs),
            &attr_count);
        break;
    }
    IREE_ASSERT_OK(status);
    ExpectAttrs(loom_make_named_attr_slice(attrs, attr_count), expected_attrs);
  }

  void ExpectOpAttrs(const loom_op_t* op,
                     std::initializer_list<ExpectedAttr> expected_attrs) const {
    ExpectAttrs(loom_low_op_attrs(op), expected_attrs);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(AmdgpuSystemMemoryTest, AppendsLoadAttrsByArchitecture) {
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna3.core"), SystemMemoryAttrKind::kLoad,
                      {{IREE_SV("glc"), 1}});
  ExpectAppendedAttrs(IREE_SV("amdgpu.cdna3.core"), SystemMemoryAttrKind::kLoad,
                      {{IREE_SV("sc0"), 1}, {IREE_SV("sc1"), 1}});
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna4.core"), SystemMemoryAttrKind::kLoad,
                      {{IREE_SV("scope"), kSystemCacheScope}});
}

TEST_F(AmdgpuSystemMemoryTest, AppendsReleaseStoreAttrsByArchitecture) {
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna3.core"),
                      SystemMemoryAttrKind::kReleaseStore, {});
  ExpectAppendedAttrs(IREE_SV("amdgpu.cdna3.core"),
                      SystemMemoryAttrKind::kReleaseStore,
                      {{IREE_SV("sc0"), 1}, {IREE_SV("sc1"), 1}});
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna4.core"),
                      SystemMemoryAttrKind::kReleaseStore,
                      {{IREE_SV("scope"), kSystemCacheScope}});
}

TEST_F(AmdgpuSystemMemoryTest, AppendsNoReturnAtomicAttrsByArchitecture) {
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna3.core"),
                      SystemMemoryAttrKind::kNoReturnAtomic, {});
  ExpectAppendedAttrs(IREE_SV("amdgpu.cdna3.core"),
                      SystemMemoryAttrKind::kNoReturnAtomic,
                      {{IREE_SV("sc1"), 1}});
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna4.core"),
                      SystemMemoryAttrKind::kNoReturnAtomic,
                      {{IREE_SV("scope"), kSystemCacheScope}});
}

TEST_F(AmdgpuSystemMemoryTest, AppendsReturnAtomicAttrsByArchitecture) {
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna3.core"),
                      SystemMemoryAttrKind::kReturnAtomic, {});
  ExpectAppendedAttrs(IREE_SV("amdgpu.cdna3.core"),
                      SystemMemoryAttrKind::kReturnAtomic,
                      {{IREE_SV("sc0"), 1}, {IREE_SV("sc1"), 1}});
  ExpectAppendedAttrs(IREE_SV("amdgpu.rdna4.core"),
                      SystemMemoryAttrKind::kReturnAtomic,
                      {{IREE_SV("scope"), kSystemCacheScope}});
}

TEST_F(AmdgpuSystemMemoryTest, EmitsRdnaReleaseOrdering) {
  IREE_ASSERT_OK(loom_amdgpu_system_memory_build_release_ordering(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 2u);
  ExpectLowOpDescriptorRef(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ExpectOpAttrs(ops[0], {{IREE_SV("vmcnt"), 0}, {IREE_SV("lgkmcnt"), 15}});
  ExpectLowOpDescriptorRef(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ExpectOpAttrs(ops[1], {{IREE_SV("vscnt"), 0}});
}

TEST_F(AmdgpuSystemMemoryTest, EmitsCdnaReleaseOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  IREE_ASSERT_OK(loom_amdgpu_system_memory_build_release_ordering(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 1u);
  ExpectLowOpDescriptorRef(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2);
  ExpectOpAttrs(ops[0], {{IREE_SV("sc0"), 1}, {IREE_SV("sc1"), 1}});
}

TEST_F(AmdgpuSystemMemoryTest, EmitsGfx12ReleaseOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  IREE_ASSERT_OK(loom_amdgpu_system_memory_build_release_ordering(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 2u);
  ExpectLowOpDescriptorRef(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB);
  ExpectOpAttrs(ops[0], {{IREE_SV("scope"), kSystemCacheScope}});
  ExpectLowOpDescriptorRef(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT);
  ExpectOpAttrs(ops[1], {{IREE_SV("storecnt"), 0}});
}

TEST_F(AmdgpuSystemMemoryTest, EmitsRdnaAcquireOrdering) {
  IREE_ASSERT_OK(loom_amdgpu_system_memory_build_acquire_ordering(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 3u);
  ExpectLowOpDescriptorRef(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ExpectOpAttrs(ops[0], {{IREE_SV("vmcnt"), 0}, {IREE_SV("lgkmcnt"), 15}});
  ExpectLowOpDescriptorRef(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV);
  ExpectOpAttrs(ops[1], {});
  ExpectLowOpDescriptorRef(ops[2], LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV);
  ExpectOpAttrs(ops[2], {});
}

TEST_F(AmdgpuSystemMemoryTest, EmitsCdnaAcquireOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  IREE_ASSERT_OK(loom_amdgpu_system_memory_build_acquire_ordering(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 2u);
  ExpectLowOpDescriptorRef(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ExpectOpAttrs(ops[0], {{IREE_SV("vmcnt"), 0}, {IREE_SV("lgkmcnt"), 15}});
  ExpectLowOpDescriptorRef(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV);
  ExpectOpAttrs(ops[1], {{IREE_SV("sc0"), 1}, {IREE_SV("sc1"), 1}});
}

TEST_F(AmdgpuSystemMemoryTest, EmitsGfx12AcquireOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  IREE_ASSERT_OK(loom_amdgpu_system_memory_build_acquire_ordering(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 2u);
  ExpectLowOpDescriptorRef(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT);
  ExpectOpAttrs(ops[0], {{IREE_SV("loadcnt"), 0}});
  ExpectLowOpDescriptorRef(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV);
  ExpectOpAttrs(ops[1], {{IREE_SV("scope"), kSystemCacheScope}});
}

}  // namespace
