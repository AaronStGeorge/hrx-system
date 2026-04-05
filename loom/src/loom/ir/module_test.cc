// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    .name = IREE_SV("q8_0"),
};

static const loom_encoding_vtable_t kQ6KEncodingVtable = {
    .name = IREE_SV("q6_k"),
};

static const loom_encoding_vtable_t kDenseEncodingVtable = {
    .name = IREE_SV("dense"),
};

class ModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ6KEncodingVtable));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kDenseEncodingVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

//===----------------------------------------------------------------------===//
// Module lifecycle
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, AllocateAndFree) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_NE(module->body, nullptr);
  EXPECT_EQ(module->body->block_count, 1);
  loom_module_free(module);
}

TEST_F(ModuleTest, FreeNull) { loom_module_free(NULL); }

TEST_F(ModuleTest, ModuleName) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("my_module"),
                                      &block_pool_, NULL,
                                      iree_allocator_system(), &module));
  iree_string_view_t name = module->strings.entries[module->name_id];
  EXPECT_TRUE(iree_string_view_equal(name, IREE_SV("my_module")));
  loom_module_free(module);
}

TEST_F(ModuleTest, BodyBlock) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(block->op_count, 0);
  EXPECT_GT(block->op_capacity, 0);
  loom_module_free(module);
}

TEST_F(ModuleTest, RegionAppendBlockGrowthKeepsBlockReferencesStable) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_region_t* body = module->body;
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 1u);
  ASSERT_EQ(body->block_capacity, 1u);

  loom_block_t* old_entry = loom_region_entry_block(body);

  loom_value_id_t arg_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &arg_id));
  IREE_ASSERT_OK(loom_block_add_arg(module, old_entry, arg_id));

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op));
  memset(op, 0, sizeof(loom_op_t));
  op->kind = LOOM_OP_TEST_CONSTANT;
  op->parent_block = old_entry;
  IREE_ASSERT_OK(loom_block_append_op(module, old_entry, op));

  loom_block_t* appended = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &appended));

  ASSERT_EQ(body->block_count, 2u);
  EXPECT_GE(body->block_capacity, body->block_count);
  EXPECT_EQ(appended, loom_region_block(body, 1));
  EXPECT_EQ(loom_region_entry_block(body), old_entry);
  EXPECT_EQ(appended->label_id, LOOM_STRING_ID_INVALID);
  EXPECT_EQ(appended->arg_count, 0u);
  EXPECT_EQ(appended->op_count, 0u);
  EXPECT_GT(appended->op_capacity, 0u);

  loom_block_t* entry = loom_region_entry_block(body);
  EXPECT_EQ(loom_value_def_block(&module->values.entries[arg_id]), entry);
  EXPECT_EQ(loom_value_def_index(&module->values.entries[arg_id]), 0u);
  EXPECT_EQ(op->parent_block, entry);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Value definition
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, DefineValue) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id));
  EXPECT_EQ(id, 0u);
  EXPECT_EQ(module->values.count, 1u);

  loom_value_t* value = &module->values.entries[id];
  EXPECT_EQ(loom_type_kind(value->type), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_element_type(value->type), LOOM_SCALAR_TYPE_F32);
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineMultipleValues) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t id0 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t id1 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t id2 = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id0));
  IREE_ASSERT_OK(loom_module_define_value(module, i32, &id1));
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id2));

  EXPECT_EQ(id0, 0u);
  EXPECT_EQ(id1, 1u);
  EXPECT_EQ(id2, 2u);
  EXPECT_EQ(module->values.count, 3u);
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueAlignment) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, f32, &id));
  EXPECT_EQ((uintptr_t)&module->values.entries[id] % 64, 0u)
      << "Value entries must be 64-byte aligned";
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueGrowth) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  // Fill past the default capacity to trigger growth.
  iree_host_size_t initial_capacity = module->values.capacity;
  for (iree_host_size_t i = 0; i < initial_capacity + 100; ++i) {
    loom_value_id_t id = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_define_value(module, f32, &id));
    EXPECT_EQ(id, (loom_value_id_t)i);
  }
  EXPECT_GT(module->values.capacity, initial_capacity);
  EXPECT_EQ(module->values.count, initial_capacity + 100);

  // Verify all values are still accessible after growth.
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    EXPECT_EQ(loom_type_kind(module->values.entries[i].type), LOOM_TYPE_SCALAR);
  }
  loom_module_free(module);
}

TEST_F(ModuleTest, DefineValueRejectsInvalidSentinelId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  module->values.count = LOOM_VALUE_ID_INVALID;

  loom_value_id_t id = LOOM_VALUE_ID_INVALID;
  iree_status_t status = loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// String interning
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, InternString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id));
  EXPECT_NE(id, LOOM_STRING_ID_INVALID);

  iree_string_view_t stored = module->strings.entries[id];
  EXPECT_TRUE(iree_string_view_equal(stored, IREE_SV("hello")));
  loom_module_free(module);
}

TEST_F(ModuleTest, InternStringDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id1 = LOOM_STRING_ID_INVALID;
  loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id1));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id2));
  EXPECT_EQ(id1, id2);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternDifferentStrings) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id1 = LOOM_STRING_ID_INVALID;
  loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id1));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("world"), &id2));
  EXPECT_NE(id1, id2);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternEmptyString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, iree_string_view_empty(), &id));
  EXPECT_NE(id, LOOM_STRING_ID_INVALID);
  iree_string_view_t stored = module->strings.entries[id];
  EXPECT_EQ(stored.size, 0u);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternStringStress) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  // Intern 1000 unique strings and verify dedup.
  char buffer[32];
  for (int i = 0; i < 1000; ++i) {
    int length = snprintf(buffer, sizeof(buffer), "string_%d", i);
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_string_view(buffer, length), &id));
    // Intern again, expect same ID.
    loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_string_view(buffer, length), &id2));
    EXPECT_EQ(id, id2);
  }
  loom_module_free(module);
}

TEST_F(ModuleTest, LookupString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t hello_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &hello_id));

  EXPECT_EQ(loom_module_lookup_string(module, IREE_SV("hello")), hello_id);
  EXPECT_EQ(loom_module_lookup_string(module, IREE_SV("missing")),
            LOOM_STRING_ID_INVALID);
  EXPECT_EQ(module->strings.count, 2u);  // "test" module name + "hello".

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Canonical dictionary attributes
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, MakeCanonicalAttrDictSortsByKeySpellingAndCopiesEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));
  ASSERT_LT(zeta_id, alpha_id)
      << "setup should prove string_id order is not the"
         " same as spelling order";

  loom_named_attr_t entries[2] = {
      {.name_id = zeta_id, .value = loom_attr_i64(2)},
      {.name_id = alpha_id, .value = loom_attr_i64(1)},
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      &attr));

  EXPECT_EQ(attr.kind, LOOM_ATTR_DICT);
  EXPECT_EQ(attr.count, 2);
  ASSERT_NE(attr.dict_entries, nullptr);
  EXPECT_NE(attr.dict_entries, entries);
  EXPECT_EQ(attr.dict_entries[0].name_id, alpha_id);
  EXPECT_EQ(attr.dict_entries[0].reserved, 0u);
  EXPECT_EQ(attr.dict_entries[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(attr.dict_entries[0].value.i64, 1);
  EXPECT_EQ(attr.dict_entries[1].name_id, zeta_id);
  EXPECT_EQ(attr.dict_entries[1].reserved, 0u);
  EXPECT_EQ(attr.dict_entries[1].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(attr.dict_entries[1].value.i64, 2);

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRecursivelyCanonicalizesNestedDicts) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t outer_zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t outer_axis_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t inner_zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t inner_alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("zeta"), &outer_zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("axis"), &outer_axis_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("inner_z"), &inner_zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("inner_a"), &inner_alpha_id));

  int64_t original_values[3] = {7, 8, 9};
  loom_predicate_t original_predicates[1] = {{
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST, 0},
      .args = {0, 16, 0},
  }};
  loom_named_attr_t inner_entries[2] = {
      {
          .name_id = inner_zeta_id,
          .value = loom_attr_i64_array(original_values, 3),
      },
      {
          .name_id = inner_alpha_id,
          .value = loom_attr_predicate_list(original_predicates, 1),
      },
  };
  loom_named_attr_t outer_entries[2] = {
      {
          .name_id = outer_zeta_id,
          .value = loom_make_canonical_attr_dict(inner_entries,
                                                 IREE_ARRAYSIZE(inner_entries)),
      },
      {.name_id = outer_axis_id, .value = loom_attr_i64(4)},
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(outer_entries, IREE_ARRAYSIZE(outer_entries)),
      &attr));

  // Mutate the caller-owned buffers after construction. The canonical dict
  // must keep its own arena-owned snapshots.
  inner_entries[0].name_id = outer_axis_id;
  inner_entries[0].value = loom_attr_i64(99);
  outer_entries[0].name_id = outer_axis_id;
  outer_entries[0].value = loom_attr_i64(42);
  original_values[0] = 1234;
  original_predicates[0].kind = LOOM_PREDICATE_EQ;

  EXPECT_EQ(attr.count, 2);
  ASSERT_NE(attr.dict_entries, nullptr);
  EXPECT_EQ(attr.dict_entries[0].name_id, outer_axis_id);
  EXPECT_EQ(attr.dict_entries[0].value.i64, 4);
  EXPECT_EQ(attr.dict_entries[1].name_id, outer_zeta_id);
  ASSERT_EQ(attr.dict_entries[1].value.kind, LOOM_ATTR_DICT);

  loom_attribute_t nested = attr.dict_entries[1].value;
  EXPECT_EQ(nested.count, 2);
  ASSERT_NE(nested.dict_entries, nullptr);
  EXPECT_EQ(nested.dict_entries[0].name_id, inner_alpha_id);
  EXPECT_EQ(nested.dict_entries[0].reserved, 0u);
  EXPECT_EQ(nested.dict_entries[0].value.kind, LOOM_ATTR_PREDICATE_LIST);
  ASSERT_NE(nested.dict_entries[0].value.predicate_list, original_predicates);
  EXPECT_EQ(nested.dict_entries[0].value.count, 1);
  EXPECT_EQ(nested.dict_entries[0].value.predicate_list[0].kind,
            LOOM_PREDICATE_RANGE);
  EXPECT_EQ(nested.dict_entries[1].name_id, inner_zeta_id);
  EXPECT_EQ(nested.dict_entries[1].reserved, 0u);
  EXPECT_EQ(nested.dict_entries[1].value.kind, LOOM_ATTR_I64_ARRAY);
  ASSERT_NE(nested.dict_entries[1].value.i64_array, original_values);
  ASSERT_EQ(nested.dict_entries[1].value.count, 3);
  EXPECT_EQ(nested.dict_entries[1].value.i64_array[0], 7);
  EXPECT_EQ(nested.dict_entries[1].value.i64_array[1], 8);
  EXPECT_EQ(nested.dict_entries[1].value.i64_array[2], 9);

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRejectsDuplicateKeys) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("axis"), &key_id));

  loom_named_attr_t entries[2] = {
      {.name_id = key_id, .value = loom_attr_i64(0)},
      {.name_id = key_id, .value = loom_attr_i64(1)},
  };

  loom_attribute_t attr = {0};
  iree_status_t status = loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      &attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRejectsNonEmptyNullEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t attr = {0};
  iree_status_t status = loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(/*entries=*/NULL, /*count=*/1), &attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictNormalizesEqualityAndHash) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_t zeta_first_entries[2] = {
      {.name_id = zeta_id, .value = loom_attr_i64(2)},
      {.name_id = alpha_id, .value = loom_attr_i64(1)},
  };
  loom_named_attr_t alpha_first_entries[2] = {
      {.name_id = alpha_id, .value = loom_attr_i64(1)},
      {.name_id = zeta_id, .value = loom_attr_i64(2)},
  };

  loom_attribute_t zeta_first_attr = {0};
  loom_attribute_t alpha_first_attr = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(zeta_first_entries,
                                 IREE_ARRAYSIZE(zeta_first_entries)),
      &zeta_first_attr));
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(alpha_first_entries,
                                 IREE_ARRAYSIZE(alpha_first_entries)),
      &alpha_first_attr));

  EXPECT_TRUE(loom_attribute_equal(&zeta_first_attr, &alpha_first_attr));
  EXPECT_EQ(loom_attribute_hash(&zeta_first_attr),
            loom_attribute_hash(&alpha_first_attr));

  loom_module_free(module);
}

TEST_F(ModuleTest, MakeCanonicalAttrDictRejectsUnknownKeyStringId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_named_attr_t entries[1] = {{
      .name_id = 99,
      .value = loom_attr_i64(1),
  }};

  loom_attribute_t attr = {0};
  iree_status_t status = loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      &attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest,
       ReplaceCanonicalAttrDictAppliesAddReplaceRemoveAndKeepsSortedOrder) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t beta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("beta"), &beta_id));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));

  loom_named_attr_t base_entries[2] = {
      {.name_id = alpha_id, .value = loom_attr_i64(1)},
      {.name_id = zeta_id, .value = loom_attr_i64(3)},
  };
  loom_named_attr_update_t updates[3] = {
      loom_named_attr_replace(zeta_id, loom_attr_i64(30)),
      loom_named_attr_remove(alpha_id),
      loom_named_attr_replace(beta_id, loom_attr_i64(20)),
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_replace_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(base_entries, IREE_ARRAYSIZE(base_entries)),
      loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates)),
      &attr));

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  ASSERT_EQ(attr.kind, LOOM_ATTR_DICT);
  ASSERT_EQ(attr.count, 2u);
  ASSERT_NE(attr.dict_entries, nullptr);
  EXPECT_EQ(attr.dict_entries[0].name_id, beta_id);
  EXPECT_EQ(loom_attr_as_i64(attr.dict_entries[0].value), 20);
  EXPECT_EQ(attr.dict_entries[1].name_id, zeta_id);
  EXPECT_EQ(loom_attr_as_i64(attr.dict_entries[1].value), 30);

  loom_module_free(module);
}

TEST_F(ModuleTest, ReplaceCanonicalAttrDictRejectsDuplicateUpdateKeysByNameId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_t base_entries[1] = {
      {.name_id = alpha_id, .value = loom_attr_i64(1)},
  };
  loom_named_attr_update_t updates[2] = {
      loom_named_attr_replace(alpha_id, loom_attr_i64(2)),
      loom_named_attr_remove(alpha_id),
  };

  loom_attribute_t attr = {0};
  iree_status_t status = loom_module_replace_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(base_entries, IREE_ARRAYSIZE(base_entries)),
      loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates)),
      &attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest,
       ReplaceCanonicalAttrDictRecursivelyCanonicalizesNestedUpdateValues) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t outer_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("outer"), &outer_id));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));

  loom_named_attr_t nested_entries[2] = {
      {.name_id = zeta_id, .value = loom_attr_i64(2)},
      {.name_id = alpha_id, .value = loom_attr_i64(1)},
  };
  loom_named_attr_t base_entries[1] = {
      {.name_id = outer_id, .value = loom_attr_i64(0)},
  };
  loom_named_attr_update_t updates[1] = {
      loom_named_attr_replace(
          outer_id, loom_make_canonical_attr_dict(
                        nested_entries, IREE_ARRAYSIZE(nested_entries))),
  };

  loom_attribute_t attr = {0};
  IREE_ASSERT_OK(loom_module_replace_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(base_entries, IREE_ARRAYSIZE(base_entries)),
      loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates)),
      &attr));

  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, attr));
  ASSERT_EQ(attr.count, 1u);
  ASSERT_NE(attr.dict_entries, nullptr);
  ASSERT_EQ(attr.dict_entries[0].name_id, outer_id);
  ASSERT_EQ(attr.dict_entries[0].value.kind, LOOM_ATTR_DICT);
  loom_attribute_t nested = attr.dict_entries[0].value;
  ASSERT_EQ(nested.count, 2u);
  ASSERT_NE(nested.dict_entries, nullptr);
  EXPECT_NE(nested.dict_entries, nested_entries);
  EXPECT_EQ(nested.dict_entries[0].name_id, alpha_id);
  EXPECT_EQ(loom_attr_as_i64(nested.dict_entries[0].value), 1);
  EXPECT_EQ(nested.dict_entries[1].name_id, zeta_id);
  EXPECT_EQ(loom_attr_as_i64(nested.dict_entries[1].value), 2);

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsUnsortedInput) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_t entries[2] = {
      {.name_id = zeta_id, .value = loom_attr_i64(2)},
      {.name_id = alpha_id, .value = loom_attr_i64(1)},
  };
  loom_attribute_t attr =
      loom_make_canonical_attr_dict(entries, IREE_ARRAYSIZE(entries));

  iree_status_t status = loom_module_verify_canonical_attr_dict(module, attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsDuplicateKeys) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("axis"), &key_id));

  loom_named_attr_t entries[2] = {
      {.name_id = key_id, .value = loom_attr_i64(0)},
      {.name_id = key_id, .value = loom_attr_i64(1)},
  };
  loom_attribute_t attr =
      loom_make_canonical_attr_dict(entries, IREE_ARRAYSIZE(entries));

  iree_status_t status = loom_module_verify_canonical_attr_dict(module, attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsNonEmptyNullEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_attribute_t attr =
      loom_make_canonical_attr_dict(/*entries=*/NULL, /*count=*/1);

  iree_status_t status = loom_module_verify_canonical_attr_dict(module, attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest, VerifyCanonicalAttrDictRejectsEmptyDictWithNonNullEntries) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("axis"), &key_id));
  loom_named_attr_t entries[1] = {{
      .name_id = key_id,
      .value = loom_attr_i64(0),
  }};
  loom_attribute_t attr = {
      .kind = LOOM_ATTR_DICT,
      .count = 0,
      .dict_entries = entries,
  };

  iree_status_t status = loom_module_verify_canonical_attr_dict(module, attr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(ModuleTest, InternStringRejectsInvalidSentinelIdButKeepsDedupWorking) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t existing_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &existing_id));

  module->strings.count = LOOM_STRING_ID_INVALID;

  loom_string_id_t duplicate_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &duplicate_id));
  EXPECT_EQ(duplicate_id, existing_id);

  loom_string_id_t new_id = LOOM_STRING_ID_INVALID;
  iree_status_t status =
      loom_module_intern_string(module, IREE_SV("world"), &new_id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Type interning
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, InternScalarType) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t interned = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned));
  EXPECT_EQ(loom_type_kind(interned), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_element_type(interned), LOOM_SCALAR_TYPE_F32);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternTypeDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t interned1 = {0};
  loom_type_t interned2 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned1));
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned2));
  // Same type interned twice should produce identical results.
  EXPECT_EQ(interned1.header, interned2.header);
  EXPECT_EQ(module->types.count, 1u);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternDifferentTypes) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t interned_f32 = {0};
  loom_type_t interned_i32 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned_f32));
  IREE_ASSERT_OK(loom_module_intern_type(module, i32, &interned_i32));
  EXPECT_NE(interned_f32.header, interned_i32.header);
  EXPECT_EQ(module->types.count, 2u);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternShapedType) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_type_t tile_4x4_f32 =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(4), 0);
  loom_type_t interned1 = {0};
  loom_type_t interned2 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, tile_4x4_f32, &interned1));
  IREE_ASSERT_OK(loom_module_intern_type(module, tile_4x4_f32, &interned2));
  EXPECT_EQ(interned1.header, interned2.header);
  EXPECT_EQ(interned1.dims[0], interned2.dims[0]);
  EXPECT_EQ(interned1.dims[1], interned2.dims[1]);
  EXPECT_EQ(module->types.count, 1u);
  loom_module_free(module);
}

TEST_F(ModuleTest, InternFunctionTypeDedupsStructurallyAndOwnsPayload) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t arg_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t source_a = {0};
  IREE_ASSERT_OK(loom_type_function_build(&arg_type, 1, &result_type, 1,
                                          iree_allocator_system(), &source_a));
  const loom_func_type_data_t* source_a_data = loom_type_func_data(source_a);

  loom_type_t source_b = {0};
  IREE_ASSERT_OK(loom_type_function_build(&arg_type, 1, &result_type, 1,
                                          iree_allocator_system(), &source_b));

  loom_type_t interned_a = {0};
  loom_type_t interned_b = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, source_a, &interned_a));
  IREE_ASSERT_OK(loom_module_intern_type(module, source_b, &interned_b));

  EXPECT_EQ(module->types.count, 1u);
  EXPECT_TRUE(loom_type_equal(interned_a, interned_b));
  EXPECT_EQ(loom_type_hash(interned_a), loom_type_hash(interned_b));
  EXPECT_NE(loom_type_func_data(interned_a), source_a_data);

  iree_allocator_free(iree_allocator_system(), (void*)source_a_data);
  iree_allocator_free(iree_allocator_system(),
                      (void*)loom_type_func_data(source_b));

  ASSERT_EQ(loom_type_func_arg_count(interned_a), 1u);
  ASSERT_EQ(loom_type_func_result_count(interned_a), 1u);
  EXPECT_TRUE(
      loom_type_equal(loom_type_func_arg_types(interned_a)[0], arg_type));
  EXPECT_TRUE(
      loom_type_equal(loom_type_func_result_types(interned_a)[0], result_type));

  loom_module_free(module);
}

TEST_F(ModuleTest, InternDialectTypeCopiesTemporaryParamsAndPreservesNameId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("dialect.type"), &name_id));

  loom_type_t params[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t source =
      loom_type_dialect(name_id, IREE_ARRAYSIZE(params), params);

  loom_type_t interned = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, source, &interned));

  params[0] = loom_type_scalar(LOOM_SCALAR_TYPE_I64);

  EXPECT_EQ(loom_type_dialect_name_id(interned), name_id);
  ASSERT_EQ(loom_type_dialect_param_count(interned), 2u);
  EXPECT_TRUE(loom_type_equal(loom_type_dialect_params(interned)[0],
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_TRUE(
      loom_type_equal(loom_type_dialect_params(interned)[1],
                      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                                          loom_dim_pack_static(4), 0)));

  loom_type_t duplicate_params[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t duplicate_source = loom_type_dialect(
      name_id, IREE_ARRAYSIZE(duplicate_params), duplicate_params);
  loom_type_t duplicate_interned = {0};
  IREE_ASSERT_OK(
      loom_module_intern_type(module, duplicate_source, &duplicate_interned));

  EXPECT_EQ(module->types.count, 1u);
  EXPECT_TRUE(loom_type_equal(interned, duplicate_interned));

  loom_module_free(module);
}

TEST_F(ModuleTest, InternTypeRejectsInvalidSentinelIdButKeepsDedupWorking) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t interned_f32 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &interned_f32));

  module->types.count = LOOM_TYPE_ID_INVALID;

  loom_type_t duplicate_f32 = {0};
  IREE_ASSERT_OK(loom_module_intern_type(module, f32, &duplicate_f32));
  EXPECT_EQ(duplicate_f32.header, interned_f32.header);

  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t new_i32 = {0};
  iree_status_t status = loom_module_intern_type(module, i32, &new_i32);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Block operations
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, BlockAppendOp) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  // Allocate a dummy op.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op));
  memset(op, 0, sizeof(loom_op_t));
  op->kind = 0x0100;

  IREE_ASSERT_OK(loom_block_append_op(module, block, op));
  EXPECT_EQ(block->op_count, 1);
  EXPECT_EQ(block->ops[0], op);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockAppendOrdering) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  loom_op_t* ops[3];
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_arena_allocate(&module->arena, sizeof(loom_op_t),
                                       (void**)&ops[i]));
    memset(ops[i], 0, sizeof(loom_op_t));
    ops[i]->kind = (loom_op_kind_t)(0x0100 + i);
    IREE_ASSERT_OK(loom_block_append_op(module, block, ops[i]));
  }

  EXPECT_EQ(block->op_count, 3);
  EXPECT_EQ(block->ops[0], ops[0]);
  EXPECT_EQ(block->ops[1], ops[1]);
  EXPECT_EQ(block->ops[2], ops[2]);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockInsertAtBeginning) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  loom_op_t* op_a = NULL;
  loom_op_t* op_b = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_a));
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_b));
  memset(op_a, 0, sizeof(loom_op_t));
  memset(op_b, 0, sizeof(loom_op_t));

  IREE_ASSERT_OK(loom_block_append_op(module, block, op_a));
  IREE_ASSERT_OK(loom_block_insert_op(module, block, 0, op_b));

  EXPECT_EQ(block->op_count, 2);
  EXPECT_EQ(block->ops[0], op_b);
  EXPECT_EQ(block->ops[1], op_a);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockFindOp) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);

  loom_op_t* op_a = NULL;
  loom_op_t* op_b = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_a));
  IREE_ASSERT_OK(
      iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op_b));
  memset(op_a, 0, sizeof(loom_op_t));
  memset(op_b, 0, sizeof(loom_op_t));

  IREE_ASSERT_OK(loom_block_append_op(module, block, op_a));
  IREE_ASSERT_OK(loom_block_append_op(module, block, op_b));

  EXPECT_EQ(loom_block_find_op(block, op_a), 0);
  EXPECT_EQ(loom_block_find_op(block, op_b), 1);

  // Not-found case.
  loom_op_t dummy = {0};
  EXPECT_EQ(loom_block_find_op(block, &dummy), UINT16_MAX);
  loom_module_free(module);
}

TEST_F(ModuleTest, BlockGrowth) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_block_t* block = loom_module_block(module);
  uint16_t initial_capacity = block->op_capacity;

  // Append enough ops to trigger growth.
  for (int i = 0; i < initial_capacity + 10; ++i) {
    loom_op_t* op = NULL;
    IREE_ASSERT_OK(
        iree_arena_allocate(&module->arena, sizeof(loom_op_t), (void**)&op));
    memset(op, 0, sizeof(loom_op_t));
    op->kind = (loom_op_kind_t)i;
    IREE_ASSERT_OK(loom_block_append_op(module, block, op));
  }

  EXPECT_GT(block->op_capacity, initial_capacity);
  EXPECT_EQ(block->op_count, initial_capacity + 10);

  // Verify ordering survived growth.
  for (int i = 0; i < initial_capacity + 10; ++i) {
    EXPECT_EQ(block->ops[i]->kind, (loom_op_kind_t)i);
  }
  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Size hints
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, SizeHints) {
  loom_module_size_hints_t hints = {
      .value_count = 100,
      .string_count = 50,
      .type_count = 20,
      .symbol_count = 10,
  };
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      &hints, iree_allocator_system(),
                                      &module));
  // Capacities should be at least hint * growth_factor.
  EXPECT_GE(module->values.capacity, 100u);
  EXPECT_GE(module->strings.capacity, 50u);
  EXPECT_GE(module->types.capacity, 20u);
  EXPECT_GE(module->symbols.capacity, 10u);
  loom_module_free(module);
}

TEST_F(ModuleTest, AddLocationRejectsWrappedIdZero) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  module->locations.count = (iree_host_size_t)UINT32_MAX + 1;

  loom_location_id_t id = LOOM_LOCATION_UNKNOWN;
  iree_status_t status = loom_module_add_location(
      module,
      loom_location_file_range(/*source_id=*/0, /*start_line=*/1,
                               /*start_col=*/1, /*end_line=*/1,
                               /*end_col=*/2),
      &id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Encoding table
//===----------------------------------------------------------------------===//

TEST_F(ModuleTest, AddEncodingBasic) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  // Intern the encoding name.
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));

  // Intern a param name and build a named attribute.
  loom_string_id_t block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("block"), &block_id));
  loom_named_attr_t param = {
      .name_id = block_id,
      .value = loom_attr_i64(32),
  };

  loom_encoding_t encoding = {
      .name_id = name_id,
      .alias_id = LOOM_STRING_ID_INVALID,
      .attribute_count = 1,
      .attributes = &param,
  };

  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
  EXPECT_EQ(encoding_id, 1);
  EXPECT_EQ(module->encodings.count, 1u);

  // Look it up.
  const loom_encoding_t* found = loom_module_encoding(module, encoding_id);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->name_id, name_id);
  EXPECT_EQ(found->attribute_count, 1);
  EXPECT_EQ(found->attributes[0].name_id, block_id);
  EXPECT_EQ(found->attributes[0].value.i64, 32);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("dense"), &name_id));

  loom_encoding_t encoding = {
      .name_id = name_id,
      .alias_id = LOOM_STRING_ID_INVALID,
      .attribute_count = 0,
  };

  uint16_t id1 = 0, id2 = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &id1));
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &id2));
  EXPECT_EQ(id1, id2);
  EXPECT_EQ(module->encodings.count, 1u);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingDedupStructuralParamsAndBackfillsAlias) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_string_id_t shape_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("shape"), &shape_id));
  loom_string_id_t alias_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("enc"), &alias_id));

  int64_t shape_a[] = {16, 32};
  int64_t shape_b[] = {16, 32};
  loom_named_attr_t attrs_a[] = {{
      .name_id = shape_id,
      .value = loom_attr_i64_array(shape_a, IREE_ARRAYSIZE(shape_a)),
  }};
  loom_named_attr_t attrs_b[] = {{
      .name_id = shape_id,
      .value = loom_attr_i64_array(shape_b, IREE_ARRAYSIZE(shape_b)),
  }};

  loom_encoding_t plain = {
      .name_id = name_id,
      .alias_id = LOOM_STRING_ID_INVALID,
      .attribute_count = 1,
      .attributes = attrs_a,
  };
  loom_encoding_t aliased = {
      .name_id = name_id,
      .alias_id = alias_id,
      .attribute_count = 1,
      .attributes = attrs_b,
  };

  uint16_t plain_id = 0;
  uint16_t aliased_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &plain, &plain_id));
  IREE_ASSERT_OK(loom_module_add_encoding(module, &aliased, &aliased_id));

  EXPECT_EQ(plain_id, aliased_id);
  ASSERT_EQ(module->encodings.count, 1u);
  const loom_encoding_t* encoding = loom_module_encoding(module, plain_id);
  ASSERT_NE(encoding, nullptr);
  EXPECT_EQ(encoding->alias_id, alias_id);
  ASSERT_EQ(encoding->attribute_count, 1u);
  EXPECT_EQ(encoding->attributes[0].value.count, 2u);
  ASSERT_NE(encoding->attributes[0].value.i64_array, nullptr);
  EXPECT_EQ(encoding->attributes[0].value.i64_array[0], 16);
  EXPECT_EQ(encoding->attributes[0].value.i64_array[1], 32);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingRejectsDuplicateAliasForDifferentEncodings) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t q8_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("q8_0"), &q8_name_id));
  loom_string_id_t dense_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("dense"), &dense_name_id));
  loom_string_id_t alias_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("enc"), &alias_id));

  loom_encoding_t q8_encoding = {
      .name_id = q8_name_id,
      .alias_id = alias_id,
  };
  uint16_t q8_encoding_id = 0;
  IREE_ASSERT_OK(
      loom_module_add_encoding(module, &q8_encoding, &q8_encoding_id));

  loom_encoding_t dense_encoding = {
      .name_id = dense_name_id,
      .alias_id = alias_id,
  };
  uint16_t dense_encoding_id = 0;
  iree_status_t status =
      loom_module_add_encoding(module, &dense_encoding, &dense_encoding_id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(module->encodings.count, 1u);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingDifferentParams) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_string_id_t block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("block"), &block_id));

  // Same name, different block size — two distinct entries.
  loom_named_attr_t param32 = {.name_id = block_id, .value = loom_attr_i64(32)};
  loom_named_attr_t param64 = {.name_id = block_id, .value = loom_attr_i64(64)};

  loom_encoding_t enc32 = {
      .name_id = name_id,
      .alias_id = LOOM_STRING_ID_INVALID,
      .attribute_count = 1,
      .attributes = &param32,
  };
  loom_encoding_t enc64 = {
      .name_id = name_id,
      .alias_id = LOOM_STRING_ID_INVALID,
      .attribute_count = 1,
      .attributes = &param64,
  };

  uint16_t id32 = 0, id64 = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &enc32, &id32));
  IREE_ASSERT_OK(loom_module_add_encoding(module, &enc64, &id64));
  EXPECT_NE(id32, id64);
  EXPECT_EQ(module->encodings.count, 2u);

  loom_module_free(module);
}

TEST_F(ModuleTest, AddEncodingRejectsUnknownFamilyWhenRegistryIsPopulated) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t unknown_name_id = 0;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("mystery_q"),
                                           &unknown_name_id));
  loom_encoding_t encoding = {
      .name_id = unknown_name_id,
      .alias_id = LOOM_STRING_ID_INVALID,
  };
  uint16_t encoding_id = 0;
  iree_status_t status =
      loom_module_add_encoding(module, &encoding, &encoding_id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(module->encodings.count, 0u);

  loom_module_free(module);
}

TEST_F(ModuleTest, EncodingVtableLookupReturnsRegisteredFamily) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("q8_0"), &name_id));
  loom_encoding_t encoding = {
      .name_id = name_id,
      .alias_id = LOOM_STRING_ID_INVALID,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module, &encoding, &encoding_id));

  EXPECT_EQ(loom_module_encoding_vtable(module, encoding_id),
            &kQ8_0EncodingVtable);

  loom_module_free(module);
}

TEST_F(ModuleTest, EncodingLookupOutOfRange) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  EXPECT_EQ(loom_module_encoding(module, 0), nullptr);
  EXPECT_EQ(loom_module_encoding(module, 1), nullptr);
  EXPECT_EQ(loom_module_encoding(module, UINT16_MAX), nullptr);
  EXPECT_EQ(loom_module_encoding_vtable(module, 0), nullptr);
  EXPECT_EQ(loom_module_encoding_vtable(module, 1), nullptr);
  EXPECT_EQ(loom_module_encoding_vtable(module, UINT16_MAX), nullptr);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
