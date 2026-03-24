// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/type_registry.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Op registry
//===----------------------------------------------------------------------===//

TEST(OpRegistry, LookupKnownOps) {
  loom_op_kind_t kind = 0;
  EXPECT_TRUE(
      loom_op_registry_lookup(iree_make_cstring_view("test.addi"), &kind));
  EXPECT_EQ(kind, LOOM_OP_TEST_ADDI);

  EXPECT_TRUE(
      loom_op_registry_lookup(iree_make_cstring_view("func.def"), &kind));
  EXPECT_EQ(kind, LOOM_OP_FUNC_DEF);
}

TEST(OpRegistry, LookupUnknownReturnsNull) {
  loom_op_kind_t kind = 0xFFFF;
  EXPECT_FALSE(
      loom_op_registry_lookup(iree_make_cstring_view("nonexistent.op"), &kind));
  // kind should be untouched on failure.
  EXPECT_EQ(kind, 0xFFFF);
}

TEST(OpRegistry, LookupEmptyString) {
  loom_op_kind_t kind = 0;
  EXPECT_FALSE(loom_op_registry_lookup(iree_make_cstring_view(""), &kind));
}

TEST(OpRegistry, EntriesAreSorted) {
  const loom_op_registry_entry_t* entries = loom_op_registry_entries();
  iree_host_size_t count = loom_op_registry_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 1; i < count; ++i) {
    EXPECT_LT(iree_string_view_compare(entries[i - 1].name, entries[i].name), 0)
        << "entries[" << i - 1 << "] (\""
        << std::string(entries[i - 1].name.data, entries[i - 1].name.size)
        << "\") >= entries[" << i << "] (\""
        << std::string(entries[i].name.data, entries[i].name.size) << "\")";
  }
}

TEST(OpRegistry, AllEntriesValid) {
  // Verify every entry has a non-empty name and a valid kind, and
  // that lookup by name round-trips to the same kind.
  const loom_op_registry_entry_t* entries = loom_op_registry_entries();
  iree_host_size_t count = loom_op_registry_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 0; i < count; ++i) {
    EXPECT_GT(entries[i].name.size, 0u) << "entry " << i << " has empty name";
    loom_op_kind_t kind = 0;
    EXPECT_TRUE(loom_op_registry_lookup(entries[i].name, &kind))
        << "entry " << i << " (\""
        << std::string(entries[i].name.data, entries[i].name.size)
        << "\") not found by lookup";
    EXPECT_EQ(kind, entries[i].kind)
        << "entry " << i << " kind mismatch after round-trip lookup";
  }
}

//===----------------------------------------------------------------------===//
// Type registry
//===----------------------------------------------------------------------===//

TEST(TypeRegistry, LookupBuiltinTypes) {
  const loom_type_descriptor_t* desc;

  desc = loom_type_registry_lookup(iree_make_cstring_view("tile"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_TILE);
  EXPECT_EQ(desc->param_count, 3);
  EXPECT_NE(desc->format_elements, nullptr);
  EXPECT_GT(desc->format_element_count, 0);

  desc = loom_type_registry_lookup(iree_make_cstring_view("tensor"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_TENSOR);
  EXPECT_EQ(desc->param_count, 3);

  desc = loom_type_registry_lookup(iree_make_cstring_view("pool"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_POOL);
  EXPECT_EQ(desc->param_count, 1);

  desc = loom_type_registry_lookup(iree_make_cstring_view("group"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_GROUP);
  EXPECT_EQ(desc->param_count, 1);
}

TEST(TypeRegistry, LookupDialectType) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(iree_make_cstring_view("hal.buffer"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_DIALECT);
  EXPECT_EQ(desc->param_count, 0);
  EXPECT_EQ(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 0);
}

TEST(TypeRegistry, LookupUnknownReturnsNull) {
  EXPECT_EQ(
      loom_type_registry_lookup(iree_make_cstring_view("nonexistent.type")),
      nullptr);
}

TEST(TypeRegistry, LookupEmptyString) {
  EXPECT_EQ(loom_type_registry_lookup(iree_make_cstring_view("")), nullptr);
}

TEST(TypeRegistry, EntriesAreSorted) {
  const loom_type_registry_entry_t* entries = loom_type_registry_entries();
  iree_host_size_t count = loom_type_registry_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 1; i < count; ++i) {
    EXPECT_LT(iree_string_view_compare(entries[i - 1].name, entries[i].name), 0)
        << "entries[" << i - 1 << "] (\""
        << std::string(entries[i - 1].name.data, entries[i - 1].name.size)
        << "\") >= entries[" << i << "] (\""
        << std::string(entries[i].name.data, entries[i].name.size) << "\")";
  }
}

TEST(TypeRegistry, AllEntriesValid) {
  const loom_type_registry_entry_t* entries = loom_type_registry_entries();
  iree_host_size_t count = loom_type_registry_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 0; i < count; ++i) {
    EXPECT_GT(entries[i].name.size, 0u) << "entry " << i << " has empty name";
    const loom_type_descriptor_t* desc =
        loom_type_registry_lookup(entries[i].name);
    EXPECT_NE(desc, nullptr)
        << "entry " << i << " (\""
        << std::string(entries[i].name.data, entries[i].name.size)
        << "\") not found by lookup";
  }
}

TEST(TypeRegistry, TileFormatElements) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(iree_make_cstring_view("tile"));
  ASSERT_NE(desc, nullptr);
  ASSERT_EQ(desc->format_element_count, 6);
  // ShapeOf, Keyword(x), ScalarOf, Optional, Keyword(,), EncodingOf.
  EXPECT_EQ(desc->format_elements[0].kind, LOOM_TYPE_FMT_SHAPE);
  EXPECT_EQ(desc->format_elements[1].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[1].data, LOOM_KW_X);
  EXPECT_EQ(desc->format_elements[2].kind, LOOM_TYPE_FMT_SCALAR);
  EXPECT_EQ(desc->format_elements[3].kind, LOOM_TYPE_FMT_OPTIONAL);
  EXPECT_EQ(desc->format_elements[4].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[4].data, LOOM_KW_COMMA);
  EXPECT_EQ(desc->format_elements[5].kind, LOOM_TYPE_FMT_ENCODING);
}

}  // namespace
}  // namespace loom
