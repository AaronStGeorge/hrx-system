// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"

#include <string.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
  }

  void TearDown() override { loom_context_deinitialize(&context_); }

  loom_context_t context_;
};

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    .name = IREE_SV("q8_0"),
};

TEST_F(ContextTest, FinalizeBuildsOpNameLookupTable) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      .name = kTestOpName,
  };
  static const loom_op_vtable_t* const kTestDialectVtables[] = {
      &kTestOpVtable,
  };

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));
  IREE_ASSERT_OK(loom_context_finalize(&context_));

  loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
  const loom_op_vtable_t* vtable =
      loom_context_lookup_op_by_name(&context_, IREE_SV("test.nop"), &kind);
  ASSERT_EQ(vtable, &kTestOpVtable);
  EXPECT_EQ(kind, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0));
  EXPECT_EQ(loom_context_resolve_op(&context_, kind), &kTestOpVtable);
}

TEST_F(ContextTest, RegisterDialectSemanticsResolvesByOpKind) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      .name = kTestOpName,
  };
  static const loom_op_vtable_t* const kTestDialectVtables[] = {
      &kTestOpVtable,
  };
  static const loom_op_semantics_t kTestDialectSemantics[] = {
      {
          .phase = LOOM_OP_PHASE_EXECUTABLE,
          .contract_families = LOOM_CONTRACT_VECTOR_CONTRACTION,
      },
  };

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));
  IREE_ASSERT_OK(loom_context_register_dialect_semantics(
      &context_, LOOM_DIALECT_TEST, kTestDialectSemantics,
      IREE_ARRAYSIZE(kTestDialectSemantics)));

  loom_op_semantics_t semantics = loom_context_resolve_op_semantics(
      &context_, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0));
  EXPECT_EQ(semantics.phase, LOOM_OP_PHASE_EXECUTABLE);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      semantics.contract_families, LOOM_CONTRACT_VECTOR_CONTRACTION));

  loom_op_semantics_t missing = loom_context_resolve_op_semantics(
      &context_, LOOM_OP_KIND(LOOM_DIALECT_TEST, 1));
  EXPECT_EQ(missing.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(missing.contract_families, 0u);
}

TEST_F(ContextTest, RegisterDialectSemanticsRequiresMatchingVtables) {
  static const uint8_t kTestOpName[] = {
      8, 4, 't', 'e', 's', 't', '.', 'n', 'o', 'p', '\0',
  };
  static const loom_op_vtable_t kTestOpVtable = {
      .name = kTestOpName,
  };
  static const loom_op_vtable_t* const kTestDialectVtables[] = {
      &kTestOpVtable,
  };
  static const loom_op_semantics_t kTestDialectSemantics[] = {
      {
          .phase = LOOM_OP_PHASE_EXECUTABLE,
      },
  };

  iree_status_t status = loom_context_register_dialect_semantics(
      &context_, LOOM_DIALECT_TEST, kTestDialectSemantics,
      IREE_ARRAYSIZE(kTestDialectSemantics));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);

  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, kTestDialectVtables,
      IREE_ARRAYSIZE(kTestDialectVtables)));

  status = loom_context_register_dialect_semantics(&context_, LOOM_DIALECT_TEST,
                                                   kTestDialectSemantics, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);
}

TEST_F(ContextTest, RegisterSourceDeduplicatesBySpelling) {
  loom_source_id_t first_id = LOOM_SOURCE_ID_INVALID;
  loom_source_id_t second_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_context_register_source(&context_, IREE_SV("model.loom"),
                                              &first_id));
  IREE_ASSERT_OK(loom_context_register_source(&context_, IREE_SV("model.loom"),
                                              &second_id));

  EXPECT_EQ(first_id, 0u);
  EXPECT_EQ(second_id, first_id);
  ASSERT_EQ(context_.sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(context_.sources.entries[0],
                                     IREE_SV("model.loom")));
}

TEST_F(ContextTest, RegisterEmptySource) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_context_register_source(
      &context_, iree_string_view_empty(), &source_id));

  EXPECT_EQ(source_id, 0u);
  ASSERT_EQ(context_.sources.count, 1u);
  EXPECT_EQ(context_.sources.entries[0].size, 0u);
}

TEST_F(ContextTest, RegisterSourceRejectsInvalidSentinelId) {
  iree_string_view_t* entries = NULL;
  IREE_ASSERT_OK(
      iree_allocator_malloc_array(context_.allocator, LOOM_SOURCE_ID_INVALID,
                                  sizeof(*entries), (void**)&entries));
  memset(entries, 0,
         (iree_host_size_t)LOOM_SOURCE_ID_INVALID * sizeof(*entries));
  context_.sources.entries = entries;
  context_.sources.capacity = LOOM_SOURCE_ID_INVALID;
  context_.sources.count = LOOM_SOURCE_ID_INVALID;

  loom_source_id_t source_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_context_register_source(&context_, IREE_SV("overflow"), &source_id));
}

TEST_F(ContextTest, RegisterEncodingVtableAndLookupByName) {
  IREE_ASSERT_OK(
      loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));

  EXPECT_EQ(loom_context_lookup_encoding_vtable(&context_, IREE_SV("q8_0")),
            &kQ8_0EncodingVtable);
  EXPECT_EQ(loom_context_lookup_encoding_vtable(&context_, IREE_SV("q6_k")),
            nullptr);
}

TEST_F(ContextTest, RegisterEncodingVtableRejectsDuplicateName) {
  IREE_ASSERT_OK(
      loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));

  static const loom_encoding_vtable_t kDuplicate = {
      .name = IREE_SV("q8_0"),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_context_register_encoding_vtable(&context_, &kDuplicate));
}

TEST_F(ContextTest, RegisterEncodingVtableRejectsMissingName) {
  static const loom_encoding_vtable_t kMissingName = {};
  iree_status_t status =
      loom_context_register_encoding_vtable(&context_, &kMissingName);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  status = loom_context_register_encoding_vtable(&context_, NULL);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

}  // namespace
}  // namespace loom
