// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/tracing.h"
#include "iree/testing/gtest.h"

namespace {

TEST(DisabledTracingTest, SectionsDoNotEvaluateArguments) {
  int calls = 0;
  const iree_section_id_t literal =
      IREE_TRACE_SECTION_ENTER(++calls, "literal");
  const iree_section_id_t dynamic =
      IREE_TRACE_SECTION_ENTER_DYNAMIC(++calls, (++calls, "name"), ++calls);
  const iree_section_id_t tagged = IREE_TRACE_SECTION_ENTER_TAGGED(
      ++calls, "label", (++calls, "tag"), ++calls);
  IREE_TRACE_SECTION_SETUP(++calls, "category");
  IREE_TRACE_SECTION_SETUP_DYNAMIC(++calls, (++calls, "category"), ++calls);
  IREE_TRACE_SECTION_LEAVE(++calls);
  EXPECT_EQ(literal, 0u);
  EXPECT_EQ(dynamic, 0u);
  EXPECT_EQ(tagged, 0u);
  EXPECT_EQ(calls, 0);
}

TEST(DisabledTracingTest, AllocationCoordinationIsErased) {
  IREE_TRACE_ALLOC_WAIT_FOR_REALLOCATIONS();
  IREE_TRACE_REALLOC_BEGIN();
  IREE_TRACE_REALLOC_END();
}

}  // namespace
