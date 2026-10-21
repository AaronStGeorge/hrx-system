// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/immutable_objects.h"

#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include <algorithm>
#include <array>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/global/ops.h"

namespace loom::cxx_import {
namespace {
using ImmutableObjectsTest = ValueBuilderTest;

cxx::VariableSymbol* variable(cxx::TranslationUnit& unit, const char* name) {
  auto symbols = unit.globalScope()->find(name);
  if (symbols.begin() == symbols.end()) {
    return nullptr;
  }
  return cxx::symbol_cast<cxx::VariableSymbol>(*symbols.begin());
}

TEST_F(ImmutableObjectsTest, RetainsEachReachedSourceObjectOnce) {
  Source source(IREE_SV(R"(
    namespace detail {
    static constexpr unsigned table[4] = {11, 22, 33, 44};
    }
  )"),
                IREE_SV("immutable_objects.cxx"), options());
  Locations locations(source.unit(), source.diagnostics(), module_);
  SymbolNames names(source.unit(), source.diagnostics());
  ImmutableObjects objects(source.unit(), source.diagnostics(), locations,
                           names, module_);
  auto* detail = cxx::symbol_cast<cxx::NamespaceSymbol>(
      *source.unit().globalScope()->find("detail").begin());
  ASSERT_NE(detail, nullptr);
  auto* table =
      cxx::symbol_cast<cxx::VariableSymbol>(*detail->find("table").begin());
  ASSERT_NE(table, nullptr);

  auto first = objects.reference(table, table->initializer());
  auto second = objects.reference(table, table->initializer());
  EXPECT_EQ(first.symbol.symbol_id, second.symbol.symbol_id);
  EXPECT_EQ(first.type, table->type());
  EXPECT_EQ(first.alignment, 4);
  objects.build(&builder_);

  auto* definition = loom_module_block(module_)->first_op;
  ASSERT_NE(definition, nullptr);
  ASSERT_TRUE(loom_global_rodata_def_isa(definition));
  EXPECT_EQ(loom_global_rodata_def_symbol(definition).symbol_id,
            first.symbol.symbol_id);
  EXPECT_EQ(loom_global_rodata_def_alignment(definition), 4);
  auto contents = loom_global_rodata_def_contents(definition);
  constexpr std::array<uint8_t, 16> expected = {11, 0, 0, 0, 22, 0, 0, 0,
                                                33, 0, 0, 0, 44, 0, 0, 0};
  ASSERT_EQ(contents.data_length, expected.size());
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), contents.data));
  EXPECT_EQ(definition->next_op, nullptr);
}

TEST_F(ImmutableObjectsTest, RejectsUnsupportedIdentityAndRepresentation) {
  Source source(IREE_SV(R"(
    struct Inner { mutable unsigned value; };
    struct Outer { Inner inner; };
    inline constexpr unsigned external_identity[1] = {1};
    static constexpr Outer mutable_object{{2}};
    static constexpr const unsigned* address_object = nullptr;
  )"),
                IREE_SV("invalid_immutable_objects.cxx"), options());
  Locations locations(source.unit(), source.diagnostics(), module_);
  SymbolNames names(source.unit(), source.diagnostics());
  ImmutableObjects objects(source.unit(), source.diagnostics(), locations,
                           names, module_);

  auto* external_identity = variable(source.unit(), "external_identity");
  auto* mutable_object = variable(source.unit(), "mutable_object");
  auto* address_object = variable(source.unit(), "address_object");
  ASSERT_NE(external_identity, nullptr);
  ASSERT_NE(mutable_object, nullptr);
  ASSERT_NE(address_object, nullptr);
  EXPECT_THROW(
      objects.reference(external_identity, external_identity->initializer()),
      SourceRejected);
  EXPECT_THROW(objects.reference(mutable_object, mutable_object->initializer()),
               SourceRejected);
  EXPECT_THROW(objects.reference(address_object, address_object->initializer()),
               SourceRejected);
}

}  // namespace
}  // namespace loom::cxx_import
