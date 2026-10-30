// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_INDEX_EXPRESSION_H_
#define LOOM_IMPORT_CXX_BINDING_INDEX_EXPRESSION_H_

#include <cxx/ast_fwd.h>

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "loom/import/cxx/source/source.h"
#include "loom/ir/ir.h"

namespace loom::cxx_import {

class Types;

// Admits one pure source integer or Loom index expression and projects an
// evaluated value to the target-independent index domain. The source program
// owns arithmetic and evaluation; annotations only observe its result.
class IndexExpression {
 public:
  static cxx::ExpressionAST* admit(cxx::TranslationUnit& unit,
                                   Diagnostics& diagnostics, Types& types,
                                   cxx::ExpressionAST* expression,
                                   std::string_view purpose);

  static loom_value_id_t build(cxx::TranslationUnit& unit,
                               loom_builder_t* builder,
                               cxx::ExpressionAST* expression,
                               loom_value_id_t value,
                               loom_location_id_t location);
};

// One direct loom::type::indices{...} source construction split into its
// operation-level static list and dynamic SSA operands. Each source expression
// is admitted once, and consumers evaluate only dynamic elements.
class IndexList {
 public:
  static bool is_type(cxx::TranslationUnit& unit, const cxx::Type* type);

  static IndexList admit(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         Types& types, cxx::ExpressionAST* expression,
                         std::string_view purpose);

  std::span<cxx::ExpressionAST* const> dynamic_expressions() const {
    return dynamic_expressions_;
  }
  std::span<const int64_t> static_values() const { return static_values_; }

 private:
  std::vector<cxx::ExpressionAST*> dynamic_expressions_;
  std::vector<int64_t> static_values_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_INDEX_EXPRESSION_H_
