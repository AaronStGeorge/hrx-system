// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/pipeline_strand.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>

#include <array>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"

namespace loom::cxx_import {

std::optional<PipelineStrand> PipelineStrand::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::StatementAST* statement) {
  cxx::AttributeAST* strand_source = nullptr;
  cxx::AttributeAST* target_source = nullptr;
  std::array<cxx::ExpressionAST*, 3> axes = {};
  cxx::VariableSymbol* target = nullptr;
  visit_loom_attributes(
      unit, statement_attributes(statement),
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name == "strand") {
          auto* clause = attribute->attributeArgumentClause;
          auto* arguments = clause ? clause->expressionList : nullptr;
          if (!arguments || !arguments->next || !arguments->next->next ||
              arguments->next->next->next) {
            diagnostics.reject(
                unit, attribute,
                "strand requires origin, count, and stride index lists");
          }
          axes = {arguments->value, arguments->next->value,
                  arguments->next->next->value};
          strand_source = attribute;
        } else if (name == "target") {
          auto* clause = attribute->attributeArgumentClause;
          auto* arguments = clause ? clause->expressionList : nullptr;
          auto* identity = arguments && !arguments->next
                               ? cxx::ast_cast<cxx::IdExpressionAST>(
                                     unwrap_expression(arguments->value))
                               : nullptr;
          target = identity
                       ? cxx::symbol_cast<cxx::VariableSymbol>(identity->symbol)
                       : nullptr;
          if (!target) {
            diagnostics.reject(unit, attribute,
                               "strand target requires one target object name");
          }
          target_source = attribute;
        }
      });
  if (!strand_source) {
    return std::nullopt;
  }
  auto* body = cxx::ast_cast<cxx::CompoundStatementAST>(statement);
  if (!body) {
    diagnostics.reject(unit, strand_source,
                       "strand requires an attributed compound statement");
  }
  return PipelineStrand{
      body,
      strand_source,
      IndexList::admit(unit, diagnostics, types, axes[0], "strand origin"),
      IndexList::admit(unit, diagnostics, types, axes[1], "strand count"),
      IndexList::admit(unit, diagnostics, types, axes[2], "strand stride"),
      target ? target->canonical() : nullptr,
      target_source,
  };
}

}  // namespace loom::cxx_import
