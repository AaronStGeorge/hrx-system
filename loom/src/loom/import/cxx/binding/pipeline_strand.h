// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_PIPELINE_STRAND_H_
#define LOOM_IMPORT_CXX_BINDING_PIPELINE_STRAND_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <optional>

#include "loom/import/cxx/binding/index_expression.h"

namespace loom::cxx_import {

class Diagnostics;
class Types;

// One attributed compound statement projected to a pipeline strand. Worker
// domains retain static list entries and dynamic expressions separately, and
// an optional target keeps the canonical constexpr object identity.
struct PipelineStrand {
  static std::optional<PipelineStrand> resolve(cxx::TranslationUnit& unit,
                                               Diagnostics& diagnostics,
                                               Types& types,
                                               cxx::StatementAST* statement);

  cxx::CompoundStatementAST* body;
  cxx::AST* source;
  IndexList origins;
  IndexList counts;
  IndexList strides;
  cxx::VariableSymbol* target;
  cxx::AST* target_source;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_PIPELINE_STRAND_H_
