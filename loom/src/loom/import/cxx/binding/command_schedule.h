// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_COMMAND_SCHEDULE_H_
#define LOOM_IMPORT_CXX_BINDING_COMMAND_SCHEDULE_H_

#include <cxx/ast_fwd.h>

#include <optional>

namespace cxx {
class TranslationUnit;
}

namespace loom::cxx_import {

class Diagnostics;

// One attributed compound statement projected directly to a resultless
// command scheduling region. The source braces remain the lexical scope; the
// attribute only selects the region's sibling dependency contract.
struct CommandSchedule {
  enum class Kind { Serial, Concurrent };

  static std::optional<CommandSchedule> resolve(cxx::TranslationUnit& unit,
                                                Diagnostics& diagnostics,
                                                cxx::StatementAST* statement);

  // Selected schedule dependency contract.
  Kind kind;
  // Compound statement supplying the region body.
  cxx::CompoundStatementAST* body;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_COMMAND_SCHEDULE_H_
