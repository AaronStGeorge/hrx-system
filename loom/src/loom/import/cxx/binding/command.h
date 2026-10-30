// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_COMMAND_H_
#define LOOM_IMPORT_CXX_BINDING_COMMAND_H_

#include <cxx/attributes.h>

#include <optional>
#include <string_view>

#include "loom/import/cxx/binding/command_programs.h"

namespace loom::cxx_import {

class Types;
struct Partition;

// Explicit invocation of one statically selected command program. The retained
// projection is the same immutable mapping used to define the callee, so call
// emission never rediscovers which shaped components are specializations or
// launch bindings.
struct CommandLaunchIntrinsic {
  static bool supports(std::string_view name);
  static std::optional<CommandLaunchIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      CommandPrograms& programs, cxx::FunctionSymbol* function,
      const cxx::Attribute& attribute, cxx::AST* owner);

  bool equivalent(const CommandLaunchIntrinsic& other) const {
    return program == other.program && projection == other.projection;
  }

  // Statically selected command declaration or definition.
  cxx::FunctionSymbol* program;
  // Callee-owned source/native signature projection.
  const CommandProgramProjection* projection;
};

// Typed named-parameter lookup and parameter-relative buffer recovery.
// Parameter result dependencies are explicit source operands before the key
// pattern, allowing dynamic extents and encodings to bind the result view
// without hiding type reconstruction in the importer.
struct CommandParameterIntrinsic {
  enum class Operation { Parameter, Base };

  static bool supports(std::string_view name);
  static std::optional<CommandParameterIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  bool equivalent(const CommandParameterIntrinsic& other) const {
    return operation == other.operation && result == other.result &&
           source == other.source &&
           binding_component_count == other.binding_component_count;
  }

  // Selected command dialect operation.
  Operation operation;
  // Admitted result partition.
  const Partition* result;
  // Admitted source partition.
  const Partition* source;
  // Dynamic extent and encoding operands preceding a parameter key.
  size_t binding_component_count;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_COMMAND_H_
