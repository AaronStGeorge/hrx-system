// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_COMMAND_PROGRAMS_H_
#define LOOM_IMPORT_CXX_BINDING_COMMAND_PROGRAMS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class Types;

// Retained placement of one source parameter in a command program's native
// signature. Source specializations remain contiguous. A launch-bound view
// contributes its dependent extent and encoding components to the
// specialization prefix and its final handle to the binding suffix.
struct CommandParameterProjection {
  // Offset in source-component order across all source parameters.
  size_t source_component_offset;
  // Total flattened source components reconstructed for the C++ parameter.
  size_t component_count;
  // First native specialization component contributed by this parameter.
  size_t specialization_offset;
  // Number of leading source components transported as specializations.
  size_t specialization_count;
  // Native launch-binding ordinal, absent for a source specialization.
  std::optional<size_t> binding_ordinal;
};

// Immutable source-to-native signature projection for one command program.
// Consumers use these offsets directly; no body or call site rediscovers the
// launch boundary or inspects shaped types to recover component placement.
struct CommandProgramProjection {
  // Source ordinal of the first launch-bound parameter.
  size_t launch_parameter_ordinal;
  // Flattened native specialization count before the launch boundary.
  size_t specialization_count;
  // One native binding for each source parameter at or after the boundary.
  size_t binding_count;
  // Whether any native component type references another component identity.
  bool requires_binding;
  // Canonical unqualified source types in source parameter order.
  std::vector<const cxx::Type*> source_types;
  // Source parameter order, retained independently of native argument order.
  std::vector<CommandParameterProjection> parameters;
};

// Types and reserved identities consumed by one command definition or
// declaration builder. Destination identities follow native order while
// source identities follow flattened C++ parameter order for predicates.
struct BoundCommandProgramSignature {
  BoundCommandProgramSignature() = default;
  BoundCommandProgramSignature(const BoundCommandProgramSignature&) = delete;
  BoundCommandProgramSignature& operator=(const BoundCommandProgramSignature&) =
      delete;
  BoundCommandProgramSignature(BoundCommandProgramSignature&&) = default;
  BoundCommandProgramSignature& operator=(BoundCommandProgramSignature&&) =
      default;

  // Retained projection shared with body and call translation.
  const CommandProgramProjection* projection = nullptr;
  // Native specialization types in command signature order.
  std::vector<loom_type_t> specialization_types;
  // Native launch-binding types in command signature order.
  std::vector<loom_type_t> binding_types;
  // Builder-reserved identities in specialization-then-binding order.
  std::vector<loom_value_id_t> destination_identities;
  // The same identities permuted into flattened source parameter order.
  std::vector<loom_value_id_t> source_identities;
  // Overflow type payloads retained through the owning builder call.
  BoundTypeStorage type_storage;
};

// Owns command-program source admission and the single retained projection
// consumed by definitions, declarations, body binding, and explicit launches.
class CommandPrograms {
 public:
  CommandPrograms(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Reconciles the command annotation and launch boundary on one declaration.
  void declaration(cxx::FunctionSymbol* function,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::FunctionDeclaratorChunkAST* prototype, cxx::AST* owner);

  // Completes cross-declaration admission after the translation unit walk.
  void finish();

  // Whether the canonical source function is an admitted command program.
  bool contains(cxx::FunctionSymbol* function) const;

  // Returns the retained projection, admitting source component types once.
  const CommandProgramProjection& project(cxx::FunctionSymbol* function,
                                          Types& types, cxx::AST* owner);

  // Binds the projected native signature to the next command op build.
  BoundCommandProgramSignature bind(
      cxx::FunctionSymbol* function, Types& types, cxx::AST* owner,
      loom_builder_t* builder,
      SignatureIdentityRequirement identities =
          SignatureIdentityRequirement::DependentTypes);

 private:
  struct Contract {
    // First command_program annotation owning cross-declaration diagnostics.
    cxx::AttributeAST* source = nullptr;
    // First launch marker owning orphan/conflict diagnostics.
    cxx::AttributeAST* launch_source = nullptr;
    // Source boundary declared by launch_source; absent means no bindings.
    std::optional<size_t> launch_parameter_ordinal;
    // Best parameter source ranges, preferring the concrete definition.
    std::vector<cxx::ParameterDeclarationAST*> parameter_sources;
    // Final immutable projection, constructed on first signature use.
    std::unique_ptr<CommandProgramProjection> projection;
  };

  cxx::AST* parameter_source(const Contract& contract, size_t ordinal,
                             cxx::AST* fallback) const;

  // Invocation-owned frontend and canonical symbols.
  cxx::TranslationUnit& unit_;
  // Source admission diagnostic boundary.
  Diagnostics& diagnostics_;
  // Only command-program declarations occupy the index.
  std::unordered_map<cxx::FunctionSymbol*, Contract> contracts_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_COMMAND_PROGRAMS_H_
