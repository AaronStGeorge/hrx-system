// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_PIPELINE_CONTRACTS_H_
#define LOOM_IMPORT_CXX_BINDING_PIPELINE_CONTRACTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/signature.h"

namespace loom::cxx_import {

class Types;

// Owns pipeline declarations and definitions used as launchable symbols. The
// first supported product carrier is kernel scope; one retained staged
// signature governs declarations, definitions, composition, and launches.
class PipelineContracts {
 public:
  PipelineContracts(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Reconciles the pipeline annotation and launch boundary on one declaration.
  void declaration(cxx::FunctionSymbol* function,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::FunctionDeclaratorChunkAST* prototype, cxx::AST* owner);

  // Completes cross-declaration admission after the translation unit walk.
  void finish();

  // Whether the canonical source function is a kernel-scoped pipeline.
  bool contains(cxx::FunctionSymbol* function) const;

  // Returns the retained source/native staged signature projection.
  const StagedSignatureProjection& project(cxx::FunctionSymbol* function,
                                           Types& types, cxx::AST* owner);

  // Binds the projected signature to the next pipeline.decl build.
  BoundStagedSignature bind(cxx::FunctionSymbol* function, Types& types,
                            cxx::AST* owner, loom_builder_t* builder,
                            SignatureIdentityRequirement identities =
                                SignatureIdentityRequirement::DependentTypes);

 private:
  struct Contract {
    // First pipeline annotation owning cross-declaration diagnostics.
    cxx::AttributeAST* source = nullptr;
    // First launch marker owning orphan/conflict diagnostics.
    cxx::AttributeAST* launch_source = nullptr;
    // Source boundary declared by launch_source; absent means no bindings.
    std::optional<size_t> launch_parameter_ordinal;
    // Best parameter source ranges, preferring a later redeclaration.
    std::vector<cxx::ParameterDeclarationAST*> parameter_sources;
    // Final immutable projection, constructed on first signature use.
    std::unique_ptr<StagedSignatureProjection> projection;
  };

  cxx::AST* parameter_source(const Contract& contract, size_t ordinal,
                             cxx::AST* fallback) const;

  // Invocation-owned frontend and canonical symbols.
  cxx::TranslationUnit& unit_;
  // Source admission diagnostic boundary.
  Diagnostics& diagnostics_;
  // Kernel-scoped pipeline contracts occupy the index.
  std::unordered_map<cxx::FunctionSymbol*, Contract> contracts_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_PIPELINE_CONTRACTS_H_
