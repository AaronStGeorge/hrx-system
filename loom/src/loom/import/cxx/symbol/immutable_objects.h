// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SYMBOL_IMMUTABLE_OBJECTS_H_
#define LOOM_IMPORT_CXX_SYMBOL_IMMUTABLE_OBJECTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/ir/module.h"

namespace loom::cxx_import {

// One reached source-owned immutable object. Its symbol and source layout are
// stable for the translation-unit lifetime; SSA roots remain function-local.
struct ImmutableObject {
  // Private symbol naming the emitted read-only payload.
  loom_symbol_ref_t symbol;
  // Stable module-owned spelling shared with the private symbol.
  iree_string_view_t name;
  // Complete source object type represented by the payload bytes.
  const cxx::Type* type;
  // Source ABI alignment, including an explicit stronger declaration.
  int64_t alignment;
};

// Owns immutable C++ object identity and payloads for one output module.
// Objects are admitted lazily from reached expressions, keyed by canonical
// source symbol, and emitted after callable translation has completed.
class ImmutableObjects {
 public:
  ImmutableObjects(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                   Locations& locations, SymbolNames& names,
                   loom_module_t* module)
      : unit_(unit),
        diagnostics_(diagnostics),
        locations_(locations),
        names_(names),
        module_(module) {}

  // Admits or reuses one namespace-scope internal-linkage constant object.
  // Unsupported lifetime, identity, mutability, or representation semantics
  // diagnose at the source use that reached the object.
  ImmutableObject reference(cxx::VariableSymbol* variable, cxx::AST* owner);

  // Emits every reached object definition into the module insertion block.
  void build(loom_builder_t* builder);

 private:
  struct Definition {
    // Public reference returned to function translators.
    ImmutableObject object;
    // Concrete selected-source-ABI representation.
    std::vector<uint8_t> contents;
    // First source use responsible for retaining this definition.
    cxx::AST* source;
  };

  // Source frontend and data model used for semantic admission and layout.
  cxx::TranslationUnit& unit_;
  // Public source rejection boundary.
  Diagnostics& diagnostics_;
  // Retains definition provenance in the output module.
  Locations& locations_;
  // Shared module namespace for functions, configs, and objects.
  SymbolNames& names_;
  // Output symbol and payload owner.
  loom_module_t* module_;
  // Canonical source identity to retained definition ordinal.
  std::unordered_map<cxx::VariableSymbol*, size_t> variables_;
  // Encounter-ordered reached definitions.
  std::vector<Definition> definitions_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SYMBOL_IMMUTABLE_OBJECTS_H_
