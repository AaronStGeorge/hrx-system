// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/immutable_objects.h"

#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/global/ops.h"

namespace loom::cxx_import {
namespace {

bool contains_mutable_subobject(cxx::TranslationUnit& unit,
                                const cxx::Type* type) {
  auto traits = unit.typeTraits();
  auto* element = traits.remove_cv(traits.remove_all_extents(type));
  auto* class_type = cxx::type_cast<cxx::ClassType>(element);
  if (!class_type) {
    return false;
  }
  auto* definition = class_type->definition();
  traits.requireCompleteClass(definition);
  if (!definition || !definition->isComplete()) {
    return true;
  }
  for (auto* member : traits.aggregate_elements(definition)) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member);
    if (field && field->isMutable()) {
      return true;
    }
    if (contains_mutable_subobject(unit,
                                   traits.aggregate_element_type(member))) {
      return true;
    }
  }
  return false;
}

std::string object_name(cxx::VariableSymbol* variable) {
  std::string spelling = cxx::to_string(variable->name());
  for (auto* owner : variable->enclosingSymbols()) {
    auto* space = cxx::symbol_cast<cxx::NamespaceSymbol>(owner);
    if (space && space->name()) {
      spelling = cxx::to_string(space->name()) + "_" + spelling;
    }
  }
  for (char& character : spelling) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '_') {
      character = '_';
    }
  }
  return spelling;
}

}  // namespace

ImmutableObject ImmutableObjects::reference(cxx::VariableSymbol* variable,
                                            cxx::AST* owner) {
  auto* canonical = variable->canonical();
  if (auto found = variables_.find(canonical); found != variables_.end()) {
    return definitions_[found->second].object;
  }
  auto* definition = canonical->resolvedDefinition();
  auto traits = unit_.typeTraits();
  auto* source_type = definition->type();
  if (!cxx::symbol_cast<cxx::NamespaceSymbol>(definition->parent()) ||
      definition->isThreadLocal()) {
    diagnostics_.reject(
        unit_, owner,
        "immutable storage requires a namespace-scope object without "
        "thread-local storage");
  }
  if (!cxx::has_internal_linkage(canonical)) {
    diagnostics_.reject(
        unit_, owner, "immutable storage requires an internal-linkage object");
  }
  if (!traits.is_const(source_type) || traits.is_volatile(source_type) ||
      contains_mutable_subobject(unit_, source_type)) {
    diagnostics_.reject(
        unit_, owner,
        "immutable storage requires a nonvolatile deeply const object");
  }
  if (!definition->constValue()) {
    diagnostics_.reject(unit_, owner,
                        "immutable storage requires constant initialization");
  }
  cxx::ASTInterpreter interpreter(&unit_);
  auto contents =
      interpreter.objectRepresentation(*definition->constValue(), source_type);
  if (!contents) {
    diagnostics_.reject(
        unit_, owner,
        "immutable storage requires a fully initialized, trivially copyable, "
        "address-free object representation");
  }
  auto natural_alignment =
      unit_.control()->memoryLayout()->alignmentOf(source_type);
  if (!natural_alignment) {
    diagnostics_.reject(unit_, owner, "unknown immutable object alignment");
  }
  auto spelling = names_.unique(object_name(definition));
  loom_string_id_t name;
  check(loom_module_intern_string(module_, view(spelling), &name));
  loom_symbol_id_t symbol;
  check(loom_module_add_symbol(module_, name, &symbol));
  ImmutableObject object = {
      .symbol = {0, symbol},
      .type = source_type,
      .alignment = std::max<int64_t>(*natural_alignment,
                                     definition->explicitAlignment()),
  };
  variables_.emplace(canonical, definitions_.size());
  definitions_.push_back(
      {.object = object, .contents = std::move(*contents), .source = owner});
  return object;
}

void ImmutableObjects::build(loom_builder_t* builder) {
  for (const auto& definition : definitions_) {
    auto flags = definition.object.alignment > 1
                     ? LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_ALIGNMENT
                     : 0;
    loom_op_t* op;
    check(loom_global_rodata_def_build(
        builder, flags, definition.object.symbol, definition.object.alignment,
        loom_symbol_ref_array_empty(),
        iree_make_const_byte_span(definition.contents.data(),
                                  definition.contents.size()),
        locations_.get(definition.source), &op));
  }
}

}  // namespace loom::cxx_import
