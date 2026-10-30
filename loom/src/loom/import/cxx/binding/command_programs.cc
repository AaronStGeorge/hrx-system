// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/command_programs.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <string_view>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

void CommandPrograms::declaration(
    cxx::FunctionSymbol* function,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    cxx::FunctionDeclaratorChunkAST* prototype, cxx::AST* owner) {
  if (!function) {
    return;
  }
  cxx::AttributeAST* command_source = nullptr;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "command_program") {
          return;
        }
        if (command_source) {
          diagnostics_.reject(
              unit_, attribute,
              "one command_program annotation is allowed per declaration");
        }
        if (attribute->attributeArgumentClause) {
          diagnostics_.reject(unit_, attribute,
                              "command_program takes no argument clause");
        }
        command_source = attribute;
      });

  auto found = contracts_.find(function->canonical());
  if (!command_source && found == contracts_.end() &&
      !annotated(function, "command_program")) {
    return;
  }

  cxx::AttributeAST* launch_source = nullptr;
  std::optional<size_t> launch_parameter_ordinal;
  std::vector<cxx::ParameterDeclarationAST*> parameter_sources;
  if (prototype && prototype->parameterDeclarationClause) {
    size_t ordinal = 0;
    for (auto* parameter : cxx::ListView{
             prototype->parameterDeclarationClause->parameterDeclarationList}) {
      parameter_sources.push_back(parameter);
      visit_loom_attributes(
          unit_, parameter->attributeList,
          [&](std::string_view name, cxx::AttributeAST* attribute) {
            if (name != "launch") {
              return;
            }
            if (launch_source) {
              diagnostics_.reject(
                  unit_, attribute,
                  "one launch boundary is allowed per command declaration");
            }
            if (attribute->attributeArgumentClause) {
              diagnostics_.reject(unit_, attribute,
                                  "launch takes no argument clause");
            }
            launch_source = attribute;
            launch_parameter_ordinal = ordinal;
          });
      ++ordinal;
    }
  }

  auto& contract = contracts_[function->canonical()];
  if (command_source && !contract.source) {
    contract.source = command_source;
  }
  if (launch_source) {
    if (contract.launch_parameter_ordinal &&
        contract.launch_parameter_ordinal != launch_parameter_ordinal) {
      diagnostics_.reject(
          unit_, launch_source,
          "conflicting launch boundaries across command redeclarations");
    }
    contract.launch_source = launch_source;
    contract.launch_parameter_ordinal = launch_parameter_ordinal;
  }
  if (!parameter_sources.empty() &&
      (contract.parameter_sources.empty() ||
       cxx::ast_cast<cxx::FunctionDefinitionAST>(owner))) {
    contract.parameter_sources = std::move(parameter_sources);
  }
}

void CommandPrograms::finish() {
  for (auto& [function, contract] : contracts_) {
    if (!contract.source) {
      diagnostics_.reject(
          unit_, contract.launch_source,
          "launch parameter boundaries require a command_program function");
    }
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    const bool valid_owner =
        cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent()) &&
        !function->isTemplatePattern() &&
        !function->isImplicitObjectMemberFunction();
    if (!valid_owner || !signature || signature->isVariadic() ||
        !unit_.typeTraits().is_void(signature->returnType())) {
      diagnostics_.reject(
          unit_, contract.source,
          "command_program requires a concrete non-variadic void "
          "namespace-scope function");
    }
    if (annotated(function, "kernel") || annotated(function, "device") ||
        annotated(function, "op") || annotated(function, "assume") ||
        annotated(function, "check_case") ||
        annotated(function, "check_scenario") ||
        annotated(function, "check_benchmark") ||
        annotated(function, "template_def") ||
        annotated(function, "force_inline") || annotated(function, "import")) {
      diagnostics_.reject(
          unit_, contract.source,
          "command_program cannot be combined with another callable kind");
    }
    const size_t parameter_count = function->parameters().size();
    if (contract.launch_parameter_ordinal &&
        *contract.launch_parameter_ordinal >= parameter_count) {
      diagnostics_.reject(unit_, contract.launch_source,
                          "launch boundary must name a source parameter");
    }
  }
}

bool CommandPrograms::contains(cxx::FunctionSymbol* function) const {
  if (!function) {
    return false;
  }
  auto found = contracts_.find(function->canonical());
  return found != contracts_.end() && found->second.source;
}

cxx::AST* CommandPrograms::parameter_source(const Contract& contract,
                                            size_t ordinal,
                                            cxx::AST* fallback) const {
  return ordinal < contract.parameter_sources.size()
             ? static_cast<cxx::AST*>(contract.parameter_sources[ordinal])
             : fallback;
}

const CommandProgramProjection& CommandPrograms::project(
    cxx::FunctionSymbol* function, Types& types, cxx::AST* owner) {
  auto found = contracts_.find(function->canonical());
  if (found == contracts_.end() || !found->second.source) {
    diagnostics_.reject(unit_, owner,
                        "command signature requires a command_program");
  }
  auto& contract = found->second;
  if (contract.projection) {
    return *contract.projection;
  }

  auto parameters = function->parameters();
  const size_t launch_parameter_ordinal =
      contract.launch_parameter_ordinal.value_or(parameters.size());
  auto projection = std::make_unique<CommandProgramProjection>();
  projection->launch_parameter_ordinal = launch_parameter_ordinal;
  projection->source_types.reserve(parameters.size());
  projection->parameters.reserve(parameters.size());
  size_t source_component_offset = 0;
  size_t specialization_count = 0;
  size_t binding_count = 0;
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter = parameters[ordinal];
    auto* source = parameter_source(contract, ordinal, owner);
    if (unit_.typeTraits().is_reference(parameter->type()) ||
        unit_.typeTraits().is_volatile(parameter->type())) {
      diagnostics_.reject(
          unit_, source,
          "command_program parameters require non-reference, non-volatile "
          "value types");
    }
    const auto* source_type = types.unqualified_value(parameter->type());
    const auto& partition = types.partition(source_type, source);
    projection->source_types.push_back(source_type);
    projection->requires_binding |= types.requires_binding(source_type, source);
    const bool binding = ordinal >= launch_parameter_ordinal;
    size_t parameter_specializations = partition.component_count;
    std::optional<size_t> binding_ordinal;
    if (binding) {
      if (partition.kind != ValueKind::Buffer &&
          partition.kind != ValueKind::View) {
        diagnostics_.reject(
            unit_, source,
            "launch-bound command_program parameters require a direct "
            "loom::type::buffer or loom::type::view value");
      }
      IREE_ASSERT_GT(partition.component_count, 0);
      parameter_specializations = partition.component_count - 1;
      binding_ordinal = binding_count++;
    }
    projection->parameters.push_back({
        source_component_offset,
        partition.component_count,
        specialization_count,
        parameter_specializations,
        binding_ordinal,
    });
    source_component_offset += partition.component_count;
    specialization_count += parameter_specializations;
  }
  projection->specialization_count = specialization_count;
  projection->binding_count = binding_count;
  contract.projection = std::move(projection);
  return *contract.projection;
}

BoundCommandProgramSignature CommandPrograms::bind(
    cxx::FunctionSymbol* function, Types& types, cxx::AST* owner,
    loom_builder_t* builder, SignatureIdentityRequirement identities) {
  BoundCommandProgramSignature result;
  const auto& projection = project(function, types, owner);
  result.projection = &projection;
  const size_t component_count =
      projection.specialization_count + projection.binding_count;
  const bool requires_identities =
      identities == SignatureIdentityRequirement::Required ||
      projection.requires_binding;
  if (requires_identities) {
    result.destination_identities.resize(component_count);
    check(loom_builder_reserve_values(builder, component_count,
                                      result.destination_identities.data()));
    result.source_identities.resize(component_count);
    for (const auto& parameter : projection.parameters) {
      for (size_t index = 0; index < parameter.specialization_count; ++index) {
        result.source_identities[parameter.source_component_offset + index] =
            result.destination_identities[parameter.specialization_offset +
                                          index];
      }
      if (parameter.binding_ordinal) {
        result.source_identities[parameter.source_component_offset +
                                 parameter.specialization_count] =
            result.destination_identities[projection.specialization_count +
                                          *parameter.binding_ordinal];
      }
    }
  }

  result.specialization_types.reserve(projection.specialization_count);
  result.binding_types.reserve(projection.binding_count);
  std::vector<loom_type_t> source_types;
  source_types.reserve(component_count);
  for (size_t ordinal = 0; ordinal < projection.source_types.size();
       ++ordinal) {
    const auto& parameter = projection.parameters[ordinal];
    const size_t previous_count = source_types.size();
    if (requires_identities) {
      types.append_bound(
          projection.source_types[ordinal], owner,
          std::span<const loom_value_id_t>(result.source_identities)
              .subspan(parameter.source_component_offset,
                       parameter.component_count),
          result.type_storage, source_types);
    } else {
      types.append(projection.source_types[ordinal], owner, source_types);
    }
    IREE_ASSERT_EQ(source_types.size() - previous_count,
                   parameter.component_count);
  }
  for (const auto& parameter : projection.parameters) {
    auto parameter_types = std::span<const loom_type_t>(source_types)
                               .subspan(parameter.source_component_offset,
                                        parameter.component_count);
    result.specialization_types.insert(
        result.specialization_types.end(), parameter_types.begin(),
        parameter_types.begin() + parameter.specialization_count);
    if (parameter.binding_ordinal) {
      result.binding_types.push_back(parameter_types.back());
    }
  }
  return result;
}

}  // namespace loom::cxx_import
