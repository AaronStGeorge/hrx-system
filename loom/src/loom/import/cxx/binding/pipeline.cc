// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/pipeline.h"

#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <memory>
#include <variant>

#include "loom/import/cxx/binding/index_expression.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {

bool PipelineMemoryIntrinsic::supports(std::string_view name) {
  return name == "pipeline.memory";
}

std::optional<PipelineMemoryIntrinsic> PipelineMemoryIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty() ||
      !supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  auto fail = [&](const char* message) {
    diagnostics.reject(unit, owner, message);
  };
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto parameters = signature ? signature->parameterTypes()
                              : std::span<const cxx::Type* const>{};
  if (attribute.arguments.size() != 1 || !signature ||
      signature->isVariadic() || parameters.size() != 1 ||
      !IndexList::is_type(unit, parameters[0])) {
    fail("pipeline.memory requires one loom::type::indices value");
  }
  const auto& result = types.partition(signature->returnType(), owner);
  if (result.kind != ValueKind::Pool) {
    fail("pipeline.memory requires a loom::type::pool result");
  }
  auto arguments = function->templateArguments();
  auto selected = arguments.size() == 2
                      ? cxx::template_argument_value(arguments[0])
                      : std::nullopt;
  auto* space = selected ? std::get_if<cxx::ConstInt>(&*selected) : nullptr;
  if (!space || space->isNegative() ||
      space->toUWide() < LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
      space->toUWide() > LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC) {
    fail(
        "pipeline.memory requires one concrete memory space template argument");
  }
  return PipelineMemoryIntrinsic{
      static_cast<loom_value_fact_memory_space_t>(space->toUIntMax()), &result};
}

bool PipelineComposeIntrinsic::supports(std::string_view name) {
  return name == "pipeline.compose";
}

std::optional<PipelineComposeIntrinsic> PipelineComposeIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    PipelineContracts& pipelines, cxx::FunctionSymbol* function,
    const cxx::Attribute& attribute, cxx::AST* owner) {
  if (attribute.arguments.empty() ||
      !supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  auto fail = [&](const char* message) {
    diagnostics.reject(unit, owner, message);
  };
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (attribute.arguments.size() != 1 || !signature ||
      signature->isVariadic() ||
      !unit.typeTraits().is_void(signature->returnType())) {
    fail("pipeline.compose requires one binding and a fixed void signature");
  }
  auto arguments = function->templateArguments();
  auto selected = arguments.empty()
                      ? std::nullopt
                      : cxx::template_argument_value(arguments[0]);
  auto* address =
      selected ? std::get_if<std::shared_ptr<cxx::ConstAddress>>(&*selected)
               : nullptr;
  auto* pipeline =
      address && *address
          ? cxx::symbol_cast<cxx::FunctionSymbol>((*address)->symbol())
          : nullptr;
  if (!pipeline || !pipelines.contains(pipeline)) {
    fail("pipeline.compose requires a pipeline as its first template argument");
  }
  const auto& projection = pipelines.project(pipeline, types, owner);
  auto parameters = signature->parameterTypes();
  if (parameters.size() != projection.source_types.size()) {
    fail("pipeline.compose arguments must match the pipeline signature");
  }
  for (size_t index = 0; index < parameters.size(); ++index) {
    if (types.unqualified_value(parameters[index]) !=
        projection.source_types[index]) {
      fail("pipeline.compose arguments must preserve pipeline source types");
    }
  }
  return PipelineComposeIntrinsic{pipeline, &projection};
}

}  // namespace loom::cxx_import
