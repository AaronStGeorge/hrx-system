// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/shaped.h"

#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "loom/import/cxx/binding/combining.h"
#include "loom/import/cxx/binding/index_values.h"
#include "loom/import/cxx/binding/scalar_bindings.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/combining.h"

namespace loom::cxx_import {
namespace {

uint8_t math_permissions(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         const cxx::Attribute& attribute, size_t first,
                         cxx::AST* owner) {
  uint8_t flags = 0;
  for (size_t index = first; index < attribute.arguments.size(); ++index) {
    uint8_t flag;
    if (!loom_cxx_fastmath_flag_parse(view(attribute.arguments[index]->name()),
                                      &flag)) {
      diagnostics.reject(unit, owner,
                         "intrinsic has an unsupported fast-math flag");
    }
    flags |= flag;
  }
  return flags;
}

loom_vector_dot4i_kind_t dot4i_kind(bool lhs_unsigned, bool rhs_unsigned) {
  if (lhs_unsigned) {
    return rhs_unsigned ? LOOM_VECTOR_DOT4I_KIND_U8U8
                        : LOOM_VECTOR_DOT4I_KIND_U8S8;
  }
  return rhs_unsigned ? LOOM_VECTOR_DOT4I_KIND_S8U8
                      : LOOM_VECTOR_DOT4I_KIND_S8S8;
}

struct StaticVector {
  loom_type_t type;
  const VectorPartition* partition = nullptr;
  const cxx::Type* element_type;
  std::vector<int64_t> extents;
};

StaticVector require_static_vector(cxx::TranslationUnit& unit,
                                   Diagnostics& diagnostics, Types& types,
                                   const cxx::Type* source,
                                   std::string_view role, cxx::AST* owner) {
  if (const auto* native = types.native_vector(source)) {
    return {
        types.get(native, owner),
        nullptr,
        types.unqualified(native->elementType()),
        {static_cast<int64_t>(native->elementCount())},
    };
  }
  const auto& admitted = types.partition(source, owner);
  if (admitted.kind != ValueKind::Vector) {
    diagnostics.reject(unit, owner,
                       std::string(role) + " must be a vector value");
  }
  const auto& vector = static_cast<const VectorPartition&>(admitted);
  if (vector.is_dynamic()) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " requires an all-static vector shape");
  }
  return {
      vector.static_type,
      &vector,
      types.unqualified(vector.element_type),
      {vector.extents.begin(), vector.extents.end()},
  };
}

std::vector<cxx::TemplateArgument> expanded_template_arguments(
    cxx::FunctionSymbol* function) {
  return cxx::expand_template_arguments(function->templateArguments());
}

int64_t constant_template_index(cxx::TranslationUnit& unit,
                                Diagnostics& diagnostics,
                                const cxx::TemplateArgument& argument,
                                std::string_view role, cxx::AST* owner) {
  auto value = cxx::template_argument_value(argument);
  auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!number || number->isNegative() ||
      number->toUWide() > static_cast<uint64_t>(INT64_MAX)) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " requires nonnegative constant indices");
  }
  return static_cast<int64_t>(number->toUIntMax());
}

uint64_t static_element_count(cxx::TranslationUnit& unit,
                              Diagnostics& diagnostics,
                              const StaticVector& vector, std::string_view role,
                              cxx::AST* owner) {
  uint64_t count = 1;
  for (int64_t extent : vector.extents) {
    if (extent <= 0 || count > UINT64_MAX / static_cast<uint64_t>(extent)) {
      diagnostics.reject(
          unit, owner,
          std::string(role) + " has an unrepresentable static element count");
    }
    count *= static_cast<uint64_t>(extent);
  }
  return count;
}

uint64_t static_bit_count(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                          const StaticVector& vector, std::string_view role,
                          cxx::AST* owner) {
  const uint64_t element_bits =
      loom_scalar_type_bitwidth(loom_type_element_type(vector.type));
  const uint64_t element_count =
      static_element_count(unit, diagnostics, vector, role, owner);
  if (!element_bits || element_count > UINT64_MAX / element_bits) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " has an unrepresentable total bit count");
  }
  return element_count * element_bits;
}

bool same_shape(const StaticVector& lhs, const StaticVector& rhs) {
  return lhs.element_type == rhs.element_type && lhs.extents == rhs.extents;
}

std::optional<uint8_t> integer_comparison_predicate(std::string_view name) {
  if (name == "eq") {
    return LOOM_VECTOR_CMPI_PREDICATE_EQ;
  }
  if (name == "ne") {
    return LOOM_VECTOR_CMPI_PREDICATE_NE;
  }
  if (name == "slt") {
    return LOOM_VECTOR_CMPI_PREDICATE_SLT;
  }
  if (name == "sle") {
    return LOOM_VECTOR_CMPI_PREDICATE_SLE;
  }
  if (name == "sgt") {
    return LOOM_VECTOR_CMPI_PREDICATE_SGT;
  }
  if (name == "sge") {
    return LOOM_VECTOR_CMPI_PREDICATE_SGE;
  }
  if (name == "ult") {
    return LOOM_VECTOR_CMPI_PREDICATE_ULT;
  }
  if (name == "ule") {
    return LOOM_VECTOR_CMPI_PREDICATE_ULE;
  }
  if (name == "ugt") {
    return LOOM_VECTOR_CMPI_PREDICATE_UGT;
  }
  if (name == "uge") {
    return LOOM_VECTOR_CMPI_PREDICATE_UGE;
  }
  return std::nullopt;
}

std::optional<uint8_t> float_comparison_predicate(std::string_view name) {
  if (name == "oeq") {
    return LOOM_VECTOR_CMPF_PREDICATE_OEQ;
  }
  if (name == "ogt") {
    return LOOM_VECTOR_CMPF_PREDICATE_OGT;
  }
  if (name == "oge") {
    return LOOM_VECTOR_CMPF_PREDICATE_OGE;
  }
  if (name == "olt") {
    return LOOM_VECTOR_CMPF_PREDICATE_OLT;
  }
  if (name == "ole") {
    return LOOM_VECTOR_CMPF_PREDICATE_OLE;
  }
  if (name == "one") {
    return LOOM_VECTOR_CMPF_PREDICATE_ONE;
  }
  if (name == "ord") {
    return LOOM_VECTOR_CMPF_PREDICATE_ORD;
  }
  if (name == "ueq") {
    return LOOM_VECTOR_CMPF_PREDICATE_UEQ;
  }
  if (name == "ugt") {
    return LOOM_VECTOR_CMPF_PREDICATE_UGT;
  }
  if (name == "uge") {
    return LOOM_VECTOR_CMPF_PREDICATE_UGE;
  }
  if (name == "ult") {
    return LOOM_VECTOR_CMPF_PREDICATE_ULT;
  }
  if (name == "ule") {
    return LOOM_VECTOR_CMPF_PREDICATE_ULE;
  }
  if (name == "une") {
    return LOOM_VECTOR_CMPF_PREDICATE_UNE;
  }
  if (name == "uno") {
    return LOOM_VECTOR_CMPF_PREDICATE_UNO;
  }
  return std::nullopt;
}

}  // namespace

std::optional<ShapedIntrinsic::Operation> ShapedIntrinsic::admit(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const cxx::Attribute& attribute, cxx::AST* owner) {
  auto name = attribute.arguments[0]->name();
  if (name == "vector.splat") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner, "vector splat has no semantic arguments");
    }
    return Splat{};
  }
  if (name == "vector.slice") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner, "vector slice has no semantic arguments");
    }
    return Slice{};
  }
  if (name == "vector.iota") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner, "vector iota has no semantic arguments");
    }
    return Iota{};
  }
  if (name == "vector.concat") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector concat has no semantic arguments");
    }
    return Concat{};
  }
  if (name == "vector.transpose") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector transpose has no semantic arguments");
    }
    return Transpose{};
  }
  if (name == "vector.interleave") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector interleave has no semantic arguments");
    }
    return Interleave{};
  }
  if (name == "vector.bitcast") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector bitcast has no semantic arguments");
    }
    return Bitcast{};
  }
  if (name == "vector.extract") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector extract has no semantic arguments");
    }
    return Extract{};
  }
  if (name == "vector.select") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector select has no semantic arguments");
    }
    return Select{};
  }
  if (name == "vector.cmpi" || name == "vector.cmpf") {
    const bool floating = name == "vector.cmpf";
    if (attribute.arguments.size() < 2) {
      diagnostics.reject(unit, owner, "vector comparison requires a predicate");
    }
    auto predicate =
        floating ? float_comparison_predicate(attribute.arguments[1]->name())
                 : integer_comparison_predicate(attribute.arguments[1]->name());
    if (!predicate) {
      diagnostics.reject(unit, owner,
                         "vector comparison has an unsupported predicate");
    }
    if (!floating && attribute.arguments.size() != 2) {
      diagnostics.reject(unit, owner,
                         "integer vector comparison has no fast-math flags");
    }
    return Comparison{
        *predicate,
        floating ? Comparison::Kind::Float : Comparison::Kind::Integer,
        floating ? math_permissions(unit, diagnostics, attribute, 2, owner)
                 : uint8_t{0},
    };
  }
  if (name == "vector.table.lookup") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "register table lookup has no semantic arguments");
    }
    return TableLookup{};
  }
  if (name == "vector.absf" || name == "vector.absi") {
    const bool floating = name == "vector.absf";
    if (!floating && attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "integer vector absolute value has no semantic "
                         "arguments");
    }
    return Absolute{
        floating ? Absolute::Kind::Float : Absolute::Kind::Integer,
        floating ? math_permissions(unit, diagnostics, attribute, 1, owner)
                 : uint8_t{0},
    };
  }
  if (name == "vector.minimumf" || name == "vector.maximumf" ||
      name == "vector.minnumf" || name == "vector.maxnumf") {
    const FloatExtremum::Kind kind =
        name == "vector.minimumf"   ? FloatExtremum::Kind::Minimum
        : name == "vector.maximumf" ? FloatExtremum::Kind::Maximum
        : name == "vector.minnumf"  ? FloatExtremum::Kind::MinNumber
                                    : FloatExtremum::Kind::MaxNumber;
    return FloatExtremum{
        kind, math_permissions(unit, diagnostics, attribute, 1, owner)};
  }
  if (name == "vector.minsi" || name == "vector.maxsi" ||
      name == "vector.minui" || name == "vector.maxui") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "integer vector extrema have no semantic arguments");
    }
    const IntegerExtremum::Kind kind =
        name == "vector.minsi"   ? IntegerExtremum::Kind::MinSigned
        : name == "vector.maxsi" ? IntegerExtremum::Kind::MaxSigned
        : name == "vector.minui" ? IntegerExtremum::Kind::MinUnsigned
                                 : IntegerExtremum::Kind::MaxUnsigned;
    return IntegerExtremum{kind};
  }
  if (name == "vector.dot4i") {
    if (attribute.arguments.size() == 1) {
      return Dot4i{std::nullopt};
    }
    if (attribute.arguments.size() != 2) {
      diagnostics.reject(unit, owner,
                         "dot4i accepts no semantic arguments or one "
                         "signedness argument: s8s8, u8s8, s8u8, or u8u8");
    }
    auto kind = attribute.arguments[1]->name();
    if (kind == "s8s8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_S8S8};
    }
    if (kind == "u8s8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_U8S8};
    }
    if (kind == "s8u8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_S8U8};
    }
    if (kind == "u8u8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_U8U8};
    }
    diagnostics.reject(unit, owner, "unsupported dot4i signedness argument");
  }
  if (name == "vector.dot2f") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner, "dot2f has no semantic arguments");
    }
    return Dot2f{};
  }
  if (name == "vector.dotf") {
    return Dotf{math_permissions(unit, diagnostics, attribute, 1, owner)};
  }
  if (name == "vector.reduce") {
    if (attribute.arguments.size() < 2) {
      diagnostics.reject(unit, owner,
                         "vector reduction requires a combining kind");
    }
    auto kind = parse_combining_kind(attribute.arguments[1]->name());
    if (!kind) {
      diagnostics.reject(unit, owner,
                         "unsupported vector reduction combining kind");
    }
    if (loom_combining_kind_accepts_integer(*kind) &&
        attribute.arguments.size() != 2) {
      diagnostics.reject(unit, owner,
                         "integer vector reduction has no fast-math flags");
    }
    return Reduction{*kind,
                     math_permissions(unit, diagnostics, attribute, 2, owner)};
  }
  return std::nullopt;
}

ShapedIntrinsic ShapedIntrinsic::resolve(Operation operation,
                                         cxx::TranslationUnit& unit,
                                         Diagnostics& diagnostics, Types& types,
                                         cxx::FunctionSymbol* function,
                                         cxx::AST* owner) {
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (std::holds_alternative<Splat>(operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "vector.splat requires dimensions and one scalar");
    }
    const auto& admitted = types.partition(signature->returnType(), owner);
    if (admitted.kind != ValueKind::Vector) {
      diagnostics.reject(unit, owner,
                         "vector.splat result must be a semantic vector");
    }
    const auto& result = static_cast<const VectorPartition&>(admitted);
    require_logical_index_record(
        unit, diagnostics, types, signature->parameterTypes()[0],
        result.component_count - 1, owner, "vector.splat dimensions");
    if (types.unqualified(signature->parameterTypes()[1]) !=
        types.unqualified(result.element_type)) {
      diagnostics.reject(unit, owner,
                         "vector.splat value must match its result element");
    }
    return ShapedIntrinsic(operation, loom_type_none(), &result);
  }
  if (auto* slice = std::get_if<Slice>(&operation)) {
    auto parameters = signature->parameterTypes();
    if (signature->isVariadic() ||
        (parameters.size() != 2 && parameters.size() != 3)) {
      diagnostics.reject(unit, owner,
                         "vector.slice requires a source, offsets, and "
                         "optional result dimensions");
    }
    const std::array<const cxx::Type*, 2> vectors = {parameters[0],
                                                     signature->returnType()};
    std::array<const cxx::Type*, 2> elements;
    std::array<const VectorPartition*, 2> semantic = {};
    std::array<int64_t, 2> native_extents;
    std::array<std::span<const int64_t>, 2> extents;
    for (size_t index = 0; index < vectors.size(); ++index) {
      if (const auto* native = types.native_vector(vectors[index])) {
        elements[index] = types.unqualified(native->elementType());
        native_extents[index] = native->elementCount();
        extents[index] = {&native_extents[index], 1};
      } else {
        const auto& admitted = types.partition(vectors[index], owner);
        if (admitted.kind != ValueKind::Vector) {
          diagnostics.reject(unit, owner,
                             "vector.slice requires source and result vectors");
        }
        semantic[index] = &static_cast<const VectorPartition&>(admitted);
        elements[index] = semantic[index]->element_type;
        extents[index] = semantic[index]->extents;
      }
    }
    if (extents[0].size() != extents[1].size()) {
      diagnostics.reject(unit, owner,
                         "vector.slice must preserve the source rank");
    }
    bool fits = true;
    for (size_t axis = 0; axis < extents[0].size(); ++axis) {
      if (extents[0][axis] >= 0 && extents[1][axis] > extents[0][axis]) {
        fits = false;
      }
    }
    if (elements[0] != elements[1] || !fits) {
      diagnostics.reject(
          unit, owner,
          "vector.slice result must preserve the element type and fit within "
          "the source");
    }
    const auto& offsets = types.partition(parameters[1], owner);
    if (offsets.kind == ValueKind::Record) {
      slice->offset_unsigned_mask = require_index_record(
          unit, diagnostics, types, parameters[1], extents[0].size(), owner,
          "vector.slice offsets");
    } else {
      if (extents[0].size() != 1) {
        diagnostics.reject(unit, owner,
                           "vector.slice offsets require a full-rank record");
      }
      slice->offset_unsigned_mask =
          require_index_value(unit, diagnostics, types, parameters[1], owner,
                              "vector.slice offset");
    }
    slice->has_dimensions = parameters.size() == 3;
    if (semantic[1]) {
      if (semantic[1]->is_dynamic() && !slice->has_dimensions) {
        diagnostics.reject(unit, owner,
                           "dependent vector.slice requires result dimensions");
      }
      if (slice->has_dimensions) {
        require_logical_index_record(unit, diagnostics, types, parameters[2],
                                     semantic[1]->component_count - 1, owner,
                                     "vector.slice dimensions");
      }
    } else if (slice->has_dimensions) {
      diagnostics.reject(unit, owner,
                         "native vector.slice does not accept result "
                         "dimensions");
    }
    return ShapedIntrinsic(
        operation,
        semantic[1] ? loom_type_none() : types.get(vectors[1], owner),
        semantic[1], {semantic[0], nullptr, nullptr});
  }
  if (std::holds_alternative<Iota>(operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "vector.iota requires scalar base and step operands");
    }
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector.iota result", owner);
    auto element = loom_type_element_type(result.type);
    if (!loom_scalar_type_is_integer(element) ||
        element == LOOM_SCALAR_TYPE_I1 ||
        types.unqualified(signature->parameterTypes()[0]) !=
            result.element_type ||
        types.unqualified(signature->parameterTypes()[1]) !=
            result.element_type) {
      diagnostics.reject(
          unit, owner,
          "vector.iota base, step, and result element types must be one "
          "non-boolean integer type");
    }
    return ShapedIntrinsic(operation, result.type, result.partition);
  }
  if (auto* concat = std::get_if<Concat>(&operation)) {
    auto parameters = signature->parameterTypes();
    if (signature->isVariadic() || parameters.empty()) {
      diagnostics.reject(unit, owner,
                         "vector.concat requires one or more vector operands");
    }
    auto arguments = expanded_template_arguments(function);
    if (arguments.empty()) {
      diagnostics.reject(unit, owner,
                         "vector.concat requires a constant axis selector");
    }
    concat->axis = constant_template_index(unit, diagnostics, arguments[0],
                                           "vector.concat axis", owner);
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector.concat result", owner);
    if (static_cast<size_t>(concat->axis) >= result.extents.size()) {
      diagnostics.reject(unit, owner,
                         "vector.concat axis must select a result axis");
    }
    uint64_t combined_extent = 0;
    for (auto* parameter : parameters) {
      auto input = require_static_vector(unit, diagnostics, types, parameter,
                                         "vector.concat operand", owner);
      if (input.element_type != result.element_type ||
          input.extents.size() != result.extents.size()) {
        diagnostics.reject(
            unit, owner,
            "vector.concat operands and result must share rank and element "
            "type");
      }
      for (size_t axis = 0; axis < result.extents.size(); ++axis) {
        if (axis != static_cast<size_t>(concat->axis) &&
            input.extents[axis] != result.extents[axis]) {
          diagnostics.reject(
              unit, owner,
              "vector.concat must preserve every non-concatenated extent");
        }
      }
      const uint64_t input_extent =
          static_cast<uint64_t>(input.extents[concat->axis]);
      if (input_extent > static_cast<uint64_t>(INT64_MAX) - combined_extent) {
        diagnostics.reject(unit, owner,
                           "vector.concat result extent is unrepresentable");
      }
      combined_extent += input_extent;
    }
    if (combined_extent !=
        static_cast<uint64_t>(result.extents[concat->axis])) {
      diagnostics.reject(
          unit, owner,
          "vector.concat result extent must equal the input extent sum");
    }
    return ShapedIntrinsic(operation, result.type, result.partition);
  }
  if (auto* transpose = std::get_if<Transpose>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector.transpose requires one vector operand");
    }
    auto source = require_static_vector(unit, diagnostics, types,
                                        signature->parameterTypes()[0],
                                        "vector.transpose source", owner);
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector.transpose result", owner);
    auto arguments = expanded_template_arguments(function);
    if (arguments.size() != source.extents.size() + 2) {
      diagnostics.reject(
          unit, owner,
          "vector.transpose requires one constant selector per source axis");
    }
    transpose->permutation.reserve(source.extents.size());
    for (size_t index = 1; index + 1 < arguments.size(); ++index) {
      transpose->permutation.push_back(
          constant_template_index(unit, diagnostics, arguments[index],
                                  "vector.transpose permutation", owner));
    }
    if (source.element_type != result.element_type ||
        source.extents.size() != result.extents.size()) {
      diagnostics.reject(
          unit, owner,
          "vector.transpose source and result must share rank and element "
          "type");
    }
    std::vector<bool> selected(source.extents.size());
    for (size_t axis = 0; axis < transpose->permutation.size(); ++axis) {
      int64_t source_axis = transpose->permutation[axis];
      if (static_cast<size_t>(source_axis) >= source.extents.size() ||
          selected[source_axis]) {
        diagnostics.reject(unit, owner,
                           "vector.transpose permutation must select each "
                           "source axis exactly once");
      }
      selected[source_axis] = true;
      if (result.extents[axis] != source.extents[source_axis]) {
        diagnostics.reject(
            unit, owner,
            "vector.transpose result shape must match its permutation");
      }
    }
    return ShapedIntrinsic(operation, result.type, result.partition);
  }
  if (auto* interleave = std::get_if<Interleave>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "vector.interleave requires even and odd operands");
    }
    auto even = require_static_vector(unit, diagnostics, types,
                                      signature->parameterTypes()[0],
                                      "vector.interleave even operand", owner);
    auto odd = require_static_vector(unit, diagnostics, types,
                                     signature->parameterTypes()[1],
                                     "vector.interleave odd operand", owner);
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector.interleave result", owner);
    auto arguments = expanded_template_arguments(function);
    if (arguments.empty()) {
      diagnostics.reject(unit, owner,
                         "vector.interleave requires a constant axis selector");
    }
    interleave->axis = constant_template_index(unit, diagnostics, arguments[0],
                                               "vector.interleave axis", owner);
    if (!same_shape(even, odd) || even.element_type != result.element_type ||
        even.extents.size() != result.extents.size() ||
        static_cast<size_t>(interleave->axis) >= even.extents.size()) {
      diagnostics.reject(
          unit, owner,
          "vector.interleave requires equal inputs and a same-rank result");
    }
    for (size_t axis = 0; axis < even.extents.size(); ++axis) {
      int64_t expected = even.extents[axis];
      if (axis == static_cast<size_t>(interleave->axis)) {
        if (expected > INT64_MAX / 2) {
          diagnostics.reject(unit, owner,
                             "vector.interleave result extent is "
                             "unrepresentable");
        }
        expected *= 2;
      }
      if (result.extents[axis] != expected) {
        diagnostics.reject(
            unit, owner,
            "vector.interleave result must double the selected input extent");
      }
    }
    return ShapedIntrinsic(operation, result.type, result.partition);
  }
  if (std::holds_alternative<Bitcast>(operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector.bitcast requires one vector operand");
    }
    auto source = require_static_vector(unit, diagnostics, types,
                                        signature->parameterTypes()[0],
                                        "vector.bitcast source", owner);
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector.bitcast result", owner);
    uint64_t source_bits = static_bit_count(unit, diagnostics, source,
                                            "vector.bitcast source", owner);
    uint64_t result_bits = static_bit_count(unit, diagnostics, result,
                                            "vector.bitcast result", owner);
    if (source_bits != result_bits) {
      diagnostics.reject(
          unit, owner, "vector.bitcast requires equal total value bit counts");
    }
    return ShapedIntrinsic(operation, result.type, result.partition, {},
                           source.type);
  }
  if (auto* extract = std::get_if<Extract>(&operation)) {
    auto parameters = signature->parameterTypes();
    if (signature->isVariadic() || parameters.empty()) {
      diagnostics.reject(unit, owner,
                         "vector.extract requires one vector source");
    }
    auto source = require_static_vector(unit, diagnostics, types, parameters[0],
                                        "vector.extract source", owner);
    if (parameters.size() > 1 &&
        parameters.size() - 1 > source.extents.size()) {
      diagnostics.reject(
          unit, owner,
          "vector.extract must consume one or more leading source axes");
    }
    if (parameters.size() == 1) {
      auto arguments = expanded_template_arguments(function);
      if (arguments.size() < 3) {
        diagnostics.reject(
            unit, owner,
            "vector.extract requires one or more constant leading indices");
      }
      extract->static_indices.reserve(arguments.size() - 2);
      for (size_t index = 1; index + 1 < arguments.size(); ++index) {
        extract->static_indices.push_back(
            constant_template_index(unit, diagnostics, arguments[index],
                                    "vector.extract selector", owner));
      }
    } else {
      for (size_t index = 1; index < parameters.size(); ++index) {
        if (require_index_value(unit, diagnostics, types, parameters[index],
                                owner, "vector.extract index")) {
          extract->index_unsigned_mask |= uint16_t{1} << (index - 1);
        }
      }
    }
    size_t index_count = parameters.size() == 1 ? extract->static_indices.size()
                                                : parameters.size() - 1;
    if (index_count == 0 || index_count > source.extents.size()) {
      diagnostics.reject(
          unit, owner,
          "vector.extract must consume one or more leading source axes");
    }
    for (size_t index = 0; index < extract->static_indices.size(); ++index) {
      if (extract->static_indices[index] >= source.extents[index]) {
        diagnostics.reject(unit, owner,
                           "vector.extract selector exceeds its source extent");
      }
    }
    const size_t result_rank = source.extents.size() - index_count;
    if (result_rank == 0) {
      auto* result = types.unqualified(signature->returnType());
      if (result != source.element_type) {
        diagnostics.reject(
            unit, owner,
            "vector.extract scalar result must match the source element type");
      }
      return ShapedIntrinsic(operation, types.get(result, owner));
    }
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector.extract result", owner);
    std::vector<int64_t> tail(source.extents.begin() + index_count,
                              source.extents.end());
    if (result.element_type != source.element_type || result.extents != tail) {
      diagnostics.reject(
          unit, owner,
          "vector.extract result must match the unconsumed source axes");
    }
    return ShapedIntrinsic(operation, result.type, result.partition);
  }
  if (std::holds_alternative<Select>(operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 3) {
      diagnostics.reject(
          unit, owner,
          "vector.select requires a condition and two value operands");
    }
    auto condition = require_static_vector(unit, diagnostics, types,
                                           signature->parameterTypes()[0],
                                           "vector.select condition", owner);
    auto true_value = require_static_vector(unit, diagnostics, types,
                                            signature->parameterTypes()[1],
                                            "vector.select true value", owner);
    auto false_value = require_static_vector(
        unit, diagnostics, types, signature->parameterTypes()[2],
        "vector.select false value", owner);
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector.select result", owner);
    if (loom_type_element_type(condition.type) != LOOM_SCALAR_TYPE_I1 ||
        condition.extents != result.extents ||
        !same_shape(true_value, false_value) ||
        !same_shape(true_value, result)) {
      diagnostics.reject(
          unit, owner,
          "vector.select requires an i1 condition shape and one value type");
    }
    return ShapedIntrinsic(
        operation, result.type, result.partition,
        {condition.partition, true_value.partition, false_value.partition});
  }
  if (auto* comparison = std::get_if<Comparison>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "vector comparison requires two operands");
    }
    auto left = require_static_vector(unit, diagnostics, types,
                                      signature->parameterTypes()[0],
                                      "vector comparison operand", owner);
    auto right = require_static_vector(unit, diagnostics, types,
                                       signature->parameterTypes()[1],
                                       "vector comparison operand", owner);
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "vector comparison result", owner);
    const bool floating = comparison->kind == Comparison::Kind::Float;
    const bool input_category_matches =
        floating ? types.is_float(left.element_type)
                 : unit.typeTraits().is_integral(left.element_type) &&
                       left.element_type->kind() != cxx::TypeKind::kBool;
    if (!loom_type_equal(left.type, right.type) ||
        left.extents != result.extents ||
        loom_type_element_type(result.type) != LOOM_SCALAR_TYPE_I1 ||
        !input_category_matches) {
      diagnostics.reject(
          unit, owner,
          "vector comparison requires matching numeric operands and an i1 "
          "result with the same shape");
    }
    if (!floating && comparison->predicate >= LOOM_VECTOR_CMPI_PREDICATE_SLT) {
      const bool expects_unsigned =
          comparison->predicate >= LOOM_VECTOR_CMPI_PREDICATE_ULT;
      if (types.is_unsigned(left.element_type) != expects_unsigned ||
          types.is_unsigned(right.element_type) != expects_unsigned) {
        diagnostics.reject(unit, owner,
                           "integer vector comparison predicate signedness "
                           "must match its operands");
      }
    }
    return ShapedIntrinsic(operation, result.type, result.partition,
                           {left.partition, right.partition, nullptr},
                           left.type);
  }
  bool lookup = std::holds_alternative<TableLookup>(operation);
  if (auto* absolute = std::get_if<Absolute>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector absolute value requires one operand");
    }
    const auto* result = types.native_vector(signature->returnType());
    const auto* input = types.native_vector(signature->parameterTypes()[0]);
    if (!result || !input || result != input) {
      diagnostics.reject(
          unit, owner,
          "vector absolute value requires one vector type for its operand "
          "and result");
    }
    const auto* element = types.unqualified(result->elementType());
    const bool category_matches =
        absolute->kind == Absolute::Kind::Float
            ? types.is_float(element)
            : unit.typeTraits().is_integral(element) &&
                  element->kind() != cxx::TypeKind::kBool;
    if (!category_matches) {
      diagnostics.reject(unit, owner,
                         "vector absolute value element type does not match "
                         "the operation category");
    }
    return ShapedIntrinsic(operation, types.get(result, owner));
  }
  if (std::holds_alternative<FloatExtremum>(operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "floating vector extremum requires two operands");
    }
    const auto* result = types.native_vector(signature->returnType());
    const auto* left = types.native_vector(signature->parameterTypes()[0]);
    const auto* right = types.native_vector(signature->parameterTypes()[1]);
    if (!result || result != left || result != right ||
        !types.is_float(result->elementType())) {
      diagnostics.reject(
          unit, owner,
          "floating vector extremum requires one floating-point vector type "
          "for its operands and result");
    }
    return ShapedIntrinsic(operation, types.get(result, owner));
  }
  if (auto* extremum = std::get_if<IntegerExtremum>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "integer vector extremum requires two operands");
    }
    auto left = require_static_vector(unit, diagnostics, types,
                                      signature->parameterTypes()[0],
                                      "integer vector extremum operand", owner);
    auto right = require_static_vector(
        unit, diagnostics, types, signature->parameterTypes()[1],
        "integer vector extremum operand", owner);
    auto result =
        require_static_vector(unit, diagnostics, types, signature->returnType(),
                              "integer vector extremum result", owner);
    const bool expects_unsigned =
        extremum->kind == IntegerExtremum::Kind::MinUnsigned ||
        extremum->kind == IntegerExtremum::Kind::MaxUnsigned;
    if (!same_shape(left, right) || !same_shape(left, result) ||
        !unit.typeTraits().is_integral(result.element_type) ||
        result.element_type->kind() == cxx::TypeKind::kBool ||
        types.is_unsigned(result.element_type) != expects_unsigned) {
      diagnostics.reject(
          unit, owner,
          "integer vector extremum requires one matching signedness and type");
    }
    return ShapedIntrinsic(operation, result.type, result.partition,
                           {left.partition, right.partition, nullptr});
  }
  if (auto* reduction = std::get_if<Reduction>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "vector reduction requires a vector and scalar seed");
    }
    const auto* result = types.unqualified(signature->returnType());
    if (!types.is_float(result) && (!unit.typeTraits().is_integral(result) ||
                                    result->kind() == cxx::TypeKind::kBool)) {
      diagnostics.reject(unit, owner,
                         "vector reduction result must be a numeric scalar");
    }
    const auto* native = types.native_vector(signature->parameterTypes()[0]);
    const VectorPartition* semantic = nullptr;
    const cxx::Type* element = nullptr;
    if (native) {
      element = native->elementType();
    } else {
      const auto& admitted =
          types.partition(signature->parameterTypes()[0], owner);
      if (admitted.kind == ValueKind::Vector) {
        semantic = &static_cast<const VectorPartition&>(admitted);
        element = semantic->element_type;
      }
    }
    if (!element || types.unqualified(element) != result ||
        types.unqualified(signature->parameterTypes()[1]) != result) {
      diagnostics.reject(
          unit, owner,
          "vector element, scalar seed, and result types must match");
    }
    if (types.is_float(result) !=
        loom_combining_kind_accepts_float(reduction->kind)) {
      diagnostics.reject(unit, owner,
                         "vector reduction combining kind does not match the "
                         "element type");
    }
    if (((reduction->kind == LOOM_COMBINING_KIND_MINSI ||
          reduction->kind == LOOM_COMBINING_KIND_MAXSI) &&
         types.is_unsigned(result)) ||
        ((reduction->kind == LOOM_COMBINING_KIND_MINUI ||
          reduction->kind == LOOM_COMBINING_KIND_MAXUI) &&
         !types.is_unsigned(result))) {
      diagnostics.reject(unit, owner,
                         "vector reduction signedness must match the source "
                         "element type");
    }
    std::array<const VectorPartition*, 3> semantic_operands = {};
    semantic_operands[0] = semantic;
    return ShapedIntrinsic(operation, types.get(result, owner), nullptr,
                           semantic_operands);
  }
  bool scalar_result = std::holds_alternative<Dotf>(operation);
  size_t operand_count = lookup ? 2 : 3;
  if (signature->isVariadic() ||
      signature->parameterTypes().size() != operand_count) {
    diagnostics.reject(unit, owner,
                       "intrinsic declaration has the wrong operand count");
  }
  const auto* result = types.unqualified(signature->returnType());
  const auto* result_vector = types.native_vector(result);
  if (!scalar_result && !result_vector) {
    diagnostics.reject(unit, owner, "shaped intrinsic result must be a vector");
  }
  if (scalar_result && !types.is_float(result) &&
      (!unit.typeTraits().is_integral(result) ||
       result->kind() == cxx::TypeKind::kBool)) {
    diagnostics.reject(
        unit, owner,
        "vector reduction or dotf result must be a numeric scalar");
  }
  auto result_type = types.get(result, owner);
  size_t vector_count = operand_count - (scalar_result ? 1 : 0);
  std::array<const cxx::VectorType*, 3> operands;
  std::array<loom_type_t, 3> operand_types;
  for (size_t index = 0; index < vector_count; ++index) {
    operands[index] = types.native_vector(signature->parameterTypes()[index]);
    if (!operands[index]) {
      diagnostics.reject(unit, owner,
                         "shaped intrinsic operands must be vectors");
    }
    operand_types[index] = types.get(operands[index], owner);
  }
  if (scalar_result &&
      (types.unqualified(operands[0]->elementType()) != result ||
       types.unqualified(signature->parameterTypes().back()) != result)) {
    diagnostics.reject(
        unit, owner,
        "vector element, scalar seed, and result types must match");
  }

  if (lookup) {
    if (types.unqualified(operands[0]->elementType()) !=
            types.unqualified(result_vector->elementType()) ||
        operands[1]->elementCount() != result_vector->elementCount() ||
        !unit.typeTraits().is_integral(operands[1]->elementType())) {
      diagnostics.reject(unit, owner,
                         "register table lookup requires the table's element "
                         "type and the integer index vector's lane count");
    }
  } else if (auto* dot = std::get_if<Dot4i>(&operation)) {
    if (loom_type_element_type(operand_types[0]) != LOOM_SCALAR_TYPE_I8 ||
        loom_type_element_type(operand_types[1]) != LOOM_SCALAR_TYPE_I8 ||
        loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_I32 ||
        operands[0]->elementCount() != operands[1]->elementCount() ||
        operands[0]->elementCount() != result_vector->elementCount() * 4 ||
        operands[2] != result_vector) {
      diagnostics.reject(unit, owner,
                         "dot4i requires equal i8 input shapes with four lanes "
                         "per i32 accumulator/result lane");
    }
    auto source_kind =
        dot4i_kind(types.is_unsigned(operands[0]->elementType()),
                   types.is_unsigned(operands[1]->elementType()));
    if (!dot->kind) {
      dot->kind = source_kind;
    } else if (*dot->kind != source_kind) {
      diagnostics.reject(unit, owner,
                         "dot4i signedness must match the source byte types");
    }
  } else if (std::holds_alternative<Dot2f>(operation)) {
    auto input_element = loom_type_element_type(operand_types[0]);
    if ((input_element != LOOM_SCALAR_TYPE_F16 &&
         input_element != LOOM_SCALAR_TYPE_BF16) ||
        operands[0] != operands[1] ||
        loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_F32 ||
        operands[0]->elementCount() != result_vector->elementCount() * 2 ||
        operands[2] != result_vector) {
      diagnostics.reject(
          unit, owner,
          "dot2f requires equal f16 or bf16 input types with two "
          "lanes per f32 accumulator/result lane");
    }
  } else if (std::holds_alternative<Dotf>(operation)) {
    if (!types.is_float(result) || operands[0] != operands[1]) {
      diagnostics.reject(unit, owner,
                         "dotf requires equal floating-point vector types");
    }
  }
  return ShapedIntrinsic(operation, result_type);
}

Value ShapedIntrinsic::call(std::span<const Value> arguments, ValueArena& arena,
                            uint8_t math_flags, loom_builder_t* builder,
                            loom_location_id_t location) const {
  if (std::holds_alternative<Splat>(operation_)) {
    auto dimensions = arguments[0].components();
    BoundTypeStorage type_storage;
    auto result_type = result_vector_->bind_type(dimensions, type_storage);
    loom_op_t* op;
    check(loom_vector_splat_build(builder, arguments[1].ssa(), result_type,
                                  location, &op));
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> components;
    std::copy(dimensions.begin(), dimensions.end(), components.begin());
    components[dimensions.size()] = loom_op_results(op)[0];
    return arena.capture(*result_vector_,
                         std::span<const loom_value_id_t>(components)
                             .first(dimensions.size() + 1));
  }
  auto vector_value = [&](size_t index) {
    const auto* semantic = semantic_operands_[index];
    return semantic ? semantic->vector_value(arguments[index].components())
                    : arguments[index].ssa();
  };
  auto capture_result = [&](loom_value_id_t value) {
    return result_vector_ ? arena.capture(*result_vector_, {&value, 1})
                          : Value(value);
  };
  if (auto* slice = std::get_if<Slice>(&operation_)) {
    auto offsets = arguments[1].components();
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> indices;
    std::array<int64_t, LOOM_TYPE_MAX_RANK> static_offsets;
    static_offsets.fill(std::numeric_limits<int64_t>::min());
    for (size_t axis = 0; axis < offsets.size(); ++axis) {
      indices[axis] =
          cast_index(Value(offsets[axis]),
                     is_unsigned_component(slice->offset_unsigned_mask, axis),
                     builder, location);
    }
    std::span<const loom_value_id_t> dimensions;
    if (slice->has_dimensions) {
      dimensions = arguments[2].components();
    }
    BoundTypeStorage type_storage;
    auto result_type = result_vector_
                           ? result_vector_->bind_type(dimensions, type_storage)
                           : result_type_;
    loom_op_t* op;
    check(loom_vector_slice_build(builder, vector_value(0), indices.data(),
                                  offsets.size(), static_offsets.data(),
                                  offsets.size(), result_type, location, &op));
    if (!result_vector_) {
      return Value(loom_op_results(op)[0]);
    }
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> components;
    std::copy(dimensions.begin(), dimensions.end(), components.begin());
    components[dimensions.size()] = loom_op_results(op)[0];
    return arena.capture(*result_vector_,
                         std::span<const loom_value_id_t>(components)
                             .first(dimensions.size() + 1));
  }
  if (std::holds_alternative<Iota>(operation_)) {
    loom_op_t* op;
    check(loom_vector_iota_build(builder, arguments[0].ssa(),
                                 arguments[1].ssa(), result_type_, location,
                                 &op));
    return capture_result(loom_op_results(op)[0]);
  }
  if (auto* concat = std::get_if<Concat>(&operation_)) {
    std::vector<loom_value_id_t> inputs;
    inputs.reserve(arguments.size());
    for (const auto& argument : arguments) {
      inputs.push_back(argument.components().back());
    }
    loom_op_t* op;
    check(loom_vector_concat_build(builder, concat->axis, inputs.data(),
                                   inputs.size(), result_type_, location, &op));
    return capture_result(loom_op_results(op)[0]);
  }
  if (auto* transpose = std::get_if<Transpose>(&operation_)) {
    loom_op_t* op;
    check(loom_vector_transpose_build(
        builder, transpose->permutation.data(), transpose->permutation.size(),
        arguments[0].components().back(), result_type_, location, &op));
    return capture_result(loom_op_results(op)[0]);
  }
  if (auto* interleave = std::get_if<Interleave>(&operation_)) {
    loom_op_t* op;
    check(loom_vector_interleave_build(
        builder, interleave->axis, arguments[0].components().back(),
        arguments[1].components().back(), result_type_, location, &op));
    return capture_result(loom_op_results(op)[0]);
  }
  if (std::holds_alternative<Bitcast>(operation_)) {
    loom_op_t* op;
    check(loom_vector_bitcast_build(builder, arguments[0].components().back(),
                                    input_type_, result_type_, location, &op));
    return capture_result(loom_op_results(op)[0]);
  }
  if (auto* extract = std::get_if<Extract>(&operation_)) {
    std::vector<loom_value_id_t> indices;
    std::vector<int64_t> static_indices;
    if (extract->static_indices.empty()) {
      indices.reserve(arguments.size() - 1);
      static_indices.assign(arguments.size() - 1,
                            std::numeric_limits<int64_t>::min());
      for (size_t index = 1; index < arguments.size(); ++index) {
        indices.push_back(cast_index(
            arguments[index],
            is_unsigned_component(extract->index_unsigned_mask, index - 1),
            builder, location));
      }
    } else {
      static_indices = extract->static_indices;
    }
    loom_op_t* op;
    check(loom_vector_extract_build(
        builder, arguments[0].components().back(), indices.data(),
        indices.size(), static_indices.data(), static_indices.size(),
        result_type_, location, &op));
    return capture_result(loom_op_results(op)[0]);
  }
  if (std::holds_alternative<Select>(operation_)) {
    loom_op_t* op;
    check(loom_vector_select_build(builder, arguments[0].components().back(),
                                   arguments[1].components().back(),
                                   arguments[2].components().back(),
                                   result_type_, location, &op));
    return capture_result(loom_op_results(op)[0]);
  }
  loom_op_t* op;
  check(std::visit(
      [&](auto operation) -> iree_status_t {
        using T = decltype(operation);
        if constexpr (std::is_same_v<T, Splat> || std::is_same_v<T, Slice> ||
                      std::is_same_v<T, Iota> || std::is_same_v<T, Concat> ||
                      std::is_same_v<T, Transpose> ||
                      std::is_same_v<T, Interleave> ||
                      std::is_same_v<T, Bitcast> ||
                      std::is_same_v<T, Extract> || std::is_same_v<T, Select>) {
          IREE_ASSERT_UNREACHABLE(
              "shaped result handled before static dispatch");
          IREE_BUILTIN_UNREACHABLE();
        } else if constexpr (std::is_same_v<T, TableLookup>) {
          return loom_vector_table_lookup_build(builder, arguments[0].ssa(),
                                                arguments[1].ssa(),
                                                result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, Absolute>) {
          if (operation.kind == Absolute::Kind::Float) {
            return loom_vector_absf_build(builder, operation.flags | math_flags,
                                          arguments[0].ssa(), result_type_,
                                          location, &op);
          }
          return loom_vector_absi_build(builder, arguments[0].ssa(),
                                        result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, FloatExtremum>) {
          auto build = operation.kind == FloatExtremum::Kind::Minimum
                           ? loom_vector_minimumf_build
                       : operation.kind == FloatExtremum::Kind::Maximum
                           ? loom_vector_maximumf_build
                       : operation.kind == FloatExtremum::Kind::MinNumber
                           ? loom_vector_minnumf_build
                           : loom_vector_maxnumf_build;
          return build(builder, operation.flags | math_flags,
                       arguments[0].ssa(), arguments[1].ssa(), result_type_,
                       location, &op);
        } else if constexpr (std::is_same_v<T, IntegerExtremum>) {
          auto build = operation.kind == IntegerExtremum::Kind::MinSigned
                           ? loom_vector_minsi_build
                       : operation.kind == IntegerExtremum::Kind::MaxSigned
                           ? loom_vector_maxsi_build
                       : operation.kind == IntegerExtremum::Kind::MinUnsigned
                           ? loom_vector_minui_build
                           : loom_vector_maxui_build;
          return build(builder, vector_value(0), vector_value(1), result_type_,
                       location, &op);
        } else if constexpr (std::is_same_v<T, Comparison>) {
          if (operation.kind == Comparison::Kind::Float) {
            return loom_vector_cmpf_build(builder, operation.flags | math_flags,
                                          operation.predicate, vector_value(0),
                                          vector_value(1), input_type_,
                                          result_type_, location, &op);
          }
          return loom_vector_cmpi_build(
              builder, operation.predicate, vector_value(0), vector_value(1),
              input_type_, result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, Dot4i>) {
          return loom_vector_dot4i_build(
              builder, *operation.kind, arguments[0].ssa(), arguments[1].ssa(),
              arguments[2].ssa(), result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, Dot2f>) {
          return loom_vector_dot2f_build(builder, arguments[0].ssa(),
                                         arguments[1].ssa(), arguments[2].ssa(),
                                         result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, Dotf>) {
          return loom_vector_dotf_build(builder, operation.flags | math_flags,
                                        arguments[0].ssa(), arguments[1].ssa(),
                                        arguments[2].ssa(), result_type_,
                                        location, &op);
        } else if constexpr (std::is_same_v<T, Reduction>) {
          uint8_t flags = loom_combining_kind_accepts_float(operation.kind)
                              ? operation.flags | math_flags
                              : 0;
          return loom_vector_reduce_build(builder, operation.kind, flags,
                                          vector_value(0), arguments[1].ssa(),
                                          result_type_, location, &op);
        }
      },
      operation_));
  return capture_result(loom_op_results(op)[0]);
}

bool ShapedIntrinsic::equivalent(const ShapedIntrinsic& other) const {
  return operation_ == other.operation_ &&
         loom_type_equal(result_type_, other.result_type_) &&
         result_vector_ == other.result_vector_ &&
         semantic_operands_ == other.semantic_operands_ &&
         loom_type_equal(input_type_, other.input_type_);
}

}  // namespace loom::cxx_import
