// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/numeric_conversion.h"

#include <cxx/names.h>
#include <cxx/types.h>

#include <cstddef>

#include "iree/base/api.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/import/cxx/value/vector.h"

namespace loom::cxx_import {
namespace {

bool is_numeric(cxx::TranslationUnit& unit, Types& types,
                const cxx::Type* type) {
  type = types.unqualified(type);
  return types.is_float(type) || (unit.typeTraits().is_integral(type) &&
                                  type->kind() != cxx::TypeKind::kBool);
}

bool same_shape(const VectorPartition& lhs, const VectorPartition& rhs) {
  if (lhs.extents.size() != rhs.extents.size()) {
    return false;
  }
  for (size_t axis = 0; axis < lhs.extents.size(); ++axis) {
    if (lhs.extents[axis] != rhs.extents[axis]) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool NumericConversionIntrinsic::supports_reference_body(
    std::string_view name) {
  return name == "numeric.convert";
}

std::optional<NumericConversionIntrinsic::Operation>
NumericConversionIntrinsic::admit(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics,
                                  const cxx::Attribute& attribute,
                                  cxx::AST* owner) {
  if (attribute.arguments.empty() ||
      !supports_reference_body(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner,
                       "numeric conversion has no semantic arguments");
  }
  return Operation{};
}

NumericConversionIntrinsic NumericConversionIntrinsic::resolve(
    Operation operation, cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    Types& types, const cxx::FunctionType* signature, cxx::AST* owner) {
  (void)operation;
  if (signature->isVariadic() || signature->parameterTypes().size() != 1) {
    diagnostics.reject(unit, owner,
                       "numeric conversion requires exactly one operand");
  }
  const auto* input = types.unqualified(signature->parameterTypes()[0]);
  const auto* output = types.unqualified(signature->returnType());
  const auto& input_partition = types.partition(input, owner);
  const auto& output_partition = types.partition(output, owner);

  if (input_partition.kind == ValueKind::Vector ||
      output_partition.kind == ValueKind::Vector) {
    if (input_partition.kind != ValueKind::Vector ||
        output_partition.kind != ValueKind::Vector) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires two scalars or two vectors");
    }
    const auto& input_vector =
        static_cast<const VectorPartition&>(input_partition);
    const auto& output_vector =
        static_cast<const VectorPartition&>(output_partition);
    if (!same_shape(input_vector, output_vector)) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires vectors with the same logical shape");
    }
    if (!is_numeric(unit, types, input_vector.element_type) ||
        !is_numeric(unit, types, output_vector.element_type)) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires integer or floating-point elements");
    }
    return NumericConversionIntrinsic(Representation::SemanticVector, input,
                                      output, &input_vector, &output_vector);
  }

  const auto* input_vector = types.native_vector(input);
  const auto* output_vector = types.native_vector(output);
  if (input_vector || output_vector) {
    if (!input_vector || !output_vector ||
        input_vector->elementCount() != output_vector->elementCount()) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires vectors with the same lane count");
    }
    if (!is_numeric(unit, types, input_vector->elementType()) ||
        !is_numeric(unit, types, output_vector->elementType())) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires integer or floating-point elements");
    }
    return NumericConversionIntrinsic(Representation::NativeVector, input,
                                      output);
  }

  if (input_partition.kind != ValueKind::SSA ||
      output_partition.kind != ValueKind::SSA ||
      !is_numeric(unit, types, input) || !is_numeric(unit, types, output)) {
    diagnostics.reject(
        unit, owner,
        "numeric conversion requires integer or floating-point scalars");
  }
  return NumericConversionIntrinsic(Representation::Scalar, input, output);
}

Value NumericConversionIntrinsic::call(std::span<const Value> arguments,
                                       ValueArena& arena, Scalars& scalars,
                                       Vectors& vectors,
                                       cxx::AST* owner) const {
  IREE_ASSERT(arguments.size() == 1);
  switch (representation_) {
    case Representation::Scalar:
      return Value(scalars.convert(arguments[0].ssa(), input_type_,
                                   output_type_, owner));
    case Representation::NativeVector:
      return Value(vectors.convert_elements(arguments[0].ssa(), input_type_,
                                            output_type_, owner));
    case Representation::SemanticVector:
      return vectors.convert_elements(arguments[0], *input_vector_,
                                      *output_vector_, arena, owner);
  }
  IREE_ASSERT_UNREACHABLE("unknown numeric conversion representation");
  IREE_BUILTIN_UNREACHABLE();
}

}  // namespace loom::cxx_import
