// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/exponent.h"

#include <cxx/ast.h>
#include <cxx/names.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {

std::optional<ExponentBuiltin> ExponentBuiltin::resolve(
    Types& types, cxx::CallExpressionAST* call) {
  auto* callee = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
  using Builtin = cxx::BuiltinFunctionKind;
  const cxx::Type* exponent_type = nullptr;
  switch (cxx::resolveBuiltinFunctionKind(callee)) {
    case Builtin::T___BUILTIN_LDEXP:
    case Builtin::T___BUILTIN_LDEXPF:
    case Builtin::T___BUILTIN_LDEXPL:
      break;
    case Builtin::T___BUILTIN_FREXP:
    case Builtin::T___BUILTIN_FREXPF:
    case Builtin::T___BUILTIN_FREXPL:
      exponent_type = cxx::type_cast<cxx::PointerType>(
                          call->expressionList->next->value->type)
                          ->elementType();
      break;
    default:
      return std::nullopt;
  }
  return ExponentBuiltin(types.get(call->type, call), exponent_type);
}

Value ExponentBuiltin::call(std::span<const Value> arguments, Storage& storage,
                            cxx::AST* owner, loom_builder_t* builder,
                            loom_location_id_t location) const {
  loom_op_t* op;
  if (exponent_type_) {
    check(loom_scalar_frexpf_build(builder, arguments[0].ssa(), type_, location,
                                   &op));
    auto access = storage.dereference(
        storage.project(arguments[1].pointer(), exponent_type_, owner),
        exponent_type_, owner);
    storage.store(access, loom_op_results(op)[1], exponent_type_, owner);
  } else {
    check(loom_scalar_ldexpf_build(builder, arguments[0].ssa(),
                                   arguments[1].ssa(), type_, location, &op));
  }
  return Value(loom_op_results(op)[0]);
}

}  // namespace loom::cxx_import
