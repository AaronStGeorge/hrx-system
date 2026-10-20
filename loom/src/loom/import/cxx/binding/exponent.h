// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_EXPONENT_H_
#define LOOM_IMPORT_CXX_BINDING_EXPONENT_H_

#include <cxx/ast_fwd.h>

#include <optional>
#include <span>

#include "loom/import/cxx/value/storage.h"

namespace loom::cxx_import {

// Standard frexp/ldexp builtin conventions over the exact High operations.
// The frontend owns builtin identity, signatures, and argument conversions.
class ExponentBuiltin {
 public:
  static std::optional<ExponentBuiltin> resolve(Types& types,
                                                cxx::CallExpressionAST* call);

  // Consumes two already evaluated arguments. Frexp writes its second result
  // through the ordinary source storage projection and returns its fraction.
  Value call(std::span<const Value> arguments, Storage& storage,
             cxx::AST* owner, loom_builder_t* builder,
             loom_location_id_t location) const;

 private:
  ExponentBuiltin(loom_type_t type, const cxx::Type* exponent_type)
      : type_(type), exponent_type_(exponent_type) {}

  // Floating representation selected by the builtin signature.
  loom_type_t type_;
  // Source exponent pointee type for frexp; null for ldexp.
  const cxx::Type* exponent_type_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_EXPONENT_H_
