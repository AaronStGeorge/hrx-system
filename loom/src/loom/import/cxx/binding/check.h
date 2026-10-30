// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_CHECK_H_
#define LOOM_IMPORT_CXX_BINDING_CHECK_H_

#include <cxx/attributes.h>

#include <cstdint>
#include <optional>
#include <string_view>

#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

class LaunchContracts;

// Source contracts for the shared check runner's generation, invocation and
// observation operations. Declaration admission retains their concrete types
// and kernel identity; check-body translation owns constant arguments and IR.
struct CheckIntrinsic {
  enum class Operation {
    Equal,
    Approximate,
    CloseSigned,
    CloseUnsigned,
    Fill,
    Iota,
    Slice,
    Bitwise,
    Requires,
    Event,
    Trial,
    Generate,
    Compare,
    Invoke,
    EntropyFork,
    EntropyRead,
  };

  static std::optional<Operation> parse_operation(std::string_view name);
  static std::optional<CheckIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      LaunchContracts& launches, cxx::FunctionSymbol* function,
      const cxx::Attribute& attribute, cxx::AST* owner);

  bool equivalent(const CheckIntrinsic& other) const;
  bool is_observation() const {
    return operation == Operation::Equal ||
           operation == Operation::Approximate ||
           operation == Operation::CloseSigned ||
           operation == Operation::CloseUnsigned ||
           operation == Operation::Bitwise || operation == Operation::Event;
  }

  // Operation selected by the declaration's explicit binding.
  Operation operation;
  // Admitted operand type for scalar observations.
  loom_type_t scalar_type = {};
  // Admitted expected operand type for independently typed observations.
  loom_type_t expected_scalar_type = {};
  // Tensor source for slicing or observation, owned by Types.
  const TensorPartition* source_tensor = nullptr;
  // Expected tensor for independently typed observations, owned by Types.
  const TensorPartition* expected_tensor = nullptr;
  // Tensor result for generation or slicing, owned by Types.
  const TensorPartition* result_tensor = nullptr;
  // Ordinary generator or target subject selected by a template argument.
  cxx::FunctionSymbol* subject = nullptr;
  // Optional distinct oracle selected by a second template argument.
  cxx::FunctionSymbol* oracle = nullptr;
  // Finite trial count selected by trial<Count>.
  int64_t trial_count = 0;
  // Kernel-owned launch configuration, if one supplies this launch's geometry.
  cxx::FunctionSymbol* configuration = nullptr;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_CHECK_H_
