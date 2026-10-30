// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_KERNEL_LAUNCH_H_
#define LOOM_IMPORT_CXX_BINDING_KERNEL_LAUNCH_H_

#include <cxx/attributes.h>

#include <optional>
#include <string_view>

namespace loom::cxx_import {

class Diagnostics;
class LaunchContracts;
class Types;

// Source contract shared by check-runner and command-program kernel launches.
// Admission resolves the selected kernel, configuration, workload bundle, and
// ABI-shaped argument family once. Each body translator retains ownership of
// its source values: check cases materialize tensor storage while command
// programs forward native buffer and scalar values.
struct KernelLaunchIntrinsic {
  static bool supports(std::string_view name);
  static std::optional<KernelLaunchIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      LaunchContracts& launches, cxx::FunctionSymbol* function,
      const cxx::Attribute& attribute, cxx::AST* owner);

  bool equivalent(const KernelLaunchIntrinsic& other) const {
    return kernel == other.kernel && configuration == other.configuration;
  }

  // Statically selected concrete kernel declaration.
  cxx::FunctionSymbol* kernel;
  // Kernel-owned launch configuration, absent for fixed geometry.
  cxx::FunctionSymbol* configuration;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_KERNEL_LAUNCH_H_
