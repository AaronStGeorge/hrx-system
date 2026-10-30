// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_CHANNEL_H_
#define LOOM_IMPORT_CXX_BINDING_CHANNEL_H_

#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

// One concrete channel operation specialization. Admission retains the exact
// payload, ownership capability, and borrowed-view contracts; calls only bind
// those contracts to already evaluated source values.
class ChannelIntrinsic {
 public:
  // Returns whether |name| belongs to the complete channel operation family.
  static bool supports(std::string_view name);

  // Resolves a concrete specialization. Unknown operation names return no
  // value; malformed recognized declarations diagnose at |owner|.
  static std::optional<ChannelIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  // Emits the admitted operation. Value absence is a handled consuming void
  // operation, not an unrecognized intrinsic.
  std::optional<Value> call(std::span<const Value> arguments, ValueArena& arena,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

  // Compares the complete concrete source contract for redeclaration checks.
  bool equivalent(const ChannelIntrinsic& other) const;

 private:
  enum class Operation {
    Bind,
    Reserve,
    Accept,
    Wait,
    Acquire,
    Publish,
    Release,
    Fanout,
    Copy,
    Select,
  };

  static std::optional<Operation> parse_operation(std::string_view name);

  explicit ChannelIntrinsic(Operation operation) : operation_(operation) {}

  // Selected operation kind.
  Operation operation_;
  // FIFO or another descriptor-validated binding discipline.
  uint8_t discipline_ = 0;
  // Slot storage consumed by channel.bind.
  const ViewPartition* storage_view_ = nullptr;
  // Channel-family operand carrying the source payload and ownership kind.
  const ChannelPartition* source_handle_ = nullptr;
  // Channel-family result produced by bind, reserve, accept, acquire, fanout,
  // or select.
  const ChannelPartition* result_handle_ = nullptr;
  // Borrowed payload view produced by reserve, wait, or acquire.
  const ViewPartition* result_view_ = nullptr;
  // Source aggregate containing multiple operation results.
  const RecordPartition* result_record_ = nullptr;
  // Number of flattened immutable reads returned by fanout.
  size_t fanout_count_ = 0;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_CHANNEL_H_
