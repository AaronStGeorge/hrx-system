// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_PIPELINE_H_
#define LOOMCXX_PIPELINE_H_

#include <loomcxx/encoding_type.h>

namespace loom::channel {

// Delivery order for records admitted to a channel.
enum class discipline : unsigned char {
  // Reservations and acceptances observe first-in, first-out order.
  fifo = 1,
};

// Permission carried by an owned consuming access.
enum class mode : unsigned char {
  // Immutable payload access. This is the unqualified read type in Loom IR.
  immutable = 0,
  // Exclusive initialized payload access permitting mutation.
  mutable_ = 1,
};

}  // namespace loom::channel

namespace loom::type {

// A tile-level aggregate whose dimensions precede its element type, matching
// Loom IR. Each dynamic dimension and optional encoding travel with the value
// through functions, records, and structured control flow.
template <class Shape, class T, class Encoding = void>
class [[loom::type("tile")]] tile {};

// A borrowed allocation resource selected for one enclosing invocation.
class [[loom::type("pool")]] pool {};

// A communication identity carrying records of Payload. Storage, capacity,
// and transport belong to its binding rather than this handle type.
template <class Payload>
class [[loom::type("channel")]] channel {};

// Owned consumption of one channel record. Immutable access is the default;
// mutable access remains exclusive until released or transferred.
template <class Payload,
          loom::channel::mode Mode = loom::channel::mode::immutable>
class [[loom::type("read")]] read {};

// Owned producer reservation for one channel record. Publication is explicit.
template <class Payload>
class [[loom::type("write")]] write {};

}  // namespace loom::type

#endif  // LOOMCXX_PIPELINE_H_
