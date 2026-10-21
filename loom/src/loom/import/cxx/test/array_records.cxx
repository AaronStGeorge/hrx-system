// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

struct Command {
  // Source word order is preserved through copies and element aliases.
  unsigned words[2];
};

static Command adjust(Command value, unsigned index) {
  value.words[index] += 3u;
  return value;
}

// A helper's parameter owns its array. The returned record must remain valid
// after the helper's storage is gone, independently of both caller copies.
static unsigned copies(unsigned index) {
  Command original{{37u, 41u}};
  Command saved = original;
  Command result = adjust(saved, index);
  saved.words[0] += 5u;
  return result.words[0] + 100u * result.words[1] + 10000u * saved.words[0] +
         1000000u * original.words[0];
}

// Initialization publishes each completed subobject before the next clause.
static unsigned ordered() {
  Command records[2] = {{{3u, records[0].words[0] + 2u}},
                        {{records[0].words[1] + 2u}}};
  Command value{{11u, value.words[0] + 2u}};
  return records[0].words[0] + 10u * records[0].words[1] +
         100u * records[1].words[0] + 1000u * records[1].words[1] +
         10000u * value.words[0] + 1000000u * value.words[1];
}

// A same-type prvalue initializes the destination object directly. Its later
// clauses observe earlier destination elements, just like a bare brace list.
static unsigned elided() {
  Command value = Command{{17u, value.words[0] + 2u}};
  Command wrapped{(Command{{23u, wrapped.words[0] + 2u}})};
  Command direct(Command{{31u, direct.words[0] + 2u}});
  return value.words[1] + 100u * wrapped.words[1] + 10000u * direct.words[1];
}

// Only the selected initializer runs, and its scalar mutations join the same
// surrounding bindings as an ordinary conditional expression.
static unsigned selected(unsigned choice) {
  unsigned marker = 3;
  Command value = choice ? Command{{++marker, value.words[0] + 2}}
                         : (marker += 7, Command{{13, value.words[0] + 2}});
  return value.words[0] + 100u * value.words[1] + 10000u * marker;
}

static unsigned memory(unsigned index) {
  Command records[2] = {{{5u, 7u}}, {{11u, 13u}}};
  Command saved = records[index];
  Command other = saved;
  Command* object = &saved;
  unsigned* member = object->words;
  member[1] += 17u;
  *object = *object;
  records[0] = saved;
  records[1] = other;
  return records[0].words[0] + 100u * records[0].words[1] +
         10000u * records[1].words[0] + 1000000u * records[1].words[1];
}

struct Nested {
  // Two independent dimensions retain their C++ element strides.
  unsigned matrix[2][2];
  // Record elements contain their own fixed arrays.
  Command commands[2];
  // Narrow signed values retain their storage width across copies.
  short narrow[2];
  // Boolean storage uses bytes while value transport uses predicates.
  bool flags[2];
};

static Nested forward(Nested value) { return value; }

static unsigned nested(unsigned index) {
  Nested value{};
  value.matrix[1][index] = 17;
  value.commands[index].words[1] = 19;
  value.narrow[index] = -7;
  value.flags[index] = true;
  Nested copy = forward(value);
  copy.matrix[1][index] += 2;
  copy.commands[index].words[1] += 3;
  copy.narrow[index] += 5;
  copy.flags[index] = false;
  return value.matrix[1][index] + copy.matrix[1][index] +
         value.commands[index].words[1] + copy.commands[index].words[1] +
         unsigned(copy.narrow[index] + 7) + value.flags[index] +
         copy.flags[index] + value.matrix[0][0] +
         value.commands[1u - index].words[0];
}

static unsigned temporary(unsigned index) {
  return adjust(Command{{5, 7}}, 1).words[index];
}

struct ConstArray {
  // Const element storage can be copied and read through dynamic subscripts.
  const unsigned words[2];
};
static unsigned constant_member(unsigned index) {
  ConstArray original{{3, 5}};
  ConstArray copy = original;
  return copy.words[index] + original.words[1u - index];
}

typedef unsigned U4 __attribute__((vector_size(16)));
static U4 forward_vector(U4 value) { return value; }

struct VectorRecord {
  // Explicit vectors remain vector components inside an array snapshot.
  U4 values[2];
};
static unsigned vector_elements(unsigned index) {
  VectorRecord original{{U4{1, 2, 3, 4}, U4{5, 6, 7, 8}}};
  VectorRecord copy = original;
  copy.values[index] += U4{1, 1, 1, 1};
  U4 observed = forward_vector(copy.values[index]);
  U4 first = original.values[0];
  U4 array[2] = {U4{11, 13, 17, 19}, U4{23, 29, 31, 37}};
  U4 selected = array[index];
  return observed[3] + first[0] + selected[0];
}

LOOM_CHECK_CASE(record_array_copies) {
  const auto first = copies(0);
  const auto second = copies(1);
  const auto memory_first = memory(0);
  const auto memory_second = memory(1);
  loom::check::expect_equal(first, 37424140u);
  loom::check::expect_equal(second, 37424437u);
  loom::check::expect_equal(memory_first, 7052405u);
  loom::check::expect_equal(memory_second, 13113011u);
}

LOOM_CHECK_CASE(record_array_initialization) {
  const auto sequence = ordered();
  const auto elided_sequence = elided();
  const auto selected_first = selected(1);
  const auto selected_second = selected(0);
  const auto first = nested(0);
  const auto second = nested(1);
  const auto temporary_first = temporary(0);
  const auto temporary_second = temporary(1);
  const auto constant = constant_member(1);
  const auto vector_first = vector_elements(0);
  const auto vector_second = vector_elements(1);
  loom::check::expect_equal(sequence, 13110753u);
  loom::check::expect_equal(elided_sequence, 332519u);
  loom::check::expect_equal(selected_first, 40604u);
  loom::check::expect_equal(selected_second, 101513u);
  loom::check::expect_equal(first, 83u);
  loom::check::expect_equal(second, 83u);
  loom::check::expect_equal(temporary_first, 5u);
  loom::check::expect_equal(temporary_second, 10u);
  loom::check::expect_equal(constant, 8u);
  loom::check::expect_equal(vector_first, 17u);
  loom::check::expect_equal(vector_second, 33u);
}
