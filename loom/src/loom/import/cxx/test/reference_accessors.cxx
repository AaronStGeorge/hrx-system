// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

static unsigned local_alias_value(unsigned value) {
  unsigned& alias = value;
  alias += 3u;
  alias *= 2u;
  return alias + value;
}

static unsigned& at(unsigned* values, unsigned index) { return values[index]; }
static unsigned observe(volatile unsigned& value) { return value; }
static unsigned first(const unsigned (&values)[4]) { return values[0]; }
static void escape(unsigned& value) {
  unsigned* pointer = &value;
  ++*pointer;
}
static unsigned& identity(unsigned& value) { return value; }
static unsigned& forward(unsigned& value) { return identity(value); }

static unsigned storage_alias(unsigned sample, unsigned input, unsigned index) {
  volatile unsigned observed = sample;
  unsigned values[4] = {input, 3u, 5u, 7u};
  unsigned& element = at(values, index);
  element += first(values);
  escape(element);
  return observe(observed) + element;
}

static unsigned forwarded_alias(unsigned value) {
  forward(value) += 3u;
  return value;
}

template <typename T, int Low, int High>
struct Array1D {
  T data[High - Low + 1];
  T& operator()(int index) { return data[index - Low]; }
  const T& operator()(int index) const { return data[index - Low]; }
  T& operator[](int index) { return data[index - Low]; }
};

static unsigned array_access(unsigned value, int index) {
  Array1D<unsigned, 1, 4> array{};
  array[index] = value;
  array(index) += 2u;
  const auto& observed = array;
  return observed(index);
}

template <typename T, unsigned Rows, unsigned Columns>
struct Matrix {
  T (&values)[Rows][Columns];
  T& operator()(unsigned row, unsigned column) { return values[row][column]; }
  const T& operator()(unsigned row, unsigned column) const {
    return values[Rows - 1u - row][column];
  }
};

static unsigned matrix_access(unsigned value, unsigned row, unsigned column) {
  unsigned values[2][3] = {};
  Matrix<unsigned, 2, 3> matrix{values};
  matrix(row, column) = value;
  const auto& observed = matrix;
  return matrix(row, column) + observed(1u - row, column);
}

struct DefaultReference {
  unsigned* pointer;
  unsigned& alias = *pointer;
};

static unsigned default_reference(unsigned* values, unsigned amount) {
  DefaultReference reference{values};
  reference.alias += amount;
  return *reference.pointer;
}

static unsigned default_reference_value(unsigned input, unsigned amount) {
  unsigned value = input;
  const unsigned result = default_reference(&value, amount);
  return result * 100u + value;
}

LOOM_CHECK_CASE(reference_accessors) {
  const auto local = local_alias_value(4u);
  const auto storage = storage_alias(11u, 4u, 2u);
  const auto forwarded = forwarded_alias(7u);
  const auto array = array_access(13u, 3);
  const auto matrix = matrix_access(17u, 1u, 2u);
  const auto defaulted = default_reference_value(19u, 5u);
  loom::check::expect_equal(local, 28u);
  loom::check::expect_equal(storage, 21u);
  loom::check::expect_equal(forwarded, 10u);
  loom::check::expect_equal(array, 15u);
  loom::check::expect_equal(matrix, 34u);
  loom::check::expect_equal(defaulted, 2424u);
}
