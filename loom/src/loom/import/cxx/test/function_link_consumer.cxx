// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

#include "function_link_helpers.h"

[[loom::symbol("library.adjust")]] unsigned adjust(unsigned value);

static unsigned local_adjust(unsigned value) { return value + 11u; }

unsigned apply_adjustment(unsigned value) {
  return adjust(value) + shared_inline(value) + shared_template<7u>(value) +
         local_adjust(value);
}

static unsigned observe_linked_views(unsigned position) {
  const unsigned char dense_values[4] = {2, 3, 5, 7};
  const unsigned char strided_values[8] = {11, 0, 13, 0, 17, 0, 19, 0};
  auto dense = loom::buffer::view<4>(dense_values, {},
                                     loom::encoding::layout::dense<1>());
  auto strided = loom::buffer::view<4>(strided_values, {},
                                       loom::encoding::layout::strided(2u));
  return linked_read_byte(dense, position) +
         linked_read_byte(strided, position);
}

static unsigned observe_linked_dynamic_view() {
  const unsigned char values[6] = {29, 31, 37, 41, 43, 47};
  const auto count = loom::index::cast(4u);
  auto view = loom::buffer::view<loom::type::dynamic>(
      values, {count}, loom::encoding::layout::dense<1>());
  return linked_read_dynamic_byte(view);
}

static unsigned observe_linked_crop(unsigned columns) {
  unsigned char values[46];
  for (unsigned index = 0; index < 46u; ++index) {
    values[index] = static_cast<unsigned char>(index);
  }
  const unsigned char* input = values + 2;
  auto source = loom::buffer::view<4, loom::type::dynamic>(
      input, {columns}, loom::encoding::layout::dense<2>());
  auto crop = linked_crop_2x3(source);
  return loom::view::load(crop, 1u, 1u);
}

LOOM_CHECK_CASE(function_link) {
  const auto adjusted = apply_adjustment(2u);
  const auto first = observe_linked_views(0u);
  const auto last = observe_linked_views(3u);
  const auto dynamic = observe_linked_dynamic_view();
  const auto narrow_crop = observe_linked_crop(8u);
  const auto wide_crop = observe_linked_crop(11u);
  loom::check::expect_equal(adjusted, 56u);
  loom::check::expect_equal(first, 13u);
  loom::check::expect_equal(last, 26u);
  loom::check::expect_equal(dynamic, 41u);
  loom::check::expect_equal(narrow_crop, 21u);
  loom::check::expect_equal(wide_crop, 27u);
}
