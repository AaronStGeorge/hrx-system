// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

#include "function_link_helpers.h"

static unsigned local_adjust(unsigned value) { return value + 5u; }

[[loom::symbol("library.adjust")]] unsigned adjust(unsigned value) {
  return shared_inline(value) + shared_template<7u>(value) +
         local_adjust(value);
}

[[loom::symbol("library.read_byte")]] unsigned linked_read_byte(
    LinkedByteView values, unsigned position) {
  loom::assume(position < 4u);
  return loom::view::load(values, position);
}

[[loom::symbol("library.read_dynamic_byte")]] unsigned linked_read_dynamic_byte(
    LinkedDynamicByteView values) {
  return loom::view::load(values, 3u);
}

[[loom::symbol("library.crop_2x3")]] LinkedCrop linked_crop_2x3(
    LinkedCropSource source) {
  return loom::view::subview<2, 3>(source, {1u, 2u}, {});
}
