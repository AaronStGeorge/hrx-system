// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TEST_FUNCTION_LINK_HELPERS_H_
#define LOOM_IMPORT_CXX_TEST_FUNCTION_LINK_HELPERS_H_

#include <loomcxx/view.h>

inline unsigned shared_inline(unsigned value) { return value * 3u + 1u; }

template <unsigned Bias>
constexpr unsigned shared_template(unsigned value) {
  return value * 2u + Bias;
}

using LinkedByteView =
    loom::type::view<loom::type::shape<4>, const unsigned char,
                     loom::type::encoding<loom::encoding::role::layout, 1>>;
using LinkedDynamicByteView =
    loom::type::view<loom::type::shape<loom::type::dynamic>,
                     const unsigned char,
                     loom::type::encoding<loom::encoding::role::layout, 1>>;
using LinkedCropSource =
    loom::type::view<loom::type::shape<4, loom::type::dynamic>,
                     const unsigned char,
                     loom::type::encoding<loom::encoding::role::layout, 2>>;
using LinkedCrop =
    loom::type::view<loom::type::shape<2, 3>, const unsigned char,
                     loom::type::encoding<loom::encoding::role::layout, 2>>;

[[loom::symbol("library.read_byte")]] unsigned linked_read_byte(
    LinkedByteView values, unsigned position);
[[loom::symbol("library.read_dynamic_byte")]] unsigned linked_read_dynamic_byte(
    LinkedDynamicByteView values);
[[loom::symbol("library.crop_2x3")]] LinkedCrop linked_crop_2x3(
    LinkedCropSource source);

#endif  // LOOM_IMPORT_CXX_TEST_FUNCTION_LINK_HELPERS_H_
