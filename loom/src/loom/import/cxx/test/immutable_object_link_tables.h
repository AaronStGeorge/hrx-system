// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TEST_IMMUTABLE_OBJECT_LINK_TABLES_H_
#define LOOM_IMPORT_CXX_TEST_IMMUTABLE_OBJECT_LINK_TABLES_H_

namespace immutable_link {

inline constexpr unsigned bias[2] = {3, 5};
extern const unsigned weights[2];

}  // namespace immutable_link

#endif  // LOOM_IMPORT_CXX_TEST_IMMUTABLE_OBJECT_LINK_TABLES_H_
