// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

using loom::kernel::scan_direction;
using loom::kernel::scan_mode;

// Three independent workgroups each end in a partial physical subgroup. Five
// columns exercise every prefix mode and the workgroup reduction.
#define DEFINE_SCAN_KERNEL(Name, Type, Kind)                                 \
  [[loom::kernel, loom::workgroup_count(3, 1, 1),                            \
    loom::workgroup_size(80, 1, 1)]]                                         \
  void Name(const Type* input, Type* output) {                               \
    unsigned item =                                                          \
        loom::kernel::workgroup::id.x * 80u + loom::kernel::workitem::id.x;  \
    Type value = input[item + 1];                                            \
    output[item * 5 + 1] = loom::kernel::workgroup::scan::Kind(value);       \
    output[item * 5 + 2] =                                                   \
        loom::kernel::workgroup::scan::Kind<scan_mode::exclusive>(value);    \
    output[item * 5 + 3] =                                                   \
        loom::kernel::workgroup::scan::Kind<scan_mode::inclusive,            \
                                            scan_direction::reverse>(value); \
    output[item * 5 + 4] =                                                   \
        loom::kernel::workgroup::scan::Kind<scan_mode::exclusive,            \
                                            scan_direction::reverse>(value); \
    output[item * 5 + 5] = loom::kernel::workgroup::reduce::Kind(value);     \
  }

DEFINE_SCAN_KERNEL(scan_u32, unsigned, addi)
DEFINE_SCAN_KERNEL(scan_u64, unsigned long long, addi)
DEFINE_SCAN_KERNEL(scan_f32, float, addf)

#undef DEFINE_SCAN_KERNEL
