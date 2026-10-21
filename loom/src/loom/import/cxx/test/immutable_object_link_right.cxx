// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

static constexpr unsigned table[2] = {10, 20};

unsigned immutable_right_lookup(unsigned index) { return table[index & 1u]; }
