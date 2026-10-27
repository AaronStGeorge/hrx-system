// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_BUFFER_MATERIALIZE_H_
#define LOOM_TRANSFORMS_BUFFER_MATERIALIZE_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materializes anonymous payloads as shared immutable executable storage and
// expands initialized allocations before storage and target planning. Workgroup
// copies partition the payload among participants and publish it with a
// barrier.
const loom_pass_info_t* loom_materialize_buffer_constants_pass_info(void);
iree_status_t loom_materialize_buffer_constants_run(loom_pass_t* pass,
                                                    loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_BUFFER_MATERIALIZE_H_
