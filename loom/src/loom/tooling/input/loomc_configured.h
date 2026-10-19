// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Build-configured foreign source imports through LoomC.

#ifndef LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_H_
#define LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_H_

#include "loom/tooling/input/loomc.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX
#define LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX 0
#endif  // LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX

// Returns configured foreign-language metadata without native loader callbacks.
loom_input_format_list_t loom_configured_input_loomc_formats(void);

// Returns the build-configured foreign-source importer, or NULL when none is
// linked. Final applications choose this composition; shared tooling accepts
// the optional callback.
#if LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX
loom_tooling_input_import_loomc_fn_t loom_configured_input_loomc_importer(void);
#else
#define loom_configured_input_loomc_importer() \
  ((loom_tooling_input_import_loomc_fn_t)NULL)
#endif  // LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_HAVE_CXX

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_INPUT_LOOMC_CONFIGURED_H_
