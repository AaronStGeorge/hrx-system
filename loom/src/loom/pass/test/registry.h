// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test pass descriptor registry.
//
// The test registry is a small, hermetic pass set for pass infrastructure
// tests. It intentionally avoids concrete production passes so parser,
// verifier, compiler, and interpreter tests can exercise pipeline mechanics
// without linking canonicalization, lowering, target, or dialect-specific pass
// behavior.

#ifndef LOOM_PASS_TEST_REGISTRY_H_
#define LOOM_PASS_TEST_REGISTRY_H_

#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TEST_PASS_TRACE_EVENT_CAPACITY 64

// One synthetic pass callback invocation captured by loom_test_pass_trace_t.
typedef struct loom_test_pass_trace_event_t {
  // Descriptor key of the pass callback that ran.
  iree_string_view_t pass_name;
  // Current module or function symbol name for the callback.
  iree_string_view_t symbol_name;
} loom_test_pass_trace_event_t;

// Optional per-entry user data consumed by the synthetic test passes.
typedef struct loom_test_pass_trace_t {
  // Number of times test.module-noop ran.
  int module_noop_invocation_count;
  // Number of times test.noop ran.
  int noop_invocation_count;
  // Number of times test.mark-changed ran.
  int mark_changed_invocation_count;
  // Number of times test.options ran.
  int options_invocation_count;
  // Number of times test.options create observed decoded options.
  int options_decoded_create_count;
  // Last decoded count option observed by test.options create.
  uint32_t decoded_options_count_value;
  // Last decoded mode enum index observed by test.options create.
  uint16_t decoded_options_mode_index;
  // Last decoded string option observed by test.options create.
  iree_string_view_t decoded_options_string_value;
  // Number of times test.fail ran before returning its intentional failure.
  int fail_invocation_count;
  // Number of populated entries in events.
  iree_host_size_t event_count;
  // Bounded chronological callback trace.
  loom_test_pass_trace_event_t events[LOOM_TEST_PASS_TRACE_EVENT_CAPACITY];
} loom_test_pass_trace_t;

// Returns a registry containing only synthetic test pass descriptors.
const loom_pass_registry_t* loom_test_pass_registry(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_TEST_REGISTRY_H_
