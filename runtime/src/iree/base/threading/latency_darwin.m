// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/latency.h"

#import <Foundation/Foundation.h>

iree_status_t iree_latency_scope_begin(iree_latency_scope_options_t options,
                                       iree_latency_scope_t* out_scope) {
  IREE_ASSERT_ARGUMENT(out_scope);
  out_scope->handle = 0;
  if (options.maximum_cpu_idle_latency_us < -1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU idle latency limit must be -1 or nonnegative");
  }
  if (options.maximum_cpu_idle_latency_us >= 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "macOS activities do not accept a numeric CPU "
                            "idle latency limit");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  @autoreleasepool {
    id activity = [[NSProcessInfo processInfo]
        beginActivityWithOptions:NSActivityUserInitiatedAllowingIdleSystemSleep |
                                 NSActivityLatencyCritical
                          reason:@"Latency-sensitive computation"];
    out_scope->handle = (uintptr_t)[activity retain];
  }
  IREE_TRACE_ZONE_END(z0);
  if (!out_scope->handle) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "requesting latency-critical activity failed");
  }
  return iree_ok_status();
}

iree_status_t iree_latency_scope_end(iree_latency_scope_t* scope) {
  if (!scope->handle) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  @autoreleasepool {
    id activity = (id)scope->handle;
    [[NSProcessInfo processInfo] endActivity:activity];
    [activity release];
  }
  scope->handle = 0;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
