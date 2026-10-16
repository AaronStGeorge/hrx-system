// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GetProcessInformation and SetProcessInformation require Windows 8
// declarations.
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif

#include "iree/base/threading/latency.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <windows.h>

// Windows exposes one process policy instead of independent request handles.
// Only scope transitions access this state; submission and waits never do.
static struct {
  // Serializes native policy changes and overlapping scope ownership.
  SRWLOCK mutex;
  // Number of scopes currently requesting HighQoS.
  uintptr_t scope_count;
  // Execution-speed policy saved before the first scope began.
  PROCESS_POWER_THROTTLING_STATE saved_policy;
} iree_latency_process_state = {SRWLOCK_INIT, 0, {0}};

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
                            "Windows HighQoS does not accept a numeric CPU "
                            "idle latency limit");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  DWORD error = ERROR_SUCCESS;
  AcquireSRWLockExclusive(&iree_latency_process_state.mutex);
  if (iree_latency_process_state.scope_count == 0) {
    PROCESS_POWER_THROTTLING_STATE policy = {
        .Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
    };
    if (!GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                               &policy, sizeof(policy))) {
      error = GetLastError();
    } else {
      iree_latency_process_state.saved_policy = policy;
      policy.ControlMask |= PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      policy.StateMask &= ~PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                 &policy, sizeof(policy))) {
        error = GetLastError();
      }
    }
  }
  if (error == ERROR_SUCCESS) {
    ++iree_latency_process_state.scope_count;
    out_scope->handle = 1;
  }
  ReleaseSRWLockExclusive(&iree_latency_process_state.mutex);
  IREE_TRACE_ZONE_END(z0);
  return error == ERROR_SUCCESS
             ? iree_ok_status()
             : iree_make_status(iree_status_code_from_win32_error(error),
                                "requesting process HighQoS failed (%lu)",
                                error);
}

iree_status_t iree_latency_scope_end(iree_latency_scope_t* scope) {
  if (!scope->handle) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  DWORD error = ERROR_SUCCESS;
  AcquireSRWLockExclusive(&iree_latency_process_state.mutex);
  if (iree_latency_process_state.scope_count == 1) {
    PROCESS_POWER_THROTTLING_STATE policy = {
        .Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
    };
    if (!GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                               &policy, sizeof(policy))) {
      error = GetLastError();
    } else {
      // Restore our execution-speed override while preserving other controls
      // the application may have changed, such as timer-resolution handling.
      const DWORD mask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      policy.ControlMask =
          (policy.ControlMask & ~mask) |
          (iree_latency_process_state.saved_policy.ControlMask & mask);
      policy.StateMask =
          (policy.StateMask & ~mask) |
          (iree_latency_process_state.saved_policy.StateMask & mask);
      if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                 &policy, sizeof(policy))) {
        error = GetLastError();
      }
    }
  }
  if (error == ERROR_SUCCESS) {
    --iree_latency_process_state.scope_count;
    scope->handle = 0;
  }
  ReleaseSRWLockExclusive(&iree_latency_process_state.mutex);
  IREE_TRACE_ZONE_END(z0);
  return error == ERROR_SUCCESS
             ? iree_ok_status()
             : iree_make_status(iree_status_code_from_win32_error(error),
                                "restoring process power policy failed (%lu)",
                                error);
}

#endif  // IREE_PLATFORM_WINDOWS
