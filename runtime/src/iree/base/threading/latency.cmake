# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Native macOS activities use Foundation. Keep that dependency isolated from
# the basic threading primitives and enable Objective-C only on macOS builds.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  enable_language(OBJC)
  set(_IREE_LATENCY_PLATFORM_SRCS "latency_darwin.m")
  set(_IREE_LATENCY_PLATFORM_LIBS "-framework Foundation")
endif()
iree_cc_library(
  NAME
    latency
  HDRS
    "latency.h"
  SRCS
    "latency.c"
    "latency_win32.c"
    ${_IREE_LATENCY_PLATFORM_SRCS}
  COPTS
    "$<$<COMPILE_LANGUAGE:OBJC>:-fno-objc-arc>"
  DEPS
    iree::base
    ${_IREE_LATENCY_PLATFORM_LIBS}
  PUBLIC
)
unset(_IREE_LATENCY_PLATFORM_SRCS)
unset(_IREE_LATENCY_PLATFORM_LIBS)
