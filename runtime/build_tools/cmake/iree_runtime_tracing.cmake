# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Compose the configured provider without duplicating its BUILD declaration.
function(iree_runtime_configure_tracing)
  set(_PROVIDER_DEPS)
  set(_PROVIDER_DEFINES)
  if(IREE_ENABLE_RUNTIME_TRACING AND IREE_TRACING_MODE GREATER 0)
    if(IREE_TRACING_PROVIDER MATCHES "^(console|tracy)$")
      list(APPEND _PROVIDER_DEPS "iree::base::tracing::${IREE_TRACING_PROVIDER}")
    else()
      if(NOT IREE_TRACING_PROVIDER_H)
        message(FATAL_ERROR "Custom tracing providers require IREE_TRACING_PROVIDER_H")
      endif()
      list(APPEND _PROVIDER_DEPS "${IREE_TRACING_PROVIDER}")
      list(APPEND _PROVIDER_DEFINES
        "IREE_TRACING_PROVIDER_H=\"${IREE_TRACING_PROVIDER_H}\""
        "IREE_TRACING_MODE=${IREE_TRACING_MODE}")
    endif()
  endif()
  iree_cc_library(NAME provider DEPS ${_PROVIDER_DEPS}
    DEFINES ${_PROVIDER_DEFINES} PUBLIC)
endfunction()
