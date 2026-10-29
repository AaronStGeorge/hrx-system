# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime tracing provider configuration."""

load(":cc.bzl", "iree_runtime_cc_library")

def iree_runtime_tracing_provider(name, header, **kwargs):
    """Defines a provider whose public header implements the tracing contract.

    Bazel uses the standard instrumentation/allocation mode. Native builds use
    IREE_TRACING_MODE to select the same public tracing configuration.

    Args:
      name: Provider library name.
      header: Public tracing implementation header.
      **kwargs: Runtime C/C++ library attributes.
    """
    iree_runtime_cc_library(
        name = name,
        defines = [
            'IREE_TRACING_PROVIDER_H=\\\"%s\\\"' % header,
            "IREE_TRACING_MODE=2",
        ],
        **kwargs
    )
