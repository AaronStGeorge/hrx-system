# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI invocation with real provider IO and captured process-facing streams."""

import io
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from subprocess import CompletedProcess
from unittest.mock import patch

from loom.model.fetch_hf import HuggingFace
from loom.model.fetch_main import main


def invoke(hub, arguments, *, cache_dir=None, error=None):
    """Calls the CLI against the real provider with explicit test-owned storage.

    Public Hub URLs remain publication identities; the provider's endpoint routes
    their IO to the loopback fixture. No environment or working-directory change
    invalidates the reusable transfer interpreters between independent cases.
    """
    output = io.StringIO()
    error = error if error is not None else io.StringIO()
    with (
        patch(
            "loom.model.fetch_storage.HuggingFace",
            partial(
                HuggingFace, endpoint=hub.endpoint, cache_dir=cache_dir or hub.cache_dir
            ),
        ),
        redirect_stdout(output),
        redirect_stderr(error),
    ):
        returncode = main(arguments)
    return CompletedProcess(arguments, returncode, output.getvalue(), error.getvalue())
