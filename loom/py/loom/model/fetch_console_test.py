# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Access-wait input and mode ownership, independent of provider transport."""

import io
import os
import signal
import threading
import unittest

import pytest

from loom.model.fetch_console import AccessWaiter
from loom.model.fetch_hf import AccessRequired


def _gate():
    return AccessRequired(
        "fixture/model", "gate", "https://example.com/fixture/model", "first", "login"
    )


@pytest.mark.parametrize("cancel", ["q", "interrupt", "eof"])
def test_terminal_cancellation_restores_input_mode(cancel):
    if os.name == "nt":
        raise unittest.SkipTest("this case exercises POSIX PTY mode restoration")
    import pty
    import termios

    master, slave = pty.openpty()

    class Controls(io.StringIO):
        def write(self, content):
            result = super().write(content)
            # Input is injected only after the waiter has entered terminal mode.
            if "[q / Ctrl-C] Cancel" in content:
                if cancel == "interrupt":
                    signal.raise_signal(signal.SIGINT)
                else:
                    os.write(master, b"q" if cancel == "q" else b"\x04")
            return result

    try:
        original = termios.tcgetattr(slave)
        with os.fdopen(slave, "r", closefd=False) as input:
            waiter = AccessWaiter(interactive=True, input=input, output=Controls())
            with pytest.raises(KeyboardInterrupt):
                waiter(_gate(), 5, threading.Event())
        assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


def test_headless_cancellation_does_not_read_input():
    cancelled = threading.Event()

    class Progress(io.StringIO):
        def write(self, content):
            result = super().write(content)
            if "Waiting for access" in content:
                cancelled.set()
            return result

    input = io.StringIO("q")
    output = Progress()
    AccessWaiter(interactive=False, input=input, output=output)(_gate(), 5, cancelled)
    assert input.tell() == 0
    assert "Waiting for access" in output.getvalue()
