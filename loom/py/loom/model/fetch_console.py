# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dependency-free console input and cancellable acquisition access waits."""

from __future__ import annotations

import os
import threading
import time
import webbrowser
from typing import TextIO

from .fetch_hf import AccessRequired
from .fetch_source import FetchError


def display_text(text: str) -> str:
    """Renders metadata as one terminal line without interpreting controls."""
    return "".join(
        character if character.isprintable() else f"\\u{ord(character):04x}"
        for character in text
    )


class TerminalInput:
    """Owns temporary terminal input mode for one interactive operation."""

    def __init__(self, stream: TextIO):
        # The caller's input terminal; output remains independently owned.
        self.stream = stream
        # POSIX attributes restored on success, cancellation, or exception.
        self.saved_mode = None
        # Windows CRT key functions, loaded only on that platform.
        self.windows = None
        # Extended Windows keys have a prefix followed by a scan code.
        self.extended_key = False

    def __enter__(self):
        if not self.stream.isatty():
            raise FetchError("interactive input requires a terminal; use --no-input")
        if os.name == "nt":
            import msvcrt

            self.windows = msvcrt
        else:
            import termios
            import tty

            self.saved_mode = termios.tcgetattr(self.stream.fileno())
            tty.setcbreak(self.stream.fileno(), termios.TCSANOW)
        return self

    def __exit__(self, *_):
        if self.saved_mode is not None:
            import termios

            termios.tcsetattr(self.stream.fileno(), termios.TCSANOW, self.saved_mode)

    def read(self, timeout: float, cancelled: threading.Event) -> str | None:
        deadline = time.monotonic() + timeout
        while not cancelled.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if self.windows is not None:
                if self.windows.kbhit():
                    key = self.windows.getwch()
                    if self.extended_key:
                        self.extended_key = False
                    elif key in ("\0", "\xe0"):
                        self.extended_key = True
                    elif key in ("\x03", "\x1a"):
                        raise KeyboardInterrupt
                    else:
                        return key
                cancelled.wait(min(remaining, 0.1))
            else:
                import select

                ready, _, _ = select.select(
                    [self.stream.fileno()], [], [], min(remaining, 0.1)
                )
                if ready:
                    key = os.read(self.stream.fileno(), 1)
                    if not key or key in (b"\x03", b"\x04"):
                        raise KeyboardInterrupt
                    return key.decode("ascii", errors="replace")
        return None


class AccessWaiter:
    """Plain access interaction; provider decisions remain in the acquisition API."""

    def __init__(self, *, interactive: bool, input: TextIO, output: TextIO):
        # Headless waiting never reads keys or opens a browser.
        self.interactive = interactive
        # Caller-owned input terminal, used only for an interactive wait.
        self.input = input
        # Status output is independent of machine-readable command stdout.
        self.output = output
        # Repeated polls print status without repeating unchanged remediation.
        self.last_action = None

    def _line(self, text: str):
        self.output.write(display_text(text) + "\n")
        self.output.flush()

    def _show(self, action: AccessRequired):
        key = (
            action.repository,
            action.reason,
            action.url,
            action.account,
            action.credential_source,
        )
        if key == self.last_action:
            return
        self.last_action = key
        self._line(str(action))
        if action.account:
            self._line(f"Effective account: {action.account}")
        if action.reason == "gate":
            self._line(f"Open {action.url} and complete the access form if you agree.")
            self._line(
                "Use the downloader's account; the publisher may need "
                "to approve access."
            )
        if action.reason in ("gate", "credentials", "permissions"):
            self._line(
                "Run hf auth login in another terminal with an authorized account."
            )
            if action.credential_source in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                self._line(
                    f"{action.credential_source} overrides the saved login. "
                    "To change it, restart this command with that variable "
                    "updated or unset."
                )
            elif action.credential_source == "HF_HUB_DISABLE_IMPLICIT_TOKEN":
                self._line(
                    "HF_HUB_DISABLE_IMPLICIT_TOKEN disables the saved login; "
                    "unset it before restarting this command."
                )

    def __call__(
        self, action: AccessRequired, delay: float, cancelled: threading.Event
    ) -> None:
        self._show(action)
        self._line(f"Waiting for access; rechecking in {delay:g} seconds.")
        deadline = time.monotonic() + delay
        if not self.interactive:
            while not cancelled.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                cancelled.wait(min(remaining, 0.25))
            return
        minimum = time.monotonic() + action.retry_after
        with TerminalInput(self.input) as terminal:
            self._line("[o] Open page   [Enter] Check now   [q / Ctrl-C] Cancel")
            while not cancelled.is_set():
                key = terminal.read(max(0, deadline - time.monotonic()), cancelled)
                if key is None:
                    return
                if key.lower() == "q":
                    cancelled.set()
                    raise KeyboardInterrupt
                if key in ("\r", "\n"):
                    remaining = minimum - time.monotonic()
                    if remaining <= 0:
                        return
                    self._line(
                        f"Provider retry delay: {remaining:.0f} seconds remaining."
                    )
                elif key.lower() == "o":
                    if not webbrowser.open(action.url, new=2):
                        self._line(f"Open this URL in your browser: {action.url}")
