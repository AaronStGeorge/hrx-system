# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Human-readable guidance from validated, author-supplied memory estimates."""


def format_bytes(value: int | None) -> str:
    """Formats an exact byte count or an explicitly unknown extent."""
    if value is None:
        return "unknown"
    if value < 1024:
        return f"{value} bytes"
    exponent = (value.bit_length() - 1) // 10
    unit = ("KiB", "MiB", "GiB", "TiB", "PiB", "EiB")[exponent - 1]
    return f"{value / 1024**exponent:.1f} {unit}"


def describe_memory(memory: dict) -> str:
    """Formats informational counts without inferring a total or hardware fit."""
    return (
        "Estimated memory: weights "
        + format_bytes(memory.get("weights_bytes"))
        + "; transients "
        + format_bytes(memory.get("transient_bytes"))
        + ". "
        + memory["workload"]
    )
