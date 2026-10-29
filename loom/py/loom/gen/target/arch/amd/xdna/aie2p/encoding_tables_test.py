# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from loom.gen.target.arch.amd.xdna.aie2p import encoding_tables


@pytest.mark.parametrize(
    "slot",
    sorted({row.slot for row in encoding_tables.CORE_ENCODING_TABLE.instructions}),
)
def test_each_instruction_slot_requires_an_exact_singleton_bundle(slot: str) -> None:
    table = encoding_tables.CORE_ENCODING_TABLE
    without_singleton = replace(
        table,
        bundle_formats=tuple(row for row in table.bundle_formats if tuple(field.slot for field in row.fields) != (slot,)),
    )
    with pytest.raises(ValueError, match=f"slot '{slot}' has no exact singleton bundle format"):
        encoding_tables._validate_singleton_bundle_formats(without_singleton)


def test_check_validates_without_output() -> None:
    assert encoding_tables.main(["--check"]) == 0


@pytest.mark.parametrize(
    "outputs",
    [
        {"encoding": "encoding_tables.inl"},
        {"machine": "machine_tables.inl"},
        {
            "register": "register_tables.inl",
            "descriptor-header": "core_descriptors.h",
            "descriptor-source": "core_descriptors.c",
            "array-descriptor-header": "array_descriptors.h",
            "array-descriptor-source": "array_descriptors.c",
            "configuration-descriptor-header": "configuration_descriptors.h",
            "configuration-descriptor-source": "configuration_descriptors.c",
        },
    ],
)
def test_packages_generate_only_their_owned_outputs(tmp_path: Path, outputs: dict[str, str]) -> None:
    arguments = [argument for name, filename in outputs.items() for argument in (f"--{name}-output", str(tmp_path / filename))]
    assert encoding_tables.main(arguments) == 0
    assert {path.name for path in tmp_path.iterdir()} == set(outputs.values())
    assert all((tmp_path / filename).stat().st_size > 0 for filename in outputs.values())
