# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from loom.gen.low_descriptor_shard import main
from loom.target.descriptor_sets import (
    descriptor_set_names,
    iter_checked_in_c_descriptor_sets,
    resolve_descriptor_set,
)


def test_resolve_descriptor_set_by_key_and_alias() -> None:
    by_key = resolve_descriptor_set("wasm.core.simd128")
    by_alias = resolve_descriptor_set("wasm_core_simd128")

    assert by_key is by_alias
    assert by_key.function_name == "loom_wasm_core_simd128_descriptor_set"


def test_descriptor_set_names_include_keys_and_aliases() -> None:
    names = descriptor_set_names()

    assert "iree.vm.core" in names
    assert "ireevm_core" in names
    assert "test.low.core" in names
    assert "test_low_core" in names


def test_checked_in_c_generation_excludes_build_generated_shards() -> None:
    keys = {descriptor_set.key for descriptor_set in iter_checked_in_c_descriptor_sets()}

    assert "wasm.core.simd128" not in keys
    assert "x86.avx512.core" not in keys
    assert "x86.packed_dot.core" not in keys
    assert "test.low.core" in keys


def test_main_generates_selected_descriptor_set(tmp_path: Path) -> None:
    source_path = tmp_path / "test_descriptors.c"
    header_path = tmp_path / "test_descriptors.h"

    assert (
        main(
            [
                "--descriptor-set=test_low_core",
                f"--source={source_path}",
                f"--header={header_path}",
            ]
        )
        == 0
    )

    assert "loom_test_low_core_descriptor_set" in header_path.read_text(encoding="utf-8")
    source = source_path.read_text(encoding="utf-8")
    assert 'LOOM_BSTRING_LITERAL("\\x0d", "test.low.core")' in source
    assert "test.spv.op_iadd.i32" in source


def test_main_reports_missing_descriptor_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--descriptor-set=missing.core",
                f"--source={tmp_path / 'missing.c'}",
                f"--header={tmp_path / 'missing.h'}",
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "unknown low descriptor set 'missing.core'" in captured.err
    assert "wasm.core.simd128" in captured.err
