# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.target.arch.amdgpu.isa_xml import parse_amdgpu_isa_xml_text
from loom.target.arch.amdgpu.isa_xml_test import SAMPLE_XML
from loom.target.arch.amdgpu.target_info import (
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
    validate_amdgpu_descriptor_set_isa_xml,
)


def test_descriptor_set_isa_xml_validation_accepts_matching_architecture() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="rdna4.xml")

    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("rdna4"), spec
    )
    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("rdna4_gfx125x"), spec
    )


def test_descriptor_set_isa_xml_validation_rejects_mismatched_architecture() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="rdna4.xml")

    with pytest.raises(
        ValueError,
        match=(
            "amdgpu.rdna3.core expects AMD RDNA 3 architecture id 8, "
            "found AMD RDNA 4 architecture id 10"
        ),
    ):
        validate_amdgpu_descriptor_set_isa_xml(
            amdgpu_descriptor_set_info_by_generator_target("rdna3"), spec
        )


def test_descriptor_set_generator_target_lookup_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown AMDGPU descriptor generator target"):
        amdgpu_descriptor_set_info_by_generator_target("gfx999")


def test_descriptor_set_storage_target_lookup_classifies_views() -> None:
    assert (
        amdgpu_descriptor_set_storage_info_by_generator_target(
            "rdna4_gfx125x"
        ).generator_target
        == "rdna4"
    )
    assert (
        amdgpu_descriptor_set_storage_info_by_generator_target("cdna3").generator_target
        == "cdna3"
    )

    view_infos = amdgpu_descriptor_set_view_infos_by_storage_generator_target("rdna4")
    assert [info.generator_target for info in view_infos] == ["rdna4_gfx125x"]
    assert amdgpu_descriptor_set_view_infos_by_storage_generator_target("cdna3") == ()


def test_descriptor_set_view_lookup_rejects_view_storage_target() -> None:
    with pytest.raises(ValueError, match="is a view target, not a storage target"):
        amdgpu_descriptor_set_view_infos_by_storage_generator_target("rdna4_gfx125x")
