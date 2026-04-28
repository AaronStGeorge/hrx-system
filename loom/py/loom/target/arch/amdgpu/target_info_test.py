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
    validate_amdgpu_descriptor_set_isa_xml,
)


def test_descriptor_set_isa_xml_validation_accepts_matching_architecture() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="rdna4.xml")

    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("gfx12"), spec
    )
    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("gfx1250"), spec
    )


def test_descriptor_set_isa_xml_validation_rejects_mismatched_architecture() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="rdna4.xml")

    with pytest.raises(
        ValueError,
        match=(
            "amdgpu.gfx11.core expects AMD RDNA 3 architecture id 8, "
            "found AMD RDNA 4 architecture id 10"
        ),
    ):
        validate_amdgpu_descriptor_set_isa_xml(
            amdgpu_descriptor_set_info_by_generator_target("gfx11"), spec
        )


def test_descriptor_set_generator_target_lookup_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown AMDGPU descriptor generator target"):
        amdgpu_descriptor_set_info_by_generator_target("gfx999")
