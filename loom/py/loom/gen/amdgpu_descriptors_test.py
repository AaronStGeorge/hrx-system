# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from loom.gen import amdgpu_descriptors
from loom.target.low_descriptors import Descriptor, DescriptorSet


def _descriptor(index: int) -> Descriptor:
    return Descriptor(
        key=f"amdgpu.test{index}",
        mnemonic=None,
        semantic_tag=None,
        operands=(),
        schedule_class="amdgpu.test",
    )


def _descriptor_set(target: str, descriptor_count: int) -> DescriptorSet:
    return DescriptorSet(
        key=f"amdgpu.{target.replace('_', '.')}.core",
        target_key="amdgpu",
        feature_key=f"amdgpu.{target}.v1",
        c_header_path=Path(f"{target}_descriptors.h"),
        c_source_path=Path(f"{target}_descriptors.c"),
        header_guard=f"{target.upper()}_DESCRIPTORS_H_",
        public_header=f"loom/target/arch/amdgpu/{target}_descriptors.h",
        function_name=f"loom_amdgpu_{target}_core_descriptor_set",
        c_table_prefix=f"Amdgpu{target.title()}Core",
        c_enum_prefix=f"AMDGPU_{target.upper()}_CORE",
        generator_version=1,
        reg_classes=(),
        resources=(),
        schedule_classes=(),
        descriptors=tuple(_descriptor(index) for index in range(descriptor_count)),
    )


def test_storage_generation_reuses_parsed_isa_for_declared_views() -> None:
    parsed_spec = object()
    parse_calls: list[Path] = []
    build_calls: list[tuple[str, object]] = []

    def parse_xml(path: Path) -> object:
        parse_calls.append(path)
        return parsed_spec

    def build_descriptor_set(target: str, spec: object) -> DescriptorSet:
        build_calls.append((target, spec))
        descriptor_count = 2 if target == "rdna4_gfx125x" else 1
        return _descriptor_set(target, descriptor_count)

    def generate_descriptor_set(descriptor_set: DescriptorSet, format_output: bool) -> SimpleNamespace:
        return SimpleNamespace(header=f"// {descriptor_set.key}\n")

    def generate_descriptor_set_shared_source(
        storage_descriptor_set: DescriptorSet,
        view_descriptor_sets: tuple[DescriptorSet, ...],
        format_output: bool,
    ) -> str:
        return "// shared\n"

    with (
        TemporaryDirectory() as temporary_directory,
        mock.patch.object(amdgpu_descriptors, "parse_amdgpu_isa_xml_path", parse_xml),
        mock.patch.object(
            amdgpu_descriptors,
            "build_amdgpu_core_descriptor_set_from_spec",
            build_descriptor_set,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "generate_descriptor_set",
            generate_descriptor_set,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "generate_descriptor_set_shared_source",
            generate_descriptor_set_shared_source,
        ),
    ):
        tmp_path = Path(temporary_directory)
        xml_path = tmp_path / "amdgpu_isa_rdna4.xml"
        assert (
            amdgpu_descriptors.main(
                [
                    "--target=rdna4",
                    f"--xml={xml_path}",
                    f"--header={tmp_path / 'rdna4_descriptors.h'}",
                    f"--source={tmp_path / 'rdna4_descriptors.c'}",
                    f"--view-header=rdna4_gfx125x={tmp_path / 'rdna4_gfx125x_descriptors.h'}",
                ]
            )
            == 0
        )

    assert parse_calls == [xml_path]
    assert build_calls == [
        ("rdna4", parsed_spec),
        ("rdna4_gfx125x", parsed_spec),
    ]
