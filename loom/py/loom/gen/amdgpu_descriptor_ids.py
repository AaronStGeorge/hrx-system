# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU target-low family ID constants."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[2]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    amdgpu_common_reg_class_ids,
    amdgpu_descriptor_id_keys,
    amdgpu_immediate_encoding_id_items,
)
from loom.target.low_descriptors import descriptor_stable_id  # noqa: E402


def _clang_format_source(source: str, assume_filename: Path) -> str:
    result = subprocess.run(
        ["clang-format", f"--assume-filename={assume_filename}"],
        input=source,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def _amdgpu_suffix(value: str) -> str:
    if not value.startswith("amdgpu."):
        raise ValueError(f"expected AMDGPU key, got '{value}'")
    return value.removeprefix("amdgpu.")


def _descriptor_id_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_ID_{_c_identifier(_amdgpu_suffix(key))}"


def _reg_class_id_constant_name(reg_class_name: str) -> str:
    return f"LOOM_AMDGPU_REG_CLASS_ID_{_c_identifier(_amdgpu_suffix(reg_class_name))}"


def _immediate_encoding_id_constant_name(name: str) -> str:
    return f"LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_{_c_identifier(name)}"


def _u64_literal(value: int) -> str:
    return f"UINT64_C(0x{value:016x})"


def _emit_header(*, header_path: Path, format_output: bool) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.amdgpu_descriptor_ids"),
        "",
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_IDS_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_IDS_H_",
        "",
        "#include <stdint.h>",
        "",
    ]
    lines.extend(f"#define {_descriptor_id_constant_name(key)} {_u64_literal(descriptor_stable_id(key))}" for key in amdgpu_descriptor_id_keys())
    lines.append("")
    lines.extend(f"#define {_reg_class_id_constant_name(reg_class_name)} {reg_class_id}u" for reg_class_name, reg_class_id in amdgpu_common_reg_class_ids())
    lines.append("")
    lines.extend(f"#define {_immediate_encoding_id_constant_name(name)} {encoding_id}u" for name, encoding_id in amdgpu_immediate_encoding_id_items())
    lines.extend(["", "#endif  // LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_IDS_H_"])
    source = "\n".join(lines) + "\n"
    if not format_output:
        return source
    return _clang_format_source(source, header_path)


def _emit_source(*, public_header: str, source_path: Path, format_output: bool) -> str:
    source = "\n".join(
        [
            "// Copyright 2026 The IREE Authors",
            "//",
            "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
            "// See https://llvm.org/LICENSE.txt for license information.",
            "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
            "",
            *line_comment_header("//", generator="loom.gen.amdgpu_descriptor_ids"),
            "",
            f'#include "{public_header}"',
            "",
        ]
    )
    if not format_output:
        return source
    return _clang_format_source(source, source_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU target-low family ID constants.")
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated descriptor ID header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated descriptor ID source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/descriptor_ids.h",
        help="Public include path for the generated header.",
    )
    args = parser.parse_args(argv)

    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(
        _emit_header(header_path=args.header, format_output=True),
        encoding="utf-8",
    )
    args.source.write_text(
        _emit_source(
            public_header=args.public_header,
            source_path=args.source,
            format_output=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
