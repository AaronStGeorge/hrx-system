# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packages the maintained example for download without repository build files."""

import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).absolute().parent / "model-publication"
    files = [
        root / name
        for name in (
            ".gitignore",
            "AGENTS.md",
            "LICENSE",
            "README.md",
            "catalog.json",
            "refresh.py",
            "smoke.py",
        )
    ]
    for pattern in (
        "model/*/model.json",
        "model/*/model.json.lock",
        "model/*/tokenizer.json",
        "model/*/program/*.loom",
    ):
        files.extend(root.glob(pattern))
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = "model-publication/" + path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


if __name__ == "__main__":
    main()
