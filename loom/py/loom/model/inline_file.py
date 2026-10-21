# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Materializes an inline model-card file as a declared build artifact."""

import argparse
import base64
from pathlib import Path

from loom.model import ManifestError, ModelCard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = ModelCard.parse(args.card.read_text()).document["files"]
    if args.file not in files or files[args.file]["origin"]["kind"] != "inline":
        raise ManifestError(f"{args.file!r} is not a declared inline file")
    args.output.write_bytes(base64.b64decode(files[args.file]["origin"]["base64"]))


if __name__ == "__main__":
    main()
