#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Refresh author-owned asset locks and catalog metadata with the installed SDK."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, help="reuse downloaded assets in this Loom cache"
    )
    args = parser.parse_args()
    root = Path(__file__).absolute().parent
    cards = sorted(root.glob("model/*/model.json"))
    if not cards:
        parser.error("no model/*/model.json files found beside this script")
    for card in cards:
        command = [
            sys.executable,
            "-m",
            "loom.model.fetch_main",
            str(card),
            "--update-lock",
            "--all-variants",
            "--no-input",
        ]
        if args.cache is not None:
            command.append("--cache=" + str(args.cache.absolute()))
        subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "loom.model.catalog",
            "--output=" + str(root / "catalog.json"),
            *[
                "--card=" + card.relative_to(root).as_posix() + "=" + str(card)
                for card in cards
            ],
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
