# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
content = args.input.read_bytes()
if not args.output.exists() or args.output.read_bytes() != content:
    args.output.write_bytes(content)
