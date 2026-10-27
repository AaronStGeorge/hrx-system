# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import gzip_file


class GzipFileTest(unittest.TestCase):
    def check_roundtrip(self, compressed, original):
        # No optional fields, zero timestamp, maximum compression, neutral OS.
        self.assertEqual(compressed[:10], bytes.fromhex("1f8b08000000000002ff"))
        self.assertEqual(zlib.decompress(compressed, wbits=31), original)

    def test_binary_and_empty_files_ignore_names_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for original in (b"", bytes(range(256)) * 8):
                outputs = []
                for index in range(2):
                    source = root / f"input {index}.bin"
                    output = root / "nested directory" / f"output {index}.gz"
                    source.write_bytes(original)
                    os.utime(source, (index + 1, index + 1))
                    gzip_file.main(["--input", str(source), "--output", str(output)])
                    outputs.append(output.read_bytes())
                    self.check_roundtrip(outputs[-1], original)
                self.assertEqual(*outputs)

    def test_missing_input_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                gzip_file.main(
                    ["--input", str(root / "missing"), "--output", str(root / "output")]
                )
            self.assertFalse((root / "output").exists())

    def test_generated_assets_roundtrip(self):
        for source, compressed in self.assets:
            with self.subTest(asset=source.name):
                original = source.read_bytes()
                encoded = compressed.read_bytes()
                self.check_roundtrip(encoded, original)
                self.assertLess(len(encoded), len(original))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, nargs=2, action="append", required=True)
    args, remaining = parser.parse_known_args()
    GzipFileTest.assets = args.asset
    unittest.main(argv=[sys.argv[0], *remaining])
