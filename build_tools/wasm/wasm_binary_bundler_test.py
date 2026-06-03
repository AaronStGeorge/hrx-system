#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for wasm_binary_bundler."""

import json
import os
import tempfile
import unittest

import wasm_binary_bundler


def _leb(value):
    result = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            return bytes(result)


def _wasm_with_import_modules(module_names):
    payload = _leb(len(module_names))
    for module_name in module_names:
        module_name = module_name.encode("utf-8")
        field_name = b"f"
        payload += _leb(len(module_name)) + module_name
        payload += _leb(len(field_name)) + field_name
        payload += b"\x00" + _leb(0)
    return (
        wasm_binary_bundler.WASM_MAGIC
        + wasm_binary_bundler.WASM_VERSION
        + b"\x02"
        + _leb(len(payload))
        + payload
    )


class WasmBinaryBundlerTest(unittest.TestCase):
    def test_parse_wasm_import_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            wasm_path = os.path.join(directory, "module.wasm")
            with open(wasm_path, "wb") as file:
                file.write(_wasm_with_import_modules(["iree_a", "iree_b"]))

            self.assertEqual(
                {"iree_a", "iree_b"},
                wasm_binary_bundler.parse_wasm_import_modules(wasm_path),
            )

    def test_bundle_inlines_active_companions_and_entry_deps(self):
        with tempfile.TemporaryDirectory() as directory:
            wasm_path = os.path.join(directory, "program.wasm")
            main_path = os.path.join(directory, "main.mjs")
            dep_path = os.path.join(directory, "dep.mjs")
            companion_path = os.path.join(directory, "companion.mjs")
            unused_companion_path = os.path.join(directory, "unused.mjs")
            modules_path = os.path.join(directory, "modules.json")
            output_path = os.path.join(directory, "program.mjs")

            with open(wasm_path, "wb") as file:
                file.write(_wasm_with_import_modules(["iree_active"]))
            with open(main_path, "w", encoding="utf-8") as file:
                file.write(
                    "import {readFileSync} from 'node:fs';\n"
                    "import {helper} from './dep.mjs';\n"
                    "console.log(helper(), readFileSync);\n"
                )
            with open(dep_path, "w", encoding="utf-8") as file:
                file.write("export function helper() { return 'ok'; }\n")
            with open(companion_path, "w", encoding="utf-8") as file:
                file.write(
                    "export function createImports(context) {\n"
                    "  return {run() { return context.value; }};\n"
                    "}\n"
                )
            with open(unused_companion_path, "w", encoding="utf-8") as file:
                file.write(
                    "export function createImports(context) {\n"
                    "  return {unused() { return context.value; }};\n"
                    "}\n"
                )
            with open(modules_path, "w", encoding="utf-8") as file:
                json.dump(
                    [
                        {"module": "iree_active", "path": companion_path},
                        {"module": "iree_unused", "path": unused_companion_path},
                    ],
                    file,
                )

            wasm_binary_bundler.bundle_wasm(
                wasm_path=wasm_path,
                main_path=main_path,
                modules_path=modules_path,
                output_path=output_path,
                wasm_filename="program.wasm",
            )

            with open(output_path, encoding="utf-8") as file:
                output = file.read()
            self.assertIn("const __IREE_WASM_BINARY = 'program.wasm';", output)
            self.assertIn("_iree_wasm_module_iree_active", output)
            self.assertIn("function helper() { return 'ok'; }", output)
            self.assertIn("import {readFileSync} from 'node:fs';", output)
            self.assertNotIn("_iree_wasm_module_iree_unused", output)
            self.assertNotIn("import {helper} from './dep.mjs';", output)

    def test_companion_without_create_imports_fails(self):
        with self.assertRaisesRegex(ValueError, "createImports"):
            wasm_binary_bundler.transform_companion(
                "export function notImports() {}\n",
                "iree_bad",
                "bad.mjs",
            )

    def test_companion_import_fails(self):
        with self.assertRaisesRegex(ValueError, "self-contained"):
            wasm_binary_bundler.transform_companion(
                "import {x} from './x.mjs';\n"
                "export function createImports(context) { return {}; }\n",
                "iree_bad",
                "bad.mjs",
            )


if __name__ == "__main__":
    unittest.main()
