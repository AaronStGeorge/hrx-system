#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bundles JavaScript companions for a wasm binary."""

import argparse
import json
import os
import re


WASM_MAGIC = b"\x00asm"
WASM_VERSION = b"\x01\x00\x00\x00"


def decode_unsigned_leb128(data, offset):
    """Decodes an unsigned LEB128 value from `data` at `offset`."""
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return result, offset
        shift += 7
    raise ValueError("truncated unsigned LEB128 value")


def _skip_limits(data, offset):
    flags = data[offset]
    offset += 1
    _, offset = decode_unsigned_leb128(data, offset)
    if flags & 0x01:
        _, offset = decode_unsigned_leb128(data, offset)
    return offset


def _skip_import_descriptor(data, offset):
    kind = data[offset]
    offset += 1
    if kind == 0x00:
        _, offset = decode_unsigned_leb128(data, offset)
    elif kind == 0x01:
        offset += 1
        offset = _skip_limits(data, offset)
    elif kind == 0x02:
        offset = _skip_limits(data, offset)
    elif kind == 0x03:
        offset += 2
    else:
        raise ValueError("unknown wasm import kind 0x%02x" % kind)
    return offset


def parse_wasm_import_modules(wasm_path):
    """Returns the set of import module names used by a wasm binary."""
    with open(wasm_path, "rb") as file:
        data = file.read()
    if len(data) < 8 or data[0:4] != WASM_MAGIC:
        raise ValueError("%s is not a wasm binary" % wasm_path)
    if data[4:8] != WASM_VERSION:
        raise ValueError("%s uses an unsupported wasm version" % wasm_path)

    offset = 8
    while offset < len(data):
        section_id = data[offset]
        offset += 1
        section_size, offset = decode_unsigned_leb128(data, offset)
        section_end = offset + section_size
        if section_end > len(data):
            raise ValueError("wasm section extends past end of file")
        if section_id != 2:
            offset = section_end
            continue

        module_names = set()
        import_count, offset = decode_unsigned_leb128(data, offset)
        for _ in range(import_count):
            module_name_length, offset = decode_unsigned_leb128(data, offset)
            module_name = data[offset : offset + module_name_length].decode("utf-8")
            offset += module_name_length

            field_name_length, offset = decode_unsigned_leb128(data, offset)
            offset += field_name_length
            offset = _skip_import_descriptor(data, offset)
            module_names.add(module_name)
        return module_names
    return set()


def _strip_export(line):
    stripped = line.lstrip()
    if stripped.startswith("export function "):
        return line.replace("export function ", "function ", 1)
    if stripped.startswith("export default function "):
        return line.replace("export default function ", "function ", 1)
    if stripped.startswith("export const "):
        return line.replace("export const ", "const ", 1)
    if stripped.startswith("export class "):
        return line.replace("export class ", "class ", 1)
    if stripped.startswith("export {"):
        return None
    return line


def transform_companion(source, module_name, source_path):
    """Transforms a companion module into an inlineable JavaScript fragment."""
    output_lines = []
    found_create_imports = False
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import "):
            raise ValueError(
                "%s imports another module; wasm companions must be self-contained"
                % source_path
            )
        if stripped.startswith("export function createImports"):
            found_create_imports = True
        stripped_line = _strip_export(line)
        if stripped_line is not None:
            output_lines.append(stripped_line)
    if not found_create_imports:
        raise ValueError("%s must export createImports(context)" % source_path)

    identifier = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
    return (
        "// --- Wasm import module: %s (%s) ---\n"
        "const _iree_wasm_module_%s = (() => {\n"
        "%s\n"
        "return createImports;\n"
        "})();\n"
    ) % (module_name, source_path, identifier, "\n".join(output_lines))


def generate_import_merger(module_names):
    entries = []
    for module_name in module_names:
        identifier = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
        entries.append(
            '    "%s": _iree_wasm_module_%s(context)' % (module_name, identifier)
        )
    return (
        "\n"
        "export function createWasmImports(context) {\n"
        "  return {\n"
        "%s\n"
        "  };\n"
        "}\n"
    ) % ",\n".join(entries)


def _relative_import_path(line):
    stripped = line.lstrip()
    if not stripped.startswith("import ") or " from " not in stripped:
        return None
    match = re.search(r"""from\s+['"](\.[^'"]+)['"]""", stripped)
    return match.group(1) if match else None


def _strip_exports(source):
    output_lines = []
    for line in source.splitlines():
        stripped_line = _strip_export(line)
        if stripped_line is not None:
            output_lines.append(stripped_line)
    return "\n".join(output_lines)


def _resolve_local_imports_recursive(source, source_dir, inlined):
    dependency_parts = []
    cleaned_lines = []
    for line in source.splitlines():
        relative_path = _relative_import_path(line)
        if relative_path is None:
            cleaned_lines.append(line)
            continue

        absolute_path = os.path.normpath(os.path.join(source_dir, relative_path))
        if absolute_path in inlined:
            continue
        inlined.add(absolute_path)
        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(
                "local import %r resolved to missing file %s"
                % (relative_path, absolute_path)
            )
        with open(absolute_path, encoding="utf-8") as file:
            dependency_source = file.read()
        nested_source, cleaned_dependency = _resolve_local_imports_recursive(
            dependency_source,
            os.path.dirname(absolute_path),
            inlined,
        )
        dependency_parts.append(nested_source)
        dependency_parts.append(
            "// --- Inlined entry dependency: %s ---\n%s\n"
            % (relative_path, _strip_exports(cleaned_dependency))
        )
    return "".join(dependency_parts), "\n".join(cleaned_lines)


def resolve_local_imports(source, source_path):
    """Inlines relative ESM imports while preserving external imports."""
    return _resolve_local_imports_recursive(
        source,
        os.path.dirname(os.path.abspath(source_path)),
        set(),
    )


def _read_manifest(manifest_path):
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    grouped_modules = {}
    module_order = []
    for entry in manifest:
        module_name = entry["module"]
        if module_name not in grouped_modules:
            grouped_modules[module_name] = []
            module_order.append(module_name)
        grouped_modules[module_name].append(entry["path"])
    return module_order, grouped_modules


def bundle_wasm(wasm_path, main_path, modules_path, output_path, wasm_filename=None):
    """Writes a self-contained JavaScript bundle next to `wasm_path`."""
    module_order, grouped_modules = _read_manifest(modules_path)
    imported_modules = parse_wasm_import_modules(wasm_path)
    active_modules = [
        module_name for module_name in module_order if module_name in imported_modules
    ]
    wasm_filename = wasm_filename or os.path.basename(wasm_path)

    parts = [
        "// Generated by IREE wasm binary bundler.\n"
        "//\n"
        "// Wasm binary: %s\n"
        "const __IREE_WASM_BINARY = '%s';\n"
        % (wasm_filename, wasm_filename)
    ]
    for module_name in active_modules:
        for source_path in grouped_modules[module_name]:
            with open(source_path, encoding="utf-8") as file:
                parts.append(transform_companion(file.read(), module_name, source_path))
    parts.append(generate_import_merger(active_modules))

    with open(main_path, encoding="utf-8") as file:
        main_source = file.read()
    inlined_dependencies, cleaned_main = resolve_local_imports(main_source, main_path)
    if inlined_dependencies:
        parts.append("\n// --- Entry dependencies ---\n%s" % inlined_dependencies)
    parts.append("\n// --- Entry point: %s ---\n%s" % (main_path, cleaned_main))

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(parts))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm", required=True)
    parser.add_argument("--wasm-filename")
    parser.add_argument("--main", required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    bundle_wasm(
        wasm_path=args.wasm,
        main_path=args.main,
        modules_path=args.modules,
        output_path=args.output,
        wasm_filename=args.wasm_filename,
    )


if __name__ == "__main__":
    main()
