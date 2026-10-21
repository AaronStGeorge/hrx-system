# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Model-card parsing and selection, without compiler, network, or host state.

Files describe content and origins. Named program sets describe linker inputs;
their names carry no encoding semantics. Entrypoints own callable root bindings,
variants own compatible parameter inputs, and configurations select an instance.
The compiler and model implementation own symbol and model-option validation.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit


class ManifestError(ValueError):
    """An invalid external model description or unavailable selection."""


def _fail(path: str, message: str) -> None:
    raise ManifestError(f"{path}: {message}")


def _object(value: Any, path: str, required=(), optional=()) -> dict:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing:
        _fail(path, f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        _fail(path, f"unknown fields: {', '.join(sorted(unknown))}")
    return value


def _mapping(value: Any, path: str, *, empty: bool = False) -> dict:
    if not isinstance(value, dict) or (not value and not empty):
        _fail(path, "expected a named object" + ("" if empty else " with entries"))
    for key in value:
        _string(key, path)
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        _fail(path, "expected a nonempty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(path, "expected valid UTF-8")
    return value


def _path(value: Any, path: str, *, directory: bool = False) -> str:
    _string(value, path)
    parsed = PurePosixPath(value)
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or ".." in parsed.parts
        or str(parsed) != value
        or (value == "." and not directory)
    ):
        _fail(path, "expected a normalized package-relative path")
    return value


def _strings(value: Any, path: str, *, empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        _fail(path, "expected an array" + ("" if empty else " with entries"))
    for index, item in enumerate(value):
        _string(item, f"{path}[{index}]")
    if len(value) != len(set(value)):
        _fail(path, "duplicate entries")
    return value


def _reference(value: Any, table: dict, path: str) -> str:
    _string(value, path)
    if value not in table:
        _fail(path, f"unknown reference {value!r}")
    return value


def _uint(value: Any, path: str) -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        _fail(path, "expected an unsigned 64-bit integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("$", f"duplicate object key {key!r}")
        result[key] = value
    return result


def _provider(value: Any, files: dict, path: str) -> dict:
    provider = {"file": value} if isinstance(value, str) else value
    _object(provider, path, ("file",), ("format", "options", "data"))
    _reference(provider["file"], files, path + ".file")
    if "format" in provider:
        _string(provider["format"], path + ".format")
    if "options" in provider:
        _strings(provider["options"], path + ".options")
    for filename in _strings(provider.get("data", []), path + ".data"):
        _reference(filename, files, path + ".data")
    return provider


def _file(record: dict, path: str) -> None:
    _object(record, path, ("origin",), ("size", "sha256", "description"))
    origin = _mapping(record["origin"], path + ".origin")
    kind = _string(origin.get("kind"), path + ".origin.kind")
    fields = {
        "local": (("kind", "path"), ()),
        "url": (("kind", "url"), ()),
        "hf": (("kind", "repository", "revision", "path"), ("git_blob",)),
        "inline": (("kind", "base64"), ()),
        "build": (("kind", "target"), ()),
    }
    if kind not in fields:
        _fail(path + ".origin.kind", f"unsupported file origin {kind!r}")
    required, optional = fields[kind]
    _object(origin, path + ".origin", required, optional)
    for key, value in origin.items():
        if key != "base64":
            _string(value, path + ".origin." + key)
    if kind == "local":
        _path(origin["path"], path + ".origin.path")
    elif kind == "url":
        try:
            url = urlsplit(origin["url"])
        except ValueError:
            _fail(path + ".origin.url", "invalid HTTPS URL")
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            _fail(path + ".origin.url", "expected an HTTPS URL without credentials")
    elif kind == "hf":
        _path(origin["path"], path + ".origin.path")
        revision = origin["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            _fail(path + ".origin.revision", "expected an immutable commit identity")
    if "size" in record:
        _uint(record["size"], path + ".size")
    if "sha256" in record:
        digest = _string(record["sha256"], path + ".sha256")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            _fail(path + ".sha256", "expected a lowercase SHA-256 digest")
    if kind == "inline":
        try:
            data = base64.b64decode(origin["base64"], validate=True)
        except (ValueError, TypeError):
            _fail(path + ".origin.base64", "invalid base64 content")
        if record.get("size", len(data)) != len(data):
            _fail(path + ".size", "does not match inline content")
        digest = hashlib.sha256(data).hexdigest()
        if record.get("sha256", digest) != digest:
            _fail(path + ".sha256", "does not match inline content")


def _entrypoint(value: dict, files: dict, path: str) -> None:
    _object(
        value,
        path,
        ("kind", "roots", "programs"),
        (
            "interface",
            "constraints",
            "components",
            "description",
            "options",
            "directory",
        ),
    )
    _string(value["kind"], path + ".kind")
    _path(value.get("directory", "."), path + ".directory", directory=True)
    _strings(value.get("options", []), path + ".options")
    for role, roots in _mapping(value["roots"], path + ".roots").items():
        _strings(roots, path + ".roots." + role, empty=False)
    for name, program in _mapping(value["programs"], path + ".programs").items():
        current = path + ".programs." + name
        _object(program, current, ("inputs", "libraries"))
        seen = set()
        for role in ("inputs", "libraries"):
            if not isinstance(program[role], list):
                _fail(current + "." + role, "expected an array")
            for index, item in enumerate(program[role]):
                provider = _provider(item, files, f"{current}.{role}[{index}]")
                if provider["file"] in seen:
                    _fail(current, f"duplicate provider {provider['file']!r}")
                seen.add(provider["file"])
        if not seen:
            _fail(current, "requires at least one provider")
    for field in ("interface", "constraints"):
        if field in value:
            _mapping(value[field], path + "." + field, empty=True)
    for name, component in _mapping(
        value.get("components", {}), path + ".components", empty=True
    ).items():
        _entrypoint(component, files, path + ".components." + name)


def _assets(value: dict, entrypoint: dict, files: dict, path: str) -> None:
    _object(value, path, ("programs", "bindings"), ("options", "components"))
    for name in _strings(value["programs"], path + ".programs", empty=False):
        _reference(name, entrypoint["programs"], path + ".programs")
    _mapping(value.get("options", {}), path + ".options", empty=True)
    for name, binding in _mapping(
        value["bindings"], path + ".bindings", empty=True
    ).items():
        current = path + ".bindings." + name
        _mapping(binding, current)
        if "file" in binding:
            _object(binding, current, ("file",))
            _reference(binding["file"], files, current + ".file")
        else:
            _object(binding, current, ("directory", "files"))
            directory = _path(
                binding["directory"], current + ".directory", directory=True
            )
            for filename in _strings(binding["files"], current + ".files"):
                _reference(filename, files, current + ".files")
                if directory != "." and not filename.startswith(directory + "/"):
                    _fail(current, f"file {filename!r} is outside the directory")
    for name, component in _mapping(
        value.get("components", {}), path + ".components", empty=True
    ).items():
        _reference(name, entrypoint.get("components", {}), path + ".components")
        _assets(
            component,
            entrypoint["components"][name],
            files,
            path + ".components." + name,
        )


@dataclass(frozen=True)
class Selection:
    """Resolved declaration; contained values are independent owned snapshots."""

    # Published logical model identity.
    model_id: str
    # Selected canonical configuration name.
    configuration: str
    # Selected runnable entrypoint name.
    entrypoint_name: str
    # Selected parameter-content variant name.
    variant: str
    # Selected provider-set name, without format semantics.
    programs: str
    # Callable contract and indexed program declarations.
    entrypoint: dict
    # Compatible asset bindings and model-owned source options.
    assets: dict
    # Canonical instance/workload settings.
    settings: dict
    # Selected files in deterministic dependency order.
    files: dict


class ModelCard:
    """An owned, structurally validated model publication."""

    def __init__(self, document: dict):
        self._document = copy.deepcopy(document)
        self._validate()

    @classmethod
    def parse(cls, source: str) -> ModelCard:
        def reject_constant(value):
            _fail("$", f"non-JSON numeric constant {value}")

        try:
            value = json.loads(
                source, object_pairs_hook=_unique_object, parse_constant=reject_constant
            )
        except json.JSONDecodeError as error:
            raise ManifestError(f"$: {error}") from error
        return cls(value)

    @property
    def document(self) -> dict:
        return copy.deepcopy(self._document)

    def _validate(self) -> None:
        doc = _object(
            self._document,
            "$",
            (
                "schema_version",
                "model",
                "files",
                "entrypoints",
                "variants",
                "configurations",
            ),
            ("qualification",),
        )
        if type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
            _fail("$.schema_version", "unsupported model-card version")
        model = _object(
            doc["model"],
            "$.model",
            ("id", "name"),
            ("description", "origin", "license", "implementation"),
        )
        _string(model["id"], "$.model.id")
        _string(model["name"], "$.model.name")
        files = _mapping(doc["files"], "$.files", empty=True)
        for name, record in files.items():
            _path(name, "$.files")
            _file(record, "$.files." + name)
        entries = _mapping(doc["entrypoints"], "$.entrypoints")
        for name, entry in entries.items():
            _entrypoint(entry, files, "$.entrypoints." + name)
        for name, variant in _mapping(doc["variants"], "$.variants").items():
            current = "$.variants." + name
            _object(variant, current, ("entrypoints",), ("description",))
            for entry_name, assets in _mapping(
                variant["entrypoints"], current + ".entrypoints"
            ).items():
                _reference(entry_name, entries, current + ".entrypoints")
                _assets(
                    assets,
                    entries[entry_name],
                    files,
                    current + ".entrypoints." + entry_name,
                )
        for name, configuration in _mapping(
            doc["configurations"], "$.configurations"
        ).items():
            current = "$.configurations." + name
            _object(
                configuration,
                current,
                ("entrypoint", "variant", "programs"),
                (
                    "execution",
                    "request",
                    "options",
                    "workload",
                    "requires",
                    "description",
                ),
            )
            for field in ("execution", "request", "options", "workload", "requires"):
                _mapping(
                    configuration.get(field, {}), current + "." + field, empty=True
                )
            self.select(name)
        for name, row in _mapping(
            doc.get("qualification", {}), "$.qualification", empty=True
        ).items():
            current = "$.qualification." + name
            _object(
                row,
                current,
                ("configuration",),
                (
                    "variant",
                    "programs",
                    "requires",
                    "checks",
                    "benchmarks",
                    "description",
                ),
            )
            self.select(
                row["configuration"],
                variant=row.get("variant"),
                programs=row.get("programs"),
            )

    def select(
        self,
        configuration: str,
        *,
        variant: str | None = None,
        programs: str | None = None,
    ) -> Selection:
        doc = self._document
        _reference(configuration, doc["configurations"], "configuration")
        settings = doc["configurations"][configuration]
        name = _reference(settings["entrypoint"], doc["entrypoints"], "entrypoint")
        unknown_options = settings.get("options", {}).keys() - set(
            doc["entrypoints"][name].get("options", [])
        )
        if unknown_options:
            _fail(
                "options",
                "undeclared model options: " + ", ".join(sorted(unknown_options)),
            )
        selected_variant = settings["variant"] if variant is None else variant
        _reference(selected_variant, doc["variants"], "variant")
        variant_entries = doc["variants"][selected_variant]["entrypoints"]
        _reference(name, variant_entries, "variant.entrypoints")
        assets = variant_entries[name]
        selected_programs = settings["programs"] if programs is None else programs
        _reference(selected_programs, doc["entrypoints"][name]["programs"], "programs")
        if selected_programs not in assets["programs"]:
            _fail("programs", "not supported by the selected asset variant")
        selected_files = {}

        def add_file(filename):
            selected_files[filename] = doc["files"][filename]

        def collect(entry, inputs, program_name):
            program = entry["programs"][program_name]
            for role in ("inputs", "libraries"):
                for value in program[role]:
                    provider = {"file": value} if isinstance(value, str) else value
                    add_file(provider["file"])
                    for filename in provider.get("data", []):
                        add_file(filename)
            for binding in inputs["bindings"].values():
                for filename in (
                    [binding["file"]] if "file" in binding else binding["files"]
                ):
                    add_file(filename)
            for component_name, component_inputs in inputs.get(
                "components", {}
            ).items():
                component = entry["components"][component_name]
                if program_name not in component_inputs["programs"]:
                    _fail(
                        "programs",
                        f"component {component_name!r} does not support selection",
                    )
                collect(component, component_inputs, program_name)

        entrypoint = doc["entrypoints"][name]
        collect(entrypoint, assets, selected_programs)
        return Selection(
            doc["model"]["id"],
            configuration,
            name,
            selected_variant,
            selected_programs,
            copy.deepcopy(entrypoint),
            copy.deepcopy(assets),
            copy.deepcopy(settings),
            copy.deepcopy(selected_files),
        )
