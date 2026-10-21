# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds a selected model's programs and native serving deployment.

The build system supplies every input and the configured public linker. This
action performs no acquisition. Runtime assets are declared runfiles: relative
references preserve their package names without copying checkpoint payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from loom.model import ManifestError, ModelCard


def build(plan: dict, linker: str) -> None:
    card = ModelCard.parse(Path(plan["card"]).read_text())
    selection = card.select(plan["configuration"], programs=plan.get("programs"))
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    runtime_output = Path(plan["runtime_output"])
    context = Path("file") / selection.entrypoint.get("directory", ".")
    (output / context).mkdir(parents=True, exist_ok=True)
    runtime_base = runtime_output / context
    bound_files = plan["files"]
    runtime_assets = set(plan["assets"])
    used_assets = set()
    for name in selection.files:
        if name not in bound_files:
            raise ManifestError(f"file {name!r} requires a declared build input")

    def asset_path(binding):
        names = [binding["file"]] if "file" in binding else binding["files"]
        for name in names:
            if name not in runtime_assets:
                raise ManifestError(f"file {name!r} requires a runtime asset binding")
            used_assets.add(name)
        if "file" in binding:
            path = Path(bound_files[binding["file"]]["runtime_path"])
        elif not binding["files"]:
            (output / "empty").mkdir(exist_ok=True)
            return os.path.relpath(runtime_output / "empty", runtime_base)
        else:
            # A directory binding is an authored layout contract. Declared data
            # must provide that layout; linking never copies checkpoint files.
            roots = set()
            for name in binding["files"]:
                suffix = Path(name).relative_to(binding["directory"])
                path = Path(bound_files[name]["runtime_path"])
                if path.parts[-len(suffix.parts) :] != suffix.parts:
                    raise ManifestError(
                        f"file {name!r} does not preserve its asset layout"
                    )
                roots.add(Path(*path.parts[: -len(suffix.parts)]))
            if len(roots) != 1:
                raise ManifestError(
                    "directory binding requires one declared asset root"
                )
            path = roots.pop()
        return os.path.relpath(path, runtime_base)

    copied = set()

    def copy_program_file(name):
        destination = output / "file" / name
        if name not in copied:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bound_files[name]["path"], destination)
            copied.add(name)
        return destination

    def relative_program_path(path):
        return os.path.relpath(path, output / context)

    admitted = {}

    def link(arguments, destination):
        subprocess.run(
            [
                linker,
                *arguments,
                "--to=" + plan["encoding"],
                "--output=" + str(destination),
            ],
            check=True,
        )

    def compile_entry(entry, assets, directory):
        if "directory" in entry and Path("file") / entry["directory"] != context:
            raise ManifestError(
                "native components share the enclosing program directory"
            )
        directory.mkdir(parents=True)
        program = entry["programs"][selection.programs]
        inputs = []
        libraries = []
        for role, paths in (("inputs", inputs), ("libraries", libraries)):
            for provider_value in program[role]:
                provider = (
                    {"file": provider_value}
                    if isinstance(provider_value, str)
                    else provider_value
                )
                name = provider["file"]
                source = copy_program_file(name)
                for auxiliary in provider.get("data", []):
                    copy_program_file(auxiliary)
                previous = admitted.get(name)
                if previous is not None and previous != provider:
                    raise ManifestError(f"conflicting provider options for {name!r}")
                if previous is None and (
                    "format" in provider or provider.get("options")
                ):
                    normalized = directory / f"provider-{len(admitted)}.loombc"
                    options = ["--mode=merge", str(source)]
                    if "format" in provider:
                        options.append("--input-format=" + provider["format"])
                    options.extend(
                        "--input-options=" + option
                        for option in provider.get("options", [])
                    )
                    subprocess.run(
                        [linker, *options, "--to=bc", "--output=" + str(normalized)],
                        check=True,
                    )
                    normalized.replace(source)
                admitted[name] = provider
                paths.append(str(source))

        linked = {}
        for index, (role, roots) in enumerate(entry["roots"].items()):
            destination = directory / f"root-{index}.loom"
            link(
                [
                    "--mode=link",
                    *inputs,
                    *("--library=" + path for path in libraries),
                    *("--root=" + root for root in roots),
                ],
                destination,
            )
            linked[role] = relative_program_path(destination)
        if set(linked) != {"prepare", "control"}:
            raise ManifestError(
                "the native model contract requires prepare and control roles"
            )

        # Dynamic stage and parameter-policy locators retain their authored
        # namespace, including program data outside the linker provider list.
        definition = {
            "kind": entry["kind"],
            "prepare": linked["prepare"],
            "control": linked["control"],
            "sources": [relative_program_path(path) for path in [*inputs, *libraries]],
            "options": assets.get("options", {}),
        }
        for name, binding in assets["bindings"].items():
            if name in definition:
                raise ManifestError(f"asset binding replaces program field {name!r}")
            definition[name] = asset_path(binding)
        for index, (name, component_assets) in enumerate(
            assets.get("components", {}).items()
        ):
            if name in definition:
                raise ManifestError(f"component replaces model field {name!r}")
            component = compile_entry(
                entry["components"][name],
                component_assets,
                directory / f"component-{index}",
            )
            definition["sources"] = list(
                dict.fromkeys([*definition["sources"], *component.pop("sources")])
            )
            component.pop("kind")
            definition[name] = component
        return definition

    definition = compile_entry(
        selection.entrypoint, selection.assets, output / "program"
    )
    for group in ("execution", "request"):
        for key, value in selection.settings.get(group, {}).items():
            if key in definition:
                raise ManifestError(f"configuration cannot replace model field {key!r}")
            definition[key] = value
    if "options" in selection.settings:
        definition["options"] = {
            **definition["options"],
            **selection.settings["options"],
        }
    deployment = {
        "name": plan["name"],
        "base_directory": str(context),
        "definition": definition,
    }
    if runtime_assets != used_assets:
        raise ManifestError("runtime assets must match the selected binding closure")
    (output / "serve.json").write_text(json.dumps(deployment, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--linker", required=True)
    args = parser.parse_args()
    build(json.loads(args.plan.read_text()), args.linker)


if __name__ == "__main__":
    main()
