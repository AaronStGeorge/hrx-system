# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Production Loom IR verification for importer check output."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loom.importers.check.cases import CheckCase
from loom.importers.check.results import CheckResult

_LOOM_OPT_RUNFILES_SUFFIX = Path("loom/src/loom/tools/loom-opt/loom-opt")


@dataclass(frozen=True, slots=True)
class LoomOutputVerifier:
    """Runs printed Loom IR through the production verifier."""

    loom_opt: Path

    @classmethod
    def resolve(cls, explicit_path: Path | None) -> LoomOutputVerifier | None:
        """Resolves the verifier tool for importer check execution."""

        if explicit_path is not None:
            return cls(explicit_path)
        resolved_path = _find_default_loom_opt()
        if resolved_path is None:
            return None
        return cls(resolved_path)

    def verify(self, case: CheckCase, loom_ir: str) -> CheckResult | None:
        """Returns a failed check result if `loom_ir` fails C verification."""

        try:
            process = subprocess.run(
                [str(self.loom_opt)],
                input=loom_ir,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return CheckResult(
                path=case.path,
                case_index=case.index,
                returncode=1,
                stdout=loom_ir,
                stderr=f"{type(exc).__name__}: {exc}\n",
                input=case.input,
                expected=case.expected,
                mismatch="generated Loom IR could not be verified",
            )
        if process.returncode == 0:
            return None
        diagnostic_text = process.stderr or process.stdout
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=process.returncode,
            stdout=loom_ir,
            stderr=diagnostic_text,
            input=case.input,
            expected=case.expected,
            mismatch="generated Loom IR failed verification",
        )


def _find_default_loom_opt() -> Path | None:
    for candidate in _runfiles_loom_opt_candidates():
        if candidate.is_file():
            return candidate
    executable = shutil.which("loom-opt")
    if executable is not None:
        return Path(executable)
    return None


def _runfiles_loom_opt_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        root = Path(runfiles_dir)
        candidates.extend(
            (
                root / "_main" / _LOOM_OPT_RUNFILES_SUFFIX,
                root / "iree" / _LOOM_OPT_RUNFILES_SUFFIX,
                root / _LOOM_OPT_RUNFILES_SUFFIX,
            )
        )
    manifest_path = os.environ.get("RUNFILES_MANIFEST_FILE")
    if manifest_path:
        candidates.extend(_manifest_loom_opt_candidates(Path(manifest_path)))
    return tuple(candidates)


def _manifest_loom_opt_candidates(manifest_path: Path) -> tuple[Path, ...]:
    if not manifest_path.is_file():
        return ()
    suffix = _LOOM_OPT_RUNFILES_SUFFIX.as_posix()
    candidates: list[Path] = []
    try:
        with manifest_path.open() as file:
            for line in file:
                logical_path, separator, physical_path = line.rstrip("\n").partition(
                    " "
                )
                if not separator or not logical_path.endswith(suffix):
                    continue
                candidates.append(Path(physical_path))
    except OSError:
        return ()
    return tuple(candidates)
