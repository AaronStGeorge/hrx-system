# Lefthook Presubmit

The root Lefthook configuration owns hook dispatch. The Python presubmit
dispatcher owns file selection, profiles, hygiene checks, and project test
routing. Project-specific policy stays in each project under
`*/build_tools/presubmit.py`.

## Commit Contract

A successful Git commit must not leave the worktree dirty and must not commit
stale formatter or generated-file output. The Git `pre-commit` hook therefore
runs in non-mutating check mode:

```bash
python build_tools/lefthook/presubmit.py --check {staged_files}
```

Local fixups happen through explicit helper scripts:

```bash
build_tools/lefthook/precommit.sh
build_tools/lefthook/presubmit.sh
```

`precommit.sh` fixes and checks staged hygiene. `presubmit.sh` fixes staged
hygiene, then runs the paranoid staged-file profile, which includes affected
project tests.

## Profiles

`default` runs repository hygiene: buildifier, Ruff, clang-format,
bazel-to-cmake, generated AMDGPU target metadata, watchwords, merge-conflict
markers, and basic text hygiene.

`paranoid` adds affected project Bazel presubmit tests and the static-analysis
lane. The static-analysis lane is intentionally empty until providers such as
clang-tidy or CodeQL are wired in.

`ci` is the full-tree, non-mutating profile. It checks all tracked files and
runs all configured project presubmit tests.

## Lefthook Groups

`pre-commit` is the installed Git hook and is check-only.

`fix` runs the staged hygiene fixer for manual use through Lefthook:

```bash
lefthook run fix
```

`presubmit` runs the paranoid profile for manual use through Lefthook:

```bash
lefthook run presubmit
```

`check` runs the CI profile:

```bash
lefthook run check
```

`lefthook.yml` requires Lefthook 2.1.9 or newer because the repository uses
custom manual hook groups in addition to Git hook names.

## Tool Installation

Python-packaged local tools are pinned in `requirements-dev.lock.txt`. That
lock contains only local development tools; Bazel build and test dependencies
belong in Bazel module fragments and `MODULE.bazel.lock`.

Standalone binaries are installed by `build_tools/devtools/install.py` into
`.venv/bin` when a virtual environment exists. The default manifest installs
Bazelisk and buildifier with pinned URLs and SHA-256 hashes:

```bash
.venv/bin/python build_tools/devtools/install.py
```

Optional future providers can be added to the installer manifest without being
part of the default install set. Presubmit providers that are optional should
skip when their tool is unavailable and print the missing executable.

## Project Dispatch

The root dispatcher writes the selected file list once and invokes each affected
project script with `--files-from`. Project scripts decide whether those files
affect their project. Shared build-system paths fan out to every configured
project entry point.

This keeps root policy focused on repository-wide hygiene while preserving
project ownership of tests, expensive checks, and future project-specific static
analysis.
