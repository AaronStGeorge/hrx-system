# Lefthook Presubmit

The root Lefthook configuration owns Git hook dispatch. `dev.py` selects the
local build lane and installs the hook through Lefthook. The Python presubmit
dispatcher owns file selection, profiles, hygiene checks, and project test
routing. Project-specific policy stays in each project under
`*/build_tools/presubmit.py`.

## Commit Contract

A successful Git commit must not leave the worktree dirty and must not commit
stale formatter or generated-file output. The Git `pre-commit` hook therefore
runs in non-mutating check mode:

```bash
python dev.py <lane> precommit --profile <profile> {staged_files}
```

Local fixups happen through explicit commands:

```bash
python dev.py bazel fix
python dev.py cmake fix
```

`fix` applies staged hygiene repairs and stages files owned by those fixers.
`precommit` checks the current local change set. `presubmit` is non-mutating and
runs the full-tree CI-shaped profile.

Normal presubmit output is terse: major phases, named checks, pass/fail status,
and failure details. Use `--verbose` when debugging the dispatcher itself or
when exact command lines and streaming tool output are useful.

Legacy shell helpers remain for direct debugging:

```bash
build_tools/lefthook/precommit.sh
build_tools/lefthook/presubmit.sh
```

## Profiles

`default` runs repository hygiene: buildifier, Ruff, clang-format,
bazel-to-cmake, generated AMDGPU target metadata, watchwords, merge-conflict
markers, and basic text hygiene.

`paranoid` adds affected project Bazel presubmit tests and the static-analysis
lane. The static-analysis lane is intentionally empty until providers such as
clang-tidy or CodeQL are wired in.

`ci` is the full-tree, non-mutating profile. It checks all tracked files and
runs all configured project presubmit tests.

The default manual `precommit` and hook-install profiles are:

| Lane | Default Profile | Scope |
|------|-----------------|-------|
| Bazel | `paranoid` | Hygiene, affected project Bazel tests, and configured static-analysis providers. |
| CMake | `default` | Shared repository hygiene. |

`presubmit` defaults to `ci` for both lanes:

```bash
python dev.py bazel presubmit
python dev.py cmake presubmit
```

Any manual run can select a profile explicitly:

```bash
python dev.py bazel precommit --profile default
python dev.py bazel precommit --profile paranoid
python dev.py bazel precommit --profile ci
python dev.py cmake presubmit --profile default
```

## Lefthook Groups

`pre-commit` is the installed Git hook and is check-only.

`dev.py` installs lane-specific local hook policy into ignored
`lefthook-local.yml`:

```bash
python dev.py bazel hook --profile paranoid
python dev.py cmake hook --profile default
```

Re-run `python dev.py <lane> hook --profile <profile>` to change the default
profile used by Git commits. The generated file records the selected lane and
profile so local hook behavior can be audited without reading Python.

`fix` runs the staged hygiene fixer for manual use through Lefthook:

```bash
lefthook run fix
```

`precommit` runs the paranoid profile over staged, unstaged, and untracked
local changes:

```bash
lefthook run precommit
```

`presubmit` runs the CI profile:

```bash
lefthook run presubmit
```

`check` is kept as a CI-profile alias:

```bash
lefthook run check
```

`lefthook.yml` requires Lefthook 2.1.9 or newer because the repository uses
custom manual hook groups in addition to Git hook names.

## Tool Installation

Python-packaged local tools are pinned in `requirements-dev.lock.txt`. That
lock contains only local development tools; Bazel build and test dependencies
belong in Bazel module fragments and `MODULE.bazel.lock`.

Standalone binaries are installed by `build_tools/devtools/install.py` into the
selected tool environment. The Bazel lane installs Bazelisk and buildifier with
pinned URLs and SHA-256 hashes:

```bash
python dev.py bazel setup
```

The CMake lane currently has no standalone tool downloads; it uses system CMake
and CTest plus the shared Python-packaged tools.

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
