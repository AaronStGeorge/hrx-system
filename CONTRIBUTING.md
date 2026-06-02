# Contributing

This repository uses Bazel as the source of truth for build graph structure and
Lefthook as the local presubmit dispatcher. Local developer tools stay outside
Bazel unless a Bazel target actually executes them.

## Tool Setup

Create a local virtual environment, install the locked Python-packaged tools,
and install the pinned standalone tools:

```bash
python -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev.lock.txt
.venv/bin/python build_tools/devtools/install.py
```

The Python lock installs Lefthook, Ruff, and clang-format. The standalone
installer places Bazelisk and buildifier under `.venv/bin`; Bazelisk is also
linked as `bazel` and follows `.bazelversion`.

## Before Commit

Run the staged-file precommit helper before committing:

```bash
build_tools/lefthook/precommit.sh
```

The helper applies supported fixups, stages the files owned by those fixups,
and then runs the same hygiene checks in non-mutating mode.

The stronger local presubmit mode also runs affected project Bazel tests:

```bash
build_tools/lefthook/presubmit.sh
```

Runtime paths run the runtime project presubmit, libhrx paths run the libhrx
project presubmit, and shared build/dependency paths fan out to every configured
project entry point.

## Git Hook

Install the Git hook from the repository root:

```bash
lefthook install
```

The Git `pre-commit` hook is check-only. Mechanical updates happen through
`build_tools/lefthook/precommit.sh`, so a successful `git commit` cannot leave
the worktree dirty or commit stale generated/formatted content.

## CI Entry Point

CI runs the same non-mutating repository profile:

```bash
lefthook run check
```

The `ci` profile checks all tracked files, runs generated-file checks without
writing, and dispatches all configured project presubmit tests.

More detail on profiles, tool installation, and project dispatch lives in
`build_tools/lefthook/README.md`.
