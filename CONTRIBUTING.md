# Contributing

This repository uses Bazel as the source of truth for build graph structure and
uses CMake for package/install-test workflows. `dev.py` is the blessed command
router for local development. It selects a structural build lane, prepares the
tool environment, and delegates real work to Bazel, CMake, Lefthook, Git, and
project-local scripts.

## Build Lanes

Bazel/source graph lane:

```bash
python dev.py bazel setup
python dev.py bazel hook
python dev.py bazel precommit
```

CMake/package lane:

```bash
python dev.py cmake setup
python dev.py cmake hook
python dev.py cmake precommit
```

The build lane is part of the command structure instead of a global flag. That
keeps hooks and agent runs unambiguous: a Bazel hook runs Bazel-lane checks, and
a CMake hook runs CMake-lane checks.

## Tool Modes

Standalone checkouts normally use the default repo-local tool environment:

```bash
python dev.py bazel setup --venv
```

Embedding or superproject workflows can use system tools without writing into
the submodule:

```bash
python dev.py bazel setup --system
```

Superprojects can also keep a managed tool environment outside this checkout:

```bash
python dev.py bazel setup --tool-root ../.tools/iree-x
```

The same modes work for the CMake lane. `--dry-run` prints the exact command
plan without executing it.

## Manual Build Commands

Use the lane command names when you want to run Bazel or CMake work directly:

```bash
python dev.py bazel configure
python dev.py bazel build
python dev.py bazel test
```

or:

```bash
python dev.py cmake configure
python dev.py cmake build
python dev.py cmake test
```

With no explicit targets, the Bazel build and test commands cover
`//runtime/...` and `//libhrx/...`. In the CMake lane, positional build
arguments are target names, so `python dev.py cmake build hrx` maps to
`cmake --build ... --target hrx`. The CMake lane writes its build tree outside
the checkout at `../builds/<checkout-name>/`.

## Before Commit

Use `precommit` for the current local change set:

```bash
python dev.py bazel precommit
```

or:

```bash
python dev.py cmake precommit
```

With no input option, `precommit` checks staged, unstaged, and untracked files.
Use `--base <git-ref>` to check the branch diff from the merge base through
`HEAD` plus local changes. Use `--staged` when you explicitly want staged files
only.

Use the fix command when you want mechanical repairs:

```bash
python dev.py bazel fix
```

or:

```bash
python dev.py cmake fix
```

`fix` applies staged formatting/generated-file repairs and stages only files
owned by those fixers. `presubmit` is non-mutating and runs the full-tree
CI-shaped check.

## Git Hook

Install the hook for the lane you want Git commits to check:

```bash
python dev.py bazel hook
```

or:

```bash
python dev.py cmake hook
```

The hook is check-only. It should never leave the worktree dirty after a
successful commit. Lane-specific hook policy is stored in ignored
`lefthook-local.yml` and generated from checked-in templates under
`build_tools/lefthook/`.

## Verification

Core developer-tool tests:

```bash
bazel test --config=presubmit //build_tools/devtools:cli_test //build_tools/devtools:command_plan_test //build_tools/devtools:setup_test
```

Checkout smoke test for the command router:

```bash
python build_tools/devtools/smoke_test.py --from-working-tree --scenario dry-run
```

The smoke test creates a temporary checkout, runs dry-run setup/hook/presubmit
commands, and verifies those dry-runs do not create a venv, hook config, or
external tool root.

More detail on profiles and hook internals lives in
`build_tools/lefthook/README.md`.
