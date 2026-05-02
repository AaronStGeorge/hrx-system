# loom-check

`loom-check` is the golden-test runner for `.loom-test` files. It splits each
file into `// ====` cases, runs the selected `// RUN:` mode, and compares the
actual output or diagnostics against the inline expectation.

Use checked-in Bazel test targets for normal verification:

```bash
iree-bazel-test --config=asan //loom/src/loom/tools/loom-check/test:test
iree-bazel-test --config=asan //loom/src/loom/...
```

To accept intentional output changes, pass the update flag through Bazel:

```bash
iree-bazel-test --config=asan <loom-check-test-target> --test_arg=--update
```

The `iree-bazel-test` wrapper detects `--test_arg=--update` and uses Bazel's
standalone TestRunner strategy so update-capable tests can rewrite checked-in
fixture files. Prefer that path over direct tool invocations when updating
repository tests.

Direct runs are useful for inspection:

```bash
iree-bazel-run //loom/src/loom/tools/loom-check -- path/to/file.loom-test
iree-bazel-run //loom/src/loom/tools/loom-check -- --update path/to/file.loom-test
```

`--update` cannot be used with stdin or verify-mode cases. Agent-oriented usage
is available from:

```bash
iree-bazel-run //loom/src/loom/tools/loom-check -- --agent_md
```
