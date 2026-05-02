# loom-import-check

Importer check fixtures are normal Bazel tests. Run them through the checked-in
test targets, not by launching Python modules directly.

For importer fixture verification:

```bash
iree-bazel-test --config=asan //loom/py/loom/importers/check:check_test
iree-bazel-test --config=asan //loom/py/loom/importers/check:mlir_import_test
iree-bazel-test --config=asan //loom/py/loom/importers/check/tilelang:tilelang_test
iree-bazel-test --config=asan \
    //loom/py/loom/importers/check/tilelang:tilelang_import_test
```

To accept updated inline Loom output for checked-in importer fixtures, pass the
runner update flag through Bazel:

```bash
iree-bazel-test --config=asan <import-check-test-target> --test_arg=--update
```

The `iree-bazel-test` wrapper detects `--test_arg=--update` and uses Bazel's
standalone TestRunner strategy so update-capable tests can rewrite checked-in
fixture files. The `--update` flag is consumed by the tiny pytest-style runner
so it is not treated as another module name. Fixture tests that support update
mode then rewrite their inline `# ----` expected-output sections and still fail
if the importer crashes or reports a failed case.

Python fixture files keep shared imports at the top, split cases with
`# ====`, and compare imported Loom IR against the inline `# ----` section.
Use update mode after intentional importer output changes instead of editing
the expected Loom IR by hand.

Importer check file sets are owned by BUILD filegroups. Add new checked fixture
files to the relevant filegroup and pass them through the test target with
`$(locations ...)`; do not hardcode testdata paths inside Python tests.

Expected diagnostics use source annotations instead of bespoke test callbacks:

```python
# ERROR@+1: TYPE/001 {field_a="lhs"} "same type"
bad_program()
```

Annotations follow the loom-check matcher shape: uppercase severity,
optional `@+N`/`@-N` line offset, optional `DOMAIN/CODE`, optional structured
diagnostic parameter matchers in `{name="value"}`, and zero or more quoted
message substrings. Structured params match the core Python `Diagnostic`
payload; message substrings are for human-facing prose checks.

Agent-oriented usage is available from:

```bash
iree-bazel-run //loom/py/loom/importers/check:loom_import_check -- --agent_md
```
