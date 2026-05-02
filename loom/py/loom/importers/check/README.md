# Loom Import Check

Importer check fixtures are normal Bazel tests. Run them through the checked-in
test targets, not by launching Python modules directly.

For TileLang fixture verification:

```bash
iree-bazel-test --config=asan //loom/py/loom/importers/check/tilelang:tilelang_test
```

To accept updated inline Loom output for the checked-in TileLang fixtures, pass
the runner update flag through Bazel:

```bash
iree-bazel-test --config=asan //loom/py/loom/importers/check/tilelang:tilelang_test --test_arg=--update
```

The `--update` flag is consumed by the tiny pytest-style runner so it is not
treated as another module name. Fixture tests that support update mode then
rewrite their inline `# ----` expected-output sections and still fail if the
importer crashes or reports a failed case.
