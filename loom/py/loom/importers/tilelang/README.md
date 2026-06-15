# TileLang Importer

The TileLang importer converts selected TileLang kernel programs into Loom IR.
Its product value is not blanket TileLang compatibility; it is a controlled
path for ported TileLang programs where TileLang/TVM can also act as a local
oracle for source and code-object comparison.

Unsupported TileLang constructs are reported as structured diagnostics. Oracle
artifacts are sidecar evidence for comparison; they are not part of the Loom IR
that the importer produces.

## Environment

TileLang has a managed importer environment. The lock lives at the repository
root:

```text
requirements-importers-tilelang.lock.txt
```

The input file is `requirements-importers-tilelang.in`; it records the lock
refresh command and intentionally keeps TileLang's Python dependency closure
outside the base developer environment.

Set up and probe the environment with:

```bash
python dev.py importers setup tilelang
python dev.py importers doctor tilelang
python dev.py importers env tilelang --format=shell
```

The setup command installs from the hash-locked requirements into
`.tmp/importers/tilelang/venv/` and writes a manifest under
`.tmp/importers/tilelang/environment.json`.

## Bazel

Use `--importer-env tilelang` for normal local and CI verification. It selects
`--config=loom-importer-tilelang` and forwards the locked site-packages path to
tests:

```bash
iree-bazel-test --config=asan \
    --importer-env tilelang \
    //loom/py/loom/importers/check/tilelang:tilelang_test \
    //loom/py/loom/importers/tilelang:tilelang_test \
    //loom/py/loom/importers/tilelang:tilelang_import_test
```

When the frontend packages are already importable without the managed
environment, the raw build selection is:

```bash
iree-bazel-test --config=asan \
    --config=loom-importer-tilelang \
    //loom/py/loom/importers/tilelang:tilelang_import_test
```

The native build setting is `--//loom/config/import:enable=tilelang`.

## CMake

CMake uses the same optionality boundary through `LOOM_IMPORT_TILELANG`:

```bash
iree-cmake-configure --importer-env tilelang
iree-cmake-build --importer-env tilelang loom_tools_loom-opt_loom-opt
iree-cmake-test --importer-env tilelang -R tilelang
```

`--importer-env tilelang` adds `-DLOOM_IMPORT_TILELANG=ON` at configure time
and runs CMake build/test commands with the locked importer environment on
`PYTHONPATH`.

## Fixture Checks

The importer golden tests are ordinary Bazel and CMake tests. Fixture files
live under `testdata/`, and checked output is stored inline in the source file
after each `# ----` marker.

Update intentional output changes through Bazel:

```bash
iree-bazel-test --config=asan \
    --importer-env tilelang \
    --test_arg=--update \
    //loom/py/loom/importers/tilelang:tilelang_import_test
```

The shared fixture runner and source annotation format are documented in
`../check/README.md`.

## Oracle Capture

TileLang checks can optionally ask TileLang/TVM for generated artifacts. Source
oracle mode captures generated device source; code-object mode additionally
compiles, unbundles, and disassembles when the local ROCm and LLVM tools are
available.

```bash
iree-bazel-run --config=asan \
    --importer-env tilelang \
    //loom/py/loom/importers/check:loom_import_check -- \
    tilelang \
    --oracle=source \
    --oracle-output-dir=build/tilelang-oracle \
    loom/py/loom/importers/tilelang/testdata/tilekernels/transpose.py
```

The checked stdout remains imported Loom IR. Oracle output is retained only
when `--oracle-output-dir` or `--dump-temp-dir` is supplied.
