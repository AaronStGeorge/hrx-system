# Loom SDK wheel

The `hrx-loom` wheel contains the Loom compiler, runtime, command-line tools, and
`loom` Python authoring and model APIs. Pip installs the commands into the Python
environment's `bin/` directory on Unix or `Scripts/` directory on Windows. An
activated virtual environment makes them available on PATH.

The native commands are `loom-benchmark`, `loom-check`, `loom-compile`,
`loom-format`, `loom-import-cxx`, `loom-link`, `loom-opt`, `loom-run`, `loom-serve`,
and `loom-test`. Python supplies `loom-compile-report`, `loom-export-cxx`,
`loom-fetch`, and `loom-lint`. The wheel includes the matching native shared
libraries beside the executables. C++ import includes its standard and Loom
headers, so it works without a separate compiler or source checkout.

The Python APIs use the standard library. The optional `hf` extra enables Hugging
Face acquisition; local and HTTP model sources work without it. Model weights
are acquired separately with `loom-fetch`.

Build the development wheel through Bazel and install it with Python 3.12 or
newer. For Linux x86-64:

```sh
iree-bazel-build --config=opt //loom/packaging/python:wheel
python -m pip install bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0-py3-none-linux_x86_64.whl
loom-serve --help
loom-benchmark --help
loom-fetch model.json --list --json
loom-fetch model.json --preset=check --output=./model
python -m loom.model.documentation --card model.json --output=model.md
```

To include the optional provider when installing a local wheel:

```sh
python -m pip install 'bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0-py3-none-linux_x86_64.whl[hf]'
```

The wheel filename describes the selected target platform: `linux_x86_64`,
`linux_aarch64`, or `win_amd64`. Windows cross-builds use
`--config=windows-x86_64` with the toolchain setup in
[BUILDING.md](../../../BUILDING.md#windows-from-linux). Linux development wheels
use the build toolchain's libc baseline; these tags make no manylinux portability
claim.

Installed users need Python and the wheel; Bazel is a package-author build tool.
`loom-serve --device=task --model=./model/model.json` loads the fetched card's
selected preset directly; choose a device supported by that model. A separate
serving deployment can supply routing aliases, placement, and execution
overrides.

The wheel carries versioned JSON schemas alongside `loom.model`. Applications
can read them without accessing a repository:

```python
from importlib.resources import files

schema = files("loom.model").joinpath("model-card.v1.schema.json").read_text()
```

The development version is `0.0.0.dev0`. This target builds a local artifact and
has no publishing command. Release publishers supply a version and compatible
model catalog association.

SDK publishers use `loom_python_wheel` from
`//loom/packaging/python:defs.bzl` to set `version` and an independent `catalog`
association, such as `github:OWNER/REPOSITORY@COMPATIBLE_TAG`. Several SDK versions
may share a tag when the publication is qualified for each consumer. The
association is stored in the wheel's `loom.json` distribution metadata.

With an association, `loom-fetch --list` discovers the current compatible models.
Bare `loom-fetch` opens the terminal picker; `loom-fetch --catalog=./catalog.json`
opens a local publication. Both ask for the selections and installation directory.
Without one, discovery requires an explicit `--catalog`; direct card URLs remain
usable. The development wheel carries no default tag. See the
[catalog workflow](../../py/loom/model/README.md#catalog-discovery) for generation,
selection, and offline snapshot behavior.
