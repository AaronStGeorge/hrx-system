# Loom Python SDK

The `hrx-loom` wheel provides the `loom` Python authoring APIs, text and bytecode
formats, portable model cards, and the `loom-fetch` command. Its base installation
uses the Python standard library. The optional `hf` extra enables Hugging Face
acquisition; local and HTTP model sources work without it.

Build and install the development wheel with Python 3.12 or newer:

```sh
iree-bazel-build //loom/packaging/python:wheel
python -m pip install bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0-py3-none-any.whl
loom-fetch model.json --list --json
loom-fetch model.json --configuration=check --output=./model
```

To include the optional provider when installing a local wheel:

```sh
python -m pip install 'bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0-py3-none-any.whl[hf]'
```

The native SDK supplies `loom-serve` and the compiler executables. A serving
deployment references the fetched local `model.json`; model acquisition and
serving configuration remain separate operations.

The wheel carries versioned JSON schemas alongside `loom.model`. Applications
can read them without accessing a repository:

```python
from importlib.resources import files

schema = files("loom.model").joinpath("model-card.v1.schema.json").read_text()
```

The development version is `0.0.0.dev0`. This target builds a local artifact and
has no publishing command. Native releases and a qualified model catalog require
their own release association.

SDK publishers use `loom_python_wheel` from
`//loom/packaging/python:defs.bzl` to set `version` and an independent `catalog`
association, such as `github:OWNER/REPOSITORY@COMPATIBLE_TAG`. Several SDK versions
may share a tag when the publication is qualified for each consumer. The
association is stored in the wheel's `loom.json` distribution metadata.

With an association, `loom-fetch --list` discovers the current compatible models.
Without one, discovery requires an explicit `--catalog`; direct card URLs remain
usable. The development wheel carries no default tag. See the
[catalog workflow](../../py/loom/model/README.md#catalog-discovery) for generation,
selection, and offline snapshot behavior.
