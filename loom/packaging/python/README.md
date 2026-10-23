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
Linux wheel packaging gives each native file an adjacent-library search path;
installed commands load their matching companions without Bazel build paths.

The Python APIs use the standard library. The optional `hf` extra enables Hugging
Face acquisition; local and HTTP model sources work without it. Model weights
are acquired separately with `loom-fetch`.

Build the development wheel through Bazel and install it with Python 3.12 or
newer. For Linux x86-64:

```sh
iree-bazel-build --config=loom-sdk --config=opt //loom/packaging/python:wheel
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

`--config=loom-sdk` selects the SDK's CPU and libamdf runtime drivers, compiler
targets, and media codecs independently of the worktree configuration. Native
AMD execution uses the operating-system device driver; it requires no ROCm SDK.
The `amdgpu2`, `rdna`, `cdna`, and `xdna` drivers use libamdf. The legacy `amdgpu`
driver is excluded from this configuration. Instrumentation and optimization
are selected separately, as shown below.

Installed users need Python, the wheel, and the platform prerequisites in the
[package description](PYPI_README.md); Bazel is a package-author build tool.
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

The checked-in [version.json](version.json) owns the SDK's base package version
and compatible model catalog association. Bazel reads it during analysis; builds
have no environment or timestamp version override. The current development
identity is `0.0.0.dev0` with no associated catalog. This target builds a local
artifact and has no publishing command.

The native build configuration determines the installed version. Publishers
supply a base version without a `+` suffix; the wheel rule adds the matching
suffix automatically:

| Build configuration | Version | Native behavior |
| --- | --- | --- |
| `--config=opt` | Base version | Optimized compiler and runtime. |
| `--config=opt --//runtime/config/base/tracing:provider=tracy` | Base version plus `+tracy` | Optimized with Tracy instrumentation. |
| `--compilation_mode=dbg` | Base version plus `+debug` | Debug compiler and runtime. |
| `--config=asan` | Base version plus `+asan` | AddressSanitizer with debug information. |

An uninstrumented fastbuild is allowed for a `.dev` version. Release versions
require one of the configurations above. Unsupported combinations fail during
Bazel analysis, so a wheel cannot silently lose its instrumentation identity.
Use the repository's complete `--config=opt` profile for optimized artifacts.

All profiles install the same distribution and commands. Pip replaces the
complete SDK when installing a different profile into the same environment;
separate virtual environments keep profiles side by side. The optional `[hf]`
extra works with every profile and does not choose a native build.

For example, the Tracy development wheel on Linux x86-64 is built and installed
with:

```sh
iree-bazel-build --config=loom-sdk --config=opt \
  --//runtime/config/base/tracing:provider=tracy //loom/packaging/python:wheel
python -m pip install bazel-bin/loom/packaging/python/hrx_loom-0.0.0.dev0+tracy-py3-none-linux_x86_64.whl
```

Tracy records startup events before a collector attaches. The
[tracing guide](../../../runtime/src/iree/base/tracing/README.md) describes the
matching collector version, connection, and capture lifetime. An instrumented
build is intended for deliberate capture; it is not an idle runtime switch.

Release publishing keeps normal wheels on the package index and instrumented
wheels as explicitly selected release assets. PyPI does not accept local-version
suffixes. A requirement such as `hrx-loom==1.2.3` also accepts `1.2.3+tracy`, so
switching back selects the normal wheel file or URL explicitly. This target
does not publish either kind of artifact.

SDK publishers set `package-version` and the independent `model-catalog`
association in `version.json`, such as
`github:OWNER/REPOSITORY@COMPATIBLE_TAG`. Several SDK versions
may share a tag when the publication is qualified for each consumer. The
association is stored in the wheel's `loom.json` distribution metadata.

With an association, `loom-fetch --list` discovers the current compatible models.
Bare `loom-fetch` opens the terminal picker; `loom-fetch --catalog=./catalog.json`
opens a local publication. Both ask for the selections and installation directory.
Without one, discovery requires an explicit `--catalog`; direct card URLs remain
usable. The development wheel carries no default tag. See the
[catalog workflow](../../py/loom/model/README.md#catalog-discovery) for generation,
selection, and offline snapshot behavior.
