# Model cards

A model card describes a published model independently of the application that
hosts it. `ModelCard.parse` validates the JSON and `select` resolves a canonical
configuration to one exact program and asset closure. Neither operation reads
weights, invokes a compiler, or performs network access.

The card separates these sections:

| Section | Meaning |
| --- | --- |
| `model` | Original model identity and implementation provenance. |
| `files` | Package-relative file names, acquisition origins, and optional size and SHA-256 identities. |
| `entrypoints` | Callable kind, named root groups, and named sets of linker inputs and libraries. |
| `variants` | Compatible program selections, parameter bindings, and model-owned options. |
| `configurations` | Named selections and canonical execution, request, and workload settings. |
| `qualification` (optional) | Explicit configurations and variant/program overrides for checks and benchmarks. |

`schema_version` is `1`. Unknown structural fields, duplicate JSON object keys,
missing references, incompatible selections, and escaping package paths are
errors. Model-specific interface, constraint, workload, and option values retain
their JSON types; the implementing model or workload owns their interpretation.

Program selection names have no format semantics. A selection named `source`
can contain bytecode libraries. Every selection has arrays of `inputs` and
`libraries`; a provider is a file name or an object with `file`, optional
`format`, importer `options`, and auxiliary `data` file names. Format detection
and importer options use the public linker's contract. Files describe content;
root groups describe what the selected program provides. A linked bytecode file
still contains IR for runtime linking and specialization.

A variant states which program selections it supports for each entrypoint.
Bindings refer to one file or to a directory with an explicit member inventory.
A directory is never an implicit download wildcard. Components have their own
root groups, program inputs, and bindings, and participate in the selected file
closure. Component program selection uses the containing selection's name.

Variant options describe model facts. A configuration may override only option
names explicitly declared by its entrypoint. This prevents a generic command
line from silently replacing fixed model geometry. The native model still
validates the option values it understands.

Origins describe local authoring files, pinned Hugging Face files, HTTPS files,
inline content, or declared build outputs. `build` records require a producer;
they do not imply an uploaded release exists. The Moonshine example in
`testdata/moonshine.json` records its experimental implementation and exact
checkpoint inputs; its empty qualification table makes no execution claim.

## Bazel serving integration

`loom_model` is exported by `//loom/build_tools/bazel:defs.bzl`. It takes a card,
a configuration name, an optional program selection, an output encoding, and
Bazel targets mapped to logical file names. These targets supply already
materialized inputs; this rule performs no downloads or weight transformations.
All selected files must be declared, including files whose published origin is
inline or remote. Directory bindings require the declared files to preserve the
model's member layout under one root.

The configured public `loom-link` toolchain links each root group, retains the
provider universe needed by dynamically declared stages, and emits a directory
containing programs and `serve.json`. `serve.json` is the current native serving
configuration and refers to asset runfiles without copying them. The native
adapter requires `prepare` and `control` root groups. It does not make the
portable card itself a native `loom-serve --model` input.

The input card is a build dependency, but is not a test runfile. Documentation
changes that leave selected outputs unchanged can reuse test results. Tests
still depend on the declared inputs, generated deployment, linked programs, and
runner. A cached build action alone does not establish a cached test result.

The recognition fixture exercises mixed text/bytecode providers with both
bytecode and linked text outputs through native serving, including concurrent
streams. Model publication actions use Bazel runfiles; their fixture targets
carry `skip-bazel_to_cmake`. The existing native deployment fixtures also retain
CMake coverage.
