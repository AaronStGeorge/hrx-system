# Model cards

A model card describes a published model independently of its host application.
`ModelCard.parse` validates plain JSON, and `select` resolves one service,
variant, and provider set. Neither operation reads weights, invokes a compiler,
or performs network access. The native `loom-serve` consumer reads the same
selected facts through a deployment document's `model` reference.

| Section | Meaning |
| --- | --- |
| `model` | Original training-source identity and implementation provenance. |
| `origins` | Shared acquisition roots, referenced as `@name/path`. |
| `programs` | Named sets of linker inputs, libraries, auxiliary data, and default JIT configuration. |
| `services` | Named service contracts with independent ABI versions. |
| `variants` | Compatible provider sets, fixed JIT configuration, scoped parameters, and named tokenizers. |
| `configurations` | Canonical selections with execution, request, workload, and preparation options. |

`schema_version: 1` selects the bundled [editor schema](model-card.v1.schema.json).
`$schema` associates it with an editor; consumers never download that URI to
validate a card. JSON comments, duplicate object keys, nonfinite numbers,
unknown structural fields, missing references, incompatible selections, and
escaping destination paths are errors. The initial qualification field permits
only an empty object. The Moonshine metadata example declares inputs and service
geometry; it makes no executable qualification claim.

Provider-set names have no encoding semantics. A set named `source` may include
bytecode libraries. `inputs` and `libraries` are arrays of path strings or
objects with `path`, optional `format`, importer `options`, and auxiliary `data`.
Auxiliary files remain outside the linker index. A program's `directory` names
its execution context without rebasing the card's file paths.

Every file path describes local placement relative to the card. Its `origin`
describes acquisition separately. For example:

```json
{
  "path": "vocabulary.data",
  "origin": "@checkpoint/tokenizer/tokenizer.json",
  "format": "huggingface-json"
}
```

A program set or parameter scope can declare an origin directory such as
`@checkpoint/`. Entries without an explicit origin append their destination path
exactly once. Without either declaration, a file's origin is its card-relative
path. Model provenance is never an implicit acquisition default. Hashes and
resolved source identities belong to acquisition locks, outside the model card.

`parameters.<scope>.files` is an ordered shard list populating one parameter
index. `tokenizers.<identity>` declares a tokenizer, independently of its
filename. Both accept strings or explicit file descriptors. Variant `config`
scalars override program defaults and reach preparation and native JITs.
Canonical `options` may use only names accepted by the selected service.

A service's kind and version define exported roles, such as `text_prepare_v1`.
An omitted version selects development names such as `text_prepare`; it does
not select the latest ABI. Several services and revisions may share a provider
set. Schema, service, and bytecode revisions have independent compatibility
checks. Native support is documented in the
[serving model contract](../../../docs/src/serving/model-documents.md).

## Bazel integration

`loom_model`, exported by `//loom/build_tools/bazel:defs.bzl`, takes a card,
canonical configuration, optional provider-set override, output encoding, and
explicit Bazel inputs. `files` maps targets to the selected linker-provider
paths. `assets` maps targets to all remaining selected paths, including provider
auxiliary data, parameters, and tokenizers. Each target supplies exactly one
file; missing, extra, overlapping, and conflicting declarations fail.

The rule performs no acquisition. It normalizes each provider through the
configured public `loom-link` toolchain, preserving provider order and roles.
It does not choose service roots or specialize model geometry. Auxiliary files
and weights appear at their declared paths as symlink artifacts. Provider
importers see the same relative namespace during compilation.

The output contains the selected portable `model.json`, normalized providers,
assets, and a separate `serve.json` referencing that card. The default output is
`serve.json`; the `model_card` and `model_files` output groups expose the card
and complete package for publication. The input card and staged source files
are action inputs, not runtime dependencies of the resulting package.

Tests consume the emitted package and their runner. Normal Bazel content
identity determines reuse: a build rerun that produces identical outputs does
not itself invalidate a test; a changed executable, selected card, or asset does.
The weighted text fixture checks an actual response through this path. The
recognition fixture checks activity and transcription as independent services
sharing one provider universe, including concurrent streams and both bytecode
and text build outputs.
