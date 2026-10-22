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

The rule performs no acquisition. With `encoding="text"` or `encoding="bc"`, it
normalizes each provider through the configured public `loom-link` toolchain,
preserving provider order and roles. With `encoding="preserve"`, it links each
declared provider into the package unchanged, retaining mixed formats and
importer options without running `loom-link`.
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

## Acquisition

The [Python SDK wheel](../../../packaging/python/README.md) installs the
`loom-fetch` command and includes an optional `hf` provider extra.
`//loom/py/loom/model:loom-fetch` installs the selected closure from a local card
or raw HTTP(S) card URL. The Python API exposes `plan_fetch` and `install`;
`ContentCache` owns document reads and verified payload storage. Listing reads
only the card. Named maps remain available to ordinary JSON tools:

```sh
loom-fetch model.json --list --json | jq '.variants | keys[]'
loom-fetch model.json --configuration=streaming --output=./installed --dry-run
loom-fetch model.json --configuration=streaming --output=./installed
```

The acquisition path requires a complete matching `model.json.lock`, or an
explicit `--lockfile` path/URL. A publication lock may cover more variants;
fetch reads only the selected files. `--update-lock` computes a new exact set,
replacing the local card's sibling lock or an explicit local `--lockfile`.
For a remote card without an explicit local lock destination, the new lock is
written only beside the installed card. `--no-lock` neither reads nor writes
locks and conflicts with lock update or an explicit lockfile. `--offline`
forbids network access, including document requests; cached documents and locked
payloads remain usable. A floating mutable URL without a lock needs a local
override in offline mode because a historical payload is not a fresh resolution.

Locks use a separate [versioned editor schema](model-lock.v1.schema.json).
`origins` records requested roots and resolved repository commits. Each `files`
record contains its logical `origin`, whole-file `size`, and `sha256`. A file's
destination name, parameter scope, and JIT configuration are independent of that
identity. `--origin=checkpoint=/local/mirror` replaces transport while retaining
the original lock identity; it never falls back to the original host. Unknown
aliases, missing local files, and mismatched locked bytes fail explicitly.

Whole-file acquisition supports HTTP(S), local, inline, and Hugging Face sources.
HF support is loaded only when needed and requires the optional
`huggingface_hub>=2.2,<3` package. An explicit local/HTTP origin override needs no
HF package or upstream access check. HF downloads use the SDK's ordinary cache;
Loom verifies and shares those bytes without another payload cache. HTTP transfers
validate status, framing, byte count, and locked hash; redirect destinations never
replace logical source coordinates.

HF `blob` and `resolve` model-card URLs resolve one immutable repository snapshot.
The sibling lock and card-relative files retain that snapshot, including cached
offline reads. Named floating HF origins resolve once per operation, and lock
updates record the selected immutable revisions. Cached files avoid authentication
requests. Missing content checks repository access and selected file metadata
before payload transfer.

The Python provider accepts a `wait_for_access(action, delay, cancelled)` policy.
Without it, confirmed gates and credential/permission/rate-limit failures return
`AccessRequired` with structured remediation. A confirmed gate supplies the model
page and effective account; it never accepts terms or requests approval itself.
One coordinator refreshes credentials and waits across all shard workers. Generic
permission and inaccessible-repository failures retain their distinct meanings.

On an interactive terminal, `loom-fetch` displays the approval URL and effective
account, then waits and rechecks with backoff. `o` opens the page, Enter checks
again, and `q` or Ctrl-C cancels. An approval on the website resumes the selected
command; the downloader reloads saved credentials between checks. An environment
token still overrides that saved login, so changing it requires restarting the
command with the variable updated or unset.

`--no-input` and `--json` disable terminal interaction. A gate returns a nonzero
status immediately; JSON mode emits an `action_required` record containing the
repository, reason, URL, account, and credential source name. Explicit
`--wait-for-access` enables automatic rechecking in these modes, with status on
stderr and the final result on stdout. Offline operation never checks access.

Ctrl-C also stops active metadata requests and file transfers. Cancellation joins
the operation's writers before releasing the installation, keeps completed cache
files reusable, and removes its incomplete ordinary HTTP staging files. Hugging
Face retains ownership of its cache and resumable partial downloads. The Python
API exposes the same operation cancellation through `ContentCache.cancel()`.

Files stream into verified content storage with bounded buffers and concurrent
transfers. The default `--materialize=share` links cache or local-source files
into the installation, so changing only configuration does not copy weights.
Filesystems without sharing require an explicit `--materialize=copy` choice.
Shared files remain shared: editing one view can alter another, and a later
locked fetch detects that modification. Verified existing destinations can be
reused; conflicts fail before payload transfer. Fetch never prunes an output.

The selected `model.json` is published after the complete closure verifies.
An installation lock is a projection of those verified commitments. Relative
runtime paths are retained, while original card-relative acquisition sources
are anchored to a document origin. A serving deployment references the local
card separately; native serving performs no network access or lock processing.
The fetch-to-serve fixture checks source, bytecode, mixed, and offline packages
against the same weighted response, and rejects substituted code before
publication.
