# Dependency Management

IREE has one dependency intent model with Bazel and CMake materializations.
Dependency declarations live next to the owner of the dependency, while
build-system machinery lives under the build-system directory that implements
it.

The default build path should use pinned, hash-verified dependencies owned by
the repository. Package discovery and caller-supplied dependencies are explicit
integration modes; they should not happen as accidental fallback behavior.

## Ownership

Shared dependencies are owned by root infrastructure:

```text
build_tools/third_party/deps.MODULE.bazel
build_tools/third_party/deps.cmake
build_tools/third_party/<dependency>/<dependency>.cmake
```

Project-owned dependencies use the same shape under the owning project:

```text
runtime/build_tools/third_party/deps.MODULE.bazel
runtime/build_tools/third_party/deps.cmake
runtime/build_tools/third_party/<dependency>/<dependency>.cmake

libhrx/build_tools/third_party/deps.MODULE.bazel
libhrx/build_tools/third_party/deps.cmake
libhrx/build_tools/third_party/<dependency>/<dependency>.cmake
```

Build-system dependencies and implementation machinery are separate from
third-party library declarations:

```text
build_tools/bazel/deps.MODULE.bazel
build_tools/bazel/deps.bzl
build_tools/cmake/iree_dependencies.cmake
build_tools/bazel_to_cmake/deps.py
```

The root `MODULE.bazel` should remain a short identity file plus ordered
`include()` calls for these dependency fragments.

## Classification

Every dependency must have one canonical repo-local name and one owner.
Promotion is mechanical: when two projects consume a project-owned dependency,
move the declaration and any shared adapter contract to
`build_tools/third_party/`.

Dependencies fall into these categories:

- `production`: required by shipping library or tool targets.
- `test`: required by tests that may be enabled in normal development builds.
- `dev`: required only by repository development, analysis tests, or local
  infrastructure.
- `toolchain`: required to build artifacts for a target platform.
- `sdk`: supplied by a platform SDK or vendor stack instead of source archives.

Production dependencies must not depend on dev-only repositories. Tests and
benchmarks should keep their dependencies behind the build options that enable
them so embedders are not forced to fetch unused packages.

## Bazel

Bazel dependency declarations currently use normal Bzlmod APIs in
`deps.MODULE.bazel` owner fragments. The root module includes those fragments
directly. An owner fragment is self-contained: it declares the module extension
or repo rule proxy it needs, declares repositories, and calls `use_repo()` for
the repositories made visible to that module.

Direct `bazel_dep()` entries belong in the narrowest owner fragment:

- Bazel rule sets and build tooling: `build_tools/bazel/deps.MODULE.bazel`.
- Shared third-party libraries/tools: `build_tools/third_party/deps.MODULE.bazel`.
- Runtime-only dependencies: `runtime/build_tools/third_party/deps.MODULE.bazel`.
- LibHRX-only dependencies: `libhrx/build_tools/third_party/deps.MODULE.bazel`.

Source archives should move behind the repository dependency extension before
their CMake pins are changed. The extension entry is responsible for recording
the URL, hash, strip prefix, Bazel overlay, CMake package name, CMake target
contract, license, and homepage in one place.

## CMake

CMake dependency inventory files use the same owner layout as Bazel once a
dependency migrates to generated source identity metadata:

```text
*/build_tools/third_party/deps.cmake
```

Generated `deps.cmake` files should contain source identity and dependency
mode declarations. Dependency-specific adapters, such as
`runtime/build_tools/third_party/flatcc/flatcc.cmake`, own target construction
and alias policy. Adapters must not repeat URL, hash, version, or dependency
mode data once the dependency has migrated to generated metadata.

The reproducible default mode should use repository-pinned source archives.
Package mode should require the caller to provide packages through
`CMAKE_PREFIX_PATH`, toolchain files, or explicit cache variables. Auto mode is
only acceptable when it reports which path was selected and still uses pinned
source archives for fallback.

Current CMake adapters still own their FetchContent declarations directly. A
dependency is fully migrated only when the source identity lives in the matching
`deps.cmake` inventory and the adapter only constructs the target contract.

## Adding Or Updating A Dependency

1. Identify the owner: root shared infrastructure, runtime, libhrx, loom, or a
   build-system/toolchain owner.
2. Add or update the Bazel declaration in that owner's `deps.MODULE.bazel`.
3. Add or update the CMake declaration in the matching `deps.cmake` once that
   owner has migrated to generated CMake dependency inventory.
4. Keep dependency-specific target construction in a small adapter under the
   same owner directory.
5. Document the target contract in the adapter and in this playbook when the
   dependency is public to other packages.
6. Regenerate checked-in dependency metadata and lockfiles with the documented
   dependency update command.
7. Run focused Bazel and CMake tests that prove the dependency's targets and
   package/source modes.

## Target Contracts

Checked-in adapters define the target contracts consumed by the repo:

- Googletest: `gtest`, `gtest_main`, `gmock`, `gmock_main`.
- Google benchmark: `benchmark`.
- Flatcc: `flatcc`, `iree-flatcc-cli`, `flatcc::parsing`,
  `flatcc::runtime`.
- Libbacktrace: `libbacktrace::libbacktrace` or an empty
  `${IREE_LIBBACKTRACE_TARGET}` when disabled or unsupported.
- Catch2: `Catch2::Catch2`.

Build files should depend on repo-local aliases where that improves
portability and where embedders need stable labels.
