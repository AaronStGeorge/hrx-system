# Sandboxing

The Linux SDK includes `loom-serve-sandbox`: a small launcher that runs
`loom-serve` without host networking, with explicit model mounts and accelerator
access. The same HTTP API, dashboard, and WebSocket streams remain available
through a filesystem Unix socket. Model files stay on disk; even very large
weight stores are bind mounted, never copied into a container.

## Linux

Install the [Linux Loom SDK](../getting-started/acquiring-loom.md) and your
distribution's `bubblewrap` package. The launcher uses Bash, GNU coreutils, and
`/usr/bin/bwrap`. The host must permit unprivileged user namespaces; an outer
container or host security policy can prohibit them even when the kernel
supports them. Run as an ordinary user with access to the selected devices.

Start an empty CPU server:

```sh
install -d -m 700 ./serve-socket
loom-serve-sandbox --socket-dir=./serve-socket -- \
  --device=task --pool_backing=fixed
```

In another terminal:

```sh
curl --fail --unix-socket ./serve-socket/serve.sock http://localhost/readyz
curl --fail --unix-socket ./serve-socket/serve.sock http://localhost/v1/models
```

`localhost` is the HTTP authority, not a TCP connection. The launcher exposes
this dedicated host directory as `/run/loom` and adds
`--listen=unix:/run/loom/serve.sock`. A trusted host-side proxy can relay HTTP
and WebSocket traffic to it without giving the server network access. The
[observation workflow](workflows/observe-model-server.md#watch-the-live-dashboard)
describes relaying the dashboard.

The directory must already exist, belong to your user, and have mode `0700`.
Its contents are writable by the server. Ctrl-C, SIGTERM, or SIGHUP requests
normal shutdown and waits for accepted work to retire; the server removes its
socket on successful teardown. An occupied socket path is an error, not
permission to delete another server's socket.

### Mount models and weights

Acquire models outside the sandbox using the ordinary
[fetch-and-serve workflow](workflows/fetch-and-serve.md). Expose each input root
read-only at its existing absolute path:

```sh
loom-serve-sandbox --socket-dir=./serve-socket \
  --ro-bind=/srv/models --ro-bind=/srv/weights \
  --gpu=/dev/dri/renderD128 -- \
  --device=amdgpu --model=/srv/models/chat/model.json
```

The example assumes an AMDGPU-compatible package at
`/srv/models/chat/model.json`, with any shared weights under `/srv/weights`.
`--ro-bind=PATH` is repeatable and accepts an existing regular file or directory.
The path is unchanged inside the sandbox, so document-relative references and
absolute symlinks work when all their targets have also been exposed. Nothing
walks the model tree to discover dependencies or copies its contents. An omitted
symlink target produces an ordinary file-access error.

All arguments after `--` pass unchanged to `loom-serve`. Its working directory
is `/`, so `--model`, `--devices`, and `--flagfile` use absolute mounted paths.
Model and deployment references remain document-relative. Paths read from a
flagfile use the same `/` working directory, not the flagfile's directory.
HTTP registration likewise sees only mounted paths. Additional TCP listeners
stay inside the private network namespace; they do not expose a host port.

`--binary=/path/to/sdk/bin/loom-serve` selects a different installed SDK binary
and its adjacent native libraries, for example when comparing two SDK builds.
Omitting it selects `loom-serve` beside the launcher. This expects the installed
SDK layout, not an executable that depends on build-tree runfiles. A trusted
bubblewrap installation elsewhere can be selected with `--bwrap=/path/to/bwrap`.
The launcher does not select an ambient `PATH` wrapper or accept raw bubblewrap
options.

### Select accelerators

`--gpu=/dev/dri/renderD128` exposes that render node and `/dev/kfd`.
`--npu=/dev/accel/accel0` exposes that accelerator node. Both options are
repeatable and can be combined. The launcher supplies read-only device discovery
metadata, including sysfs symlink targets and NUMA information. It does not mount
all of `/dev`, change host permissions, or install drivers.

For an XDNA-compatible package:

```sh
loom-serve-sandbox --socket-dir=./serve-socket \
  --ro-bind=/srv/models --ro-bind=/srv/weights \
  --npu=/dev/accel/accel0 -- \
  --device=xdna --pool_backing=fixed --model=/srv/models/npu/model.json
```

For a mixed GPU/NPU server, pass both device options, mount the directory
containing your [named-device configuration](index.md#devices-and-listeners),
and use `--devices=/absolute/path/devices.json` after `--`. Device exposure and
server placement are separate: the mount permits access; the device URI and
deployment select where work runs. Discovery metadata can describe devices
whose nodes have not been granted.

### Write profiling captures

`--rw-bind=PATH` exposes an existing output file or directory read-write at the
same absolute path. It is repeatable and opt-in. Keep output directories separate
from model and weight trees:

```sh
install -d -m 700 ./captures
loom-serve-sandbox --socket-dir=./serve-socket \
  --ro-bind=/srv/models --ro-bind=/srv/weights \
  --rw-bind="$PWD/captures" --gpu=/dev/dri/renderD128 -- \
  --device=amdgpu --model=/srv/models/chat/model.json \
  --device_profiling_mode=queue-events \
  --device_profiling_output="$PWD/captures/serve.ireeprof"
```

Send requests, then stop the launcher normally so the HAL profiling session
finishes. The capture remains in the host directory. Supported profiling data
families depend on the selected driver; `loom-serve --help=all` lists the linked
profiling options. Full compiler-report artifact dumping is not a serving flag.

Ordinary stdout/stderr still reach the caller and can be redirected on the host
without a writable mount. Collecting `/v1/status`, `/metrics`, or `/logz` through
the socket also needs no server-side writable directory.

### Boundary and failures

The launcher creates private user, mount, PID, IPC, UTS, and network namespaces,
drops capabilities, prevents new privileges and nested user namespaces, clears
the environment, and closes inherited descriptors other than stdout/stderr.
Stdin is `/dev/null`. System runtime directories and the selected SDK directory
are read-only; the host home, credentials, and ordinary host filesystem are not
implicitly mounted. Temporary files live in a private `/tmp` and disappear with
the sandbox.

Explicit mounts are trusted operator grants. A writable alias of a weight tree
makes those files writable despite another read-only mount. A mounted directory
containing host filesystem Unix sockets can expose those services despite the
private network namespace. Dedicated input, output, and socket directories keep
that boundary inspectable. The socket has no built-in authentication: access to
it grants the ordinary serving and administration API.

The host kernel, accelerator drivers, and accelerator memory isolation remain
trusted. This is not a defense against driver bugs or device denial of service,
and it sets no CPU, host-memory, or device-memory quota. Deployment-level resource
controls remain separate; a process memory limit does not necessarily account
for all accelerator allocations.

The launcher checks the complete namespace/mount policy before starting the
server. Failure leaves the policy intact; there is no privileged or unsandboxed
retry.

| Diagnostic | Action |
| --- | --- |
| Missing bubblewrap or unsupported option | Install a current distribution package, or select a trusted binary with `--bwrap=PATH`. |
| Namespace creation or sandbox setup denied | Check the host's user-namespace policy and any outer container/LSM restrictions. A host administrator must authorize this boundary; the launcher never retries with elevated privileges. |
| Device lacks read/write access | Check the host driver, device node, group/ACL, and outer device policy using the [device-access checklist](../getting-started/acquiring-loom.md#check-device-access-inside-a-sandbox). |
| Required sysfs metadata is absent | Check that the host driver is loaded and an outer sandbox exposes its discovery metadata as well as its device nodes. |
| Model file or symlink target is absent | Add the narrowest read-only root containing that dependency. |
| Output path is not writable | Create a dedicated output directory and grant it with `--rw-bind=PATH`; the host user's permissions still apply. |

`loom-serve-sandbox --help` works without initializing a device or creating a
sandbox. Platforms other than Linux use the portable `loom-serve` command with
their own deployment isolation tools.
