# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launcher contract tests; bubblewrap is the substituted platform dependency."""

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAUNCHER = Path(sys.argv.pop(1)).absolute()

# Retain exact argv before emulating only bwrap's process/startup protocol.
# Filesystem/namespace isolation is qualified with real bwrap separately.
BUBBLEWRAP = r"""
import json, os, sys
from pathlib import Path
arguments = sys.argv[1:]
with open(os.environ["RECORD"], "a") as output:
    output.write(json.dumps(arguments) + "\n")
if arguments[-1] == "/usr/bin/true":
    sys.exit(int(os.environ.get("PREFLIGHT_STATUS", "0")))
os.setsid()
descriptor = int(arguments[arguments.index("--info-fd") + 1])
os.write(descriptor, b'{"child-pid": ')
if gate := os.environ.get("STARTUP_GATE"):
    print("setup", flush=True)
    with open(gate, "rb", buffering=0) as stream:
        assert stream.read(1) == b"x"
os.write(descriptor, (str(os.getpid()) + '}\n').encode())
os.close(descriptor)
command = arguments[arguments.index("--") + 1:]
command[4] = os.environ["SERVER"]
os.execv(command[0], command)
"""

SERVER = r"""
import json, os, signal, sys
if os.environ.get("WAIT_FOR_SIGNAL"):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(23))
    print("ready", flush=True)
    signal.pause()
else:
    print(json.dumps({"arguments": sys.argv[1:], "input": sys.stdin.read(),
                      "descriptors": sorted(os.listdir("/proc/self/fd"))}), flush=True)
    sys.exit(int(os.environ.get("SERVER_STATUS", "0")))
"""


class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="loom-sandbox-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.sockets = self.directory / "socket directory"
        self.sockets.mkdir(mode=0o700)
        self.record = self.directory / "arguments.jsonl"
        self.server = self.script("loom-serve", SERVER)
        self.bubblewrap = self.script("bwrap", BUBBLEWRAP)
        self.environment = dict(
            os.environ, RECORD=str(self.record), SERVER=str(self.server)
        )

    def script(self, name, body):
        path = self.directory / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o700)
        return path

    def command(self, *arguments):
        return [
            "bash",
            str(LAUNCHER),
            "--bwrap=" + str(self.bubblewrap),
            "--binary=" + str(self.server),
            "--socket-dir=" + str(self.sockets),
            *map(str, arguments),
        ]

    def run_launcher(self, *arguments, expected=0):
        result = subprocess.run(
            self.command(*arguments),
            env=self.environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, expected, (result.stdout, result.stderr))
        return result

    def calls(self):
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_help_needs_no_platform_dependencies(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--ro-bind=PATH", result.stdout)
        self.assertIn("--rw-bind=PATH", result.stdout)
        self.assertIn("--gpu=all|RENDER_NODE", result.stdout)
        self.assertIn("--npu=all|ACCEL_NODE", result.stdout)

    def test_arguments_mounts_environment_and_descriptors(self):
        source = self.directory / "weight store"
        source.mkdir()
        self.environment["SERVER_STATUS"] = "19"
        with self.server.open("rb") as inherited:
            child = subprocess.run(
                self.command(
                    "--ro-bind=" + str(source),
                    "--",
                    "--device=task",
                    "--model=/models/a name.json",
                    "--flagfile=/models/serve.flags",
                ),
                env=self.environment,
                pass_fds=(inherited.fileno(),),
                capture_output=True,
                text=True,
            )
        self.assertEqual(child.returncode, 19, child.stderr)
        response = json.loads(child.stdout)
        self.assertEqual(
            response["arguments"],
            [
                "--listen=unix:/run/loom/serve.sock",
                "--device=task",
                "--model=/models/a name.json",
                "--flagfile=/models/serve.flags",
            ],
        )
        self.assertEqual(response["input"], "")
        # listdir opens one temporary directory FD in addition to stdio.
        self.assertEqual(response["descriptors"], ["0", "1", "2", "3"])
        preflight, launch = self.calls()
        self.assertEqual(preflight[-2:], ["--", "/usr/bin/true"])
        self.assertEqual(launch[: len(preflight) - 2], preflight[:-2])
        for flag in (
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--new-session",
            "--die-with-parent",
            "--clearenv",
        ):
            self.assertIn(flag, launch)
        self.assertIn(
            ["--cap-drop", "ALL"], [launch[i : i + 2] for i in range(len(launch))]
        )
        self.assertIn(
            ["--ro-bind", str(source), str(source)],
            [launch[i : i + 3] for i in range(len(launch))],
        )
        self.assertNotIn("--dev-bind", launch)
        self.assertNotIn("/sys/devices", launch)
        # No implicit writable host path beyond the explicitly shared socket.
        self.assertEqual(launch.count("--bind"), 1)

    def test_writable_paths_and_readonly_symlink(self):
        source = self.directory / "weights"
        source.mkdir()
        alias = self.directory / "weight link"
        alias.symlink_to(source)
        outputs = self.directory / "reports: run=1"
        outputs.mkdir()
        output = self.directory / "trace.bin"
        output.touch()
        self.run_launcher(
            "--ro-bind=" + str(alias),
            "--rw-bind=" + str(outputs),
            "--rw-bind=" + str(output),
            "--",
        )
        launch = self.calls()[1]
        triples = [launch[i : i + 3] for i in range(len(launch))]
        self.assertIn(["--ro-bind", str(source), str(alias)], triples)
        for path in (outputs, output):
            self.assertIn(["--bind", str(path), str(path)], triples)

    def test_all_grants_follow_host_drivers_without_duplicates(self):
        # The launcher only reads discovery metadata; the substituted bwrap
        # records grants without opening any host device. This also covers
        # CPU-only runners, absent device classes, and restricted outer mounts.
        nodes = []
        metadata = []
        diagnostic = None
        if not os.access("/sys/class", os.R_OK | os.X_OK):
            diagnostic = "device discovery requires readable /sys/class"
        else:
            for kind, prefix, driver, directory in (
                ("drm", "renderD", "amdgpu", "/dev/dri"),
                ("accel", "accel", "amdxdna", "/dev/accel"),
            ):
                path = Path("/sys/class") / kind
                if not path.exists():
                    continue
                if not os.access(path, os.R_OK | os.X_OK):
                    diagnostic = "cannot enumerate devices"
                    break
                selected = []
                for endpoint in sorted(path.glob(prefix + "[0-9]*")):
                    try:
                        bound_driver = (endpoint / "device/driver").resolve(strict=True)
                    except OSError:
                        diagnostic = "cannot identify device driver"
                        break
                    if bound_driver.name == driver:
                        selected.append(str(Path(directory) / endpoint.name))
                if diagnostic:
                    break
                if selected:
                    nodes.extend(selected)
                    metadata.append(str(path))
                    if kind == "drm":
                        metadata.append("/sys/class/kfd")
            if not diagnostic and nodes:
                metadata += ["/sys/devices", "/sys/dev/char"]
                if "/sys/class/kfd" in metadata:
                    nodes.append("/dev/kfd")
                if any(not os.access(path, os.R_OK | os.X_OK) for path in metadata):
                    diagnostic = "required device discovery metadata is unavailable"
                elif any(
                    not Path(node).is_char_device()
                    or not os.access(node, os.R_OK | os.W_OK)
                    for node in nodes
                ):
                    diagnostic = "device requires read/write access"
        result = self.run_launcher(
            "--gpu=all",
            "--npu=all",
            "--gpu=all",
            "--npu=all",
            "--",
            expected=1 if diagnostic else 0,
        )
        if diagnostic:
            self.assertIn(diagnostic, result.stderr)
            self.assertFalse(self.record.exists())
        else:
            launch = self.calls()[1]
            granted = [
                launch[i + 1] for i, flag in enumerate(launch) if flag == "--dev-bind"
            ]
            self.assertCountEqual(granted, nodes)
            for path in ("/dev", "/dev/dri", "/dev/accel"):
                self.assertNotIn(path, granted)
            if not nodes:
                self.assertIn("--gpu=all: none present", result.stderr)
                self.assertIn("--npu=all: none present", result.stderr)
                self.assertNotIn("/sys/devices", launch)

    def test_output_must_already_exist_and_be_writable(self):
        output = self.directory / "output"
        result = self.run_launcher("--rw-bind=" + str(output), expected=1)
        self.assertIn("cannot resolve mount source", result.stderr)
        output.touch(mode=0o400)
        # A privileged outer container can write despite the file mode. The
        # launcher checks actual access, not a guessed interpretation of mode.
        writable = os.access(output, os.W_OK)
        result = self.run_launcher(
            "--rw-bind=" + str(output), expected=0 if writable else 1
        )
        if not writable:
            self.assertIn("not writable by your user", result.stderr)
            self.assertFalse(self.record.exists())

    def test_policy_failure_never_launches_server(self):
        self.environment["PREFLIGHT_STATUS"] = "9"
        result = self.run_launcher("--", expected=1)
        self.assertIn("unprivileged user namespaces", result.stderr)
        self.assertIn("no server was started", result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_binary_equals_selects_the_matching_directory(self):
        self.run_launcher("--binary=" + str(self.server), "--")
        launch = self.calls()[1]
        self.assertIn(
            ["--ro-bind", str(self.directory), "/opt/loom/bin"],
            [launch[i : i + 3] for i in range(len(launch))],
        )

    def test_invalid_mounts_and_raw_options(self):
        for arguments, diagnostic in (
            (["--share-net"], "unknown sandbox option"),
            (["--ro-bind=/usr"], "reserved"),
            (["--ro-bind=/tmp/../usr"], "reserved"),
            (["--ro-bind=relative"], "absolute"),
            (["--ro-bind=/proc"], "not a regular input/output"),
            (["--ro-bind=/dev/null"], "regular file"),
            (["--ro-bind", self.directory], "--option=value"),
            (["--rw-bind=/usr"], "reserved"),
            (["--rw-bind=/tmp/../usr"], "reserved"),
            (["--rw-bind=relative"], "absolute"),
            (["--rw-bind=/proc"], "not a regular input/output"),
            (["--rw-bind=/dev/null"], "regular file"),
            (["--rw-bind", self.directory], "--option=value"),
            (["--npu=/dev/null"], "accelerator node"),
            (["--gpu=/dev/null"], "DRM render node"),
            (["--socket-dir="], "requires a path"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_launcher(*arguments, expected=1)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse(self.record.exists())

    def test_socket_directory_must_be_private(self):
        self.sockets.chmod(0o755)
        result = self.run_launcher("--", expected=1)
        self.assertIn("mode 0700", result.stderr)
        self.assertEqual(stat.S_IMODE(self.sockets.stat().st_mode), 0o755)
        self.assertFalse(self.record.exists())

    def test_socket_directory_created_once_without_touching_contents(self):
        self.sockets = self.directory / "new socket directory"
        self.run_launcher("--")
        self.assertEqual(stat.S_IMODE(self.sockets.stat().st_mode), 0o700)
        self.assertEqual(self.sockets.stat().st_uid, os.geteuid())
        sentinel = self.sockets / "serve.sock"
        sentinel.write_bytes(b"owned by another server")
        self.run_launcher("--")
        self.assertEqual(sentinel.read_bytes(), b"owned by another server")

    def test_socket_directory_needs_an_existing_parent(self):
        self.sockets = self.directory / "missing parent" / "socket"
        result = self.run_launcher("--", expected=1)
        self.assertIn("parent must already exist", result.stderr)
        self.assertFalse(self.sockets.parent.exists())
        self.assertFalse(self.record.exists())

    def test_socket_directory_rejects_files_and_dangling_symlinks(self):
        target = self.directory / "not a directory"
        target.write_text("preserved")
        self.sockets = target
        result = self.run_launcher("--", expected=1)
        self.assertIn("must be a directory", result.stderr)
        self.assertEqual(target.read_text(), "preserved")
        self.sockets = self.directory / "dangling"
        destination = self.directory / "absent"
        self.sockets.symlink_to(destination)
        result = self.run_launcher("--", expected=1)
        self.assertIn("cannot resolve socket directory", result.stderr)
        self.assertFalse(destination.exists())
        self.assertFalse(self.record.exists())

    def test_unsupported_platform_and_missing_bubblewrap(self):
        self.script("uname", 'print("Darwin")\n')
        self.environment["PATH"] = str(self.directory) + os.pathsep + os.environ["PATH"]
        result = self.run_launcher("--ro-bind=" + str(self.directory), expected=1)
        self.assertIn("supported on Linux only", result.stderr)
        self.environment["PATH"] = os.environ["PATH"]
        result = self.run_launcher(
            "--bwrap=" + str(self.directory / "missing"), expected=1
        )
        self.assertIn("install your distribution's bubblewrap", result.stderr)

    def test_signals_are_forwarded_and_joined(self):
        self.environment["WAIT_FOR_SIGNAL"] = "1"
        for event in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(event=event):
                with subprocess.Popen(
                    self.command("--"),
                    env=self.environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as child:
                    try:
                        self.assertEqual(child.stdout.readline(), "ready\n")
                        child.send_signal(event)
                        output, error = child.communicate()
                        self.assertEqual(child.returncode, 23, (output, error))
                    finally:
                        if child.poll() is None:
                            child.terminate()
                        child.wait()

    def test_startup_signal_preserves_partial_identity(self):
        gate = self.directory / "startup"
        os.mkfifo(gate, 0o600)
        descriptor = os.open(gate, os.O_RDWR)
        self.addCleanup(os.close, descriptor)
        self.environment["STARTUP_GATE"] = str(gate)
        self.environment["WAIT_FOR_SIGNAL"] = "1"
        with subprocess.Popen(
            self.command("--"),
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as child:
            try:
                self.assertEqual(child.stdout.readline(), "setup\n")
                child.send_signal(signal.SIGTERM)
                os.write(descriptor, b"x")
                output, error = child.communicate()
                # The queued stop may arrive before or after payload exec.
                self.assertIn(
                    child.returncode, (128 + signal.SIGTERM, 23), (output, error)
                )
            finally:
                if child.poll() is None:
                    child.terminate()
                child.wait()


if __name__ == "__main__":
    unittest.main()
