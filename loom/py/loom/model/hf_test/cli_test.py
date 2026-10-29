# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import unittest
from contextlib import contextmanager

import pytest

from loom.model.hf_test.hub import running_hub


def _card():
    return {
        "schema_version": 1,
        "model": {"id": "fixture/access", "name": "Access fixture"},
        "programs": {"p": {"inputs": ["code.loom"], "libraries": []}},
        "services": {"text": {"kind": "text", "version": 1}},
        "variants": {"v": {"services": {"text": {"programs": ["p"]}}}},
    }


def _arguments(hub, *arguments):
    return [
        sys.executable,
        "-m",
        "loom.model.fetch_main",
        hub.endpoint + "/fixture/model/blob/main/model.json",
        "--cache=" + str(hub.cache_dir.parent / "loom-cache"),
        *arguments,
    ]


def _publish(hub):
    hub.credentials.write_text("fixture-first")
    hub.files["model.json"] = json.dumps(_card()).encode()


def test_sdk_cache_write_failure_reports_context_without_traceback(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        hub.approved = True
        # A regular file cannot contain the SDK cache, on either Windows or POSIX.
        blocked_directory = tmp_path / "not-a-directory"
        blocked_directory.write_bytes(b"occupied")
        cache_directory = blocked_directory / "hf-cache"
        result = subprocess.run(
            _arguments(hub, "--list", "--json", "--no-input"),
            env={**hub.environment(), "HF_HUB_CACHE": str(cache_directory)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stderr
        document = json.loads(result.stdout)
        assert document["kind"] == "error"
        assert "Hugging Face download failed" in document["message"]
        assert "fixture/model/model.json" in document["message"]
        assert str(cache_directory) in document["message"]
        assert "Traceback" not in result.stderr
        assert "fixture-first" not in result.stdout + result.stderr


def test_redirected_progress_arrives_before_hf_download_completes(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        hub.approved = True
        hub.held_file = "code.loom"
        output = tmp_path / "output"
        with subprocess.Popen(
            _arguments(
                hub, "--no-lock", "--json", "--no-input", "--output=" + str(output)
            ),
            env=hub.environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                observed = []
                for line in process.stderr:
                    observed.append(line)
                    if line.startswith("loom-fetch  Download ") and "code.loom" in line:
                        assert any("loom-fetch  Output" in prior for prior in observed)
                        assert not hub.release_body.is_set()
                        assert not (output / "model.json").exists()
                        hub.release_body.set()
                        break
                else:
                    raise AssertionError(
                        "no live download progress: " + "".join(observed)
                    )
                result, error = process.communicate()
                assert process.returncode == 0, error
                assert json.loads(result)["card"] == str(output / "model.json")
                assert "Complete" in error
                assert "fixture-first" not in "".join(observed) + result + error
                assert "\x1b" not in "".join(observed) + error
            finally:
                hub.release_body.set()


@pytest.mark.parametrize("quiet", [False, True])
def test_headless_gate_is_one_structured_action_without_waiting(tmp_path, quiet):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        result = subprocess.run(
            _arguments(
                hub, "--list", "--json", "--no-input", *(["--quiet"] if quiet else [])
            ),
            env=hub.environment(),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stderr
        action = json.loads(result.stdout)
        assert action["kind"] == "action_required"
        assert action["reason"] == "gate"
        assert action["account"] == "first"
        assert action["url"] == hub.endpoint + "/fixture/model"
        assert "fixture-first" not in result.stdout + result.stderr
        assert "Waiting" not in result.stderr
        assert hub.requests == [
            ("GET", "/api/models/fixture/model/auth-check"),
            ("GET", "/api/whoami-v2"),
        ]


def test_lock_gate_retains_structured_action(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        local = tmp_path / "model.json"
        local.write_text(json.dumps(_card()))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "loom.model.fetch_main",
                str(local),
                "--lockfile="
                + hub.endpoint
                + f"/fixture/model/resolve/{hub.revision}/model.json.lock",
                "--json",
                "--no-input",
                "--output=" + str(tmp_path / "output"),
            ],
            env=hub.environment(),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stderr
        assert json.loads(result.stdout)["kind"] == "action_required"
        assert not (tmp_path / "output").exists()


def test_explicit_headless_wait_stays_before_payload(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        with subprocess.Popen(
            _arguments(hub, "--list", "--json", "--no-input", "--wait-for-access"),
            env=hub.environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                for line in process.stderr:
                    if "Waiting for access" in line:
                        break
                else:
                    raise AssertionError("CLI exited before its access wait")
                assert process.poll() is None
                assert not any(path.endswith("code.loom") for _, path in hub.requests)
            finally:
                if process.poll() is None:
                    process.kill()


def test_offline_does_not_enter_an_explicit_access_wait(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        result = subprocess.run(
            _arguments(
                hub, "--list", "--json", "--no-input", "--wait-for-access", "--offline"
            ),
            env=hub.environment(),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stderr
        assert json.loads(result.stdout)["kind"] == "error"
        assert "--offline" in result.stderr
        assert "Waiting" not in result.stderr
        assert hub.requests == []


def test_interrupt_stops_a_held_transfer_before_publication(tmp_path):
    if os.name == "nt":
        raise unittest.SkipTest("console control events require a Windows console")
    with running_hub(tmp_path) as hub:
        _publish(hub)
        hub.approved = True
        hub.held_file = "code.loom"
        output = tmp_path / "output"
        with subprocess.Popen(
            _arguments(
                hub, "--no-lock", "--json", "--no-input", "--output=" + str(output)
            ),
            env=hub.environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:

            def exited():
                process.wait()
                hub.body_started.set()

            watcher = threading.Thread(target=exited)
            watcher.start()
            try:
                hub.body_started.wait()
                assert process.poll() is None, process.communicate()
                process.send_signal(signal.SIGINT)
                result, error = process.communicate()
                assert process.returncode == 130, error
                assert json.loads(result) == {"kind": "cancelled"}
                assert not hub.release_body.is_set()
                assert not (output / "model.json").exists()
                assert not (output / "code.loom").exists()
            finally:
                hub.release_body.set()
                if process.poll() is None:
                    process.kill()
                watcher.join()


@contextmanager
def _terminal(hub, *arguments):
    if os.name == "nt":
        raise unittest.SkipTest("this case exercises POSIX PTY mode restoration")
    import pty
    import termios

    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    try:
        with subprocess.Popen(
            _arguments(hub, *arguments),
            env=hub.environment(),
            stdin=slave,
            stderr=slave,
            stdout=subprocess.PIPE,
        ) as process:
            try:
                yield process, master, slave
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


def _await_controls(process, master):
    content = b""
    with selectors.DefaultSelector() as selector:
        selector.register(master, selectors.EVENT_READ, "terminal")
        selector.register(process.stdout, selectors.EVENT_READ, "result")
        while b"[q / Ctrl-C] Cancel" not in content:
            for key, _ in selector.select():
                data = os.read(key.fd, 4096)
                if key.data == "result" or not data:
                    raise AssertionError(
                        f"CLI exited before controls: {content!r} {data!r}"
                    )
                content += data
    return content.decode()


def test_terminal_approval_refreshes_login_and_resumes_exact_selection(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        output = tmp_path / "output"
        with _terminal(hub, "--no-lock", "--output=" + str(output)) as (
            process,
            master,
            _,
        ):
            panel = _await_controls(process, master)
            assert "Effective account: first" in panel
            assert hub.endpoint + "/fixture/model" in panel
            assert not output.exists()
            hub.credentials.write_text("fixture-second")
            hub.approved = True
            os.write(master, b"\r")
            result, _ = process.communicate()
            assert process.returncode == 0, result
            assert (output / "code.loom").read_bytes() == b"code"
            assert (output / "model.json").is_file()
            assert not (output / "model.json.lock").exists()


@pytest.mark.parametrize("cancel", ["q", "interrupt", "eof"])
def test_terminal_cancel_restores_input_and_does_not_publish(tmp_path, cancel):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        output = tmp_path / "output"
        with _terminal(hub, "--no-lock", "--output=" + str(output)) as (
            process,
            master,
            _,
        ):
            _await_controls(process, master)
            if cancel == "interrupt":
                process.send_signal(signal.SIGINT)
            else:
                os.write(master, b"q" if cancel == "q" else b"\x04")
            process.communicate()
            assert process.returncode == 130
            assert not output.exists()
            assert not any("/resolve/" in path for _, path in hub.requests)
