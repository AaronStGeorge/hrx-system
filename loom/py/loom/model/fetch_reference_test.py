# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from unittest.mock import patch

import pytest

from loom.model.fetch_reference import parse_repository
from loom.model.manifest import ManifestError


@pytest.mark.parametrize("provider", ["github", "huggingface"])
@pytest.mark.parametrize(
    ("suffix", "preset", "revision"),
    [
        ("", None, None),
        (":Q4_K_M", "Q4_K_M", None),
        ("@v1.0", None, "v1.0"),
        (":chat@release/v2", "chat", "release/v2"),
        ("@" + "f" * 40, None, "f" * 40),
    ],
)
def test_provider_reference_has_independent_preset_and_revision(
    provider, suffix, preset, revision
):
    reference = parse_repository(provider + ":owner/repo" + suffix)
    assert (
        reference.provider,
        reference.repository,
        reference.preset,
        reference.revision,
    ) == (provider, "owner/repo", preset, revision)
    assert reference.path is None


@pytest.mark.parametrize(
    ("provider", "host"), [("github", "github.com"), ("huggingface", "huggingface.co")]
)
@pytest.mark.parametrize(
    ("suffix", "revision"),
    [("", None), ("/", None), ("/tree/release%2Fv1", "release/v1")],
)
def test_repository_urls(provider, host, suffix, revision):
    reference = parse_repository("https://" + host + "/owner/repo" + suffix)
    assert reference.provider == provider
    assert reference.repository == "owner/repo"
    assert reference.revision == revision
    assert reference.path is None
    assert reference.preset is None


def test_hf_file_urls_keep_their_explicit_path_and_revision():
    reference = parse_repository(
        "https://huggingface.co/owner/repo/blob/release%2Fv1/model/tiny.json?download=true"
    )
    assert (reference.repository, reference.revision, reference.path) == (
        "owner/repo",
        "release/v1",
        "model/tiny.json",
    )


def test_hf_endpoint_applies_to_shorthand_and_repository_urls():
    with patch.dict("os.environ", {"HF_ENDPOINT": "http://127.0.0.1:8000/hub/"}):
        shorthand = parse_repository("huggingface:owner/repo")
        url = parse_repository("http://127.0.0.1:8000/hub/owner/repo")
        assert shorthand == url
        assert shorthand.endpoint == "http://127.0.0.1:8000/hub"


@pytest.mark.parametrize(
    "value",
    [
        "owner/repo",
        "./github:local.json",
        "model.json",
        r"C:\models\model.json",
        "https://example.com/model.json",
        "https://raw.githubusercontent.com/owner/repo/abc/model.json",
    ],
)
def test_bare_paths_and_ordinary_file_urls_remain_documents(value):
    assert parse_repository(value) is None


@pytest.mark.parametrize("provider", ["github", "huggingface"])
@pytest.mark.parametrize(
    "suffix",
    [
        "owner",
        "owner/repo:",
        "owner/repo@",
        "owner/repo:chat@",
        "owner/../repo",
        "../repo",
        "owner/..",
        "owner/repo\n",
        "owner/repo:chat\x00",
    ],
)
def test_malformed_explicit_references_fail_without_becoming_paths(provider, suffix):
    with pytest.raises(ManifestError, match="expected"):
        parse_repository(provider + ":" + suffix)


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/owner/repo?ref=main",
        "https://github.com/owner/..",
        "https://huggingface.co/owner/repo#main",
        "https://huggingface.co/owner/repo?download=1",
        "https://github.com/owner/repo/tree/",
        "https://huggingface.co/owner/repo/tree/",
    ],
)
def test_ambiguous_or_unsafe_repository_urls_are_rejected(value):
    with pytest.raises(ManifestError):
        parse_repository(value)
