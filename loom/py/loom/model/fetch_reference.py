# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit repository coordinates, independent of model and preset semantics."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from .fetch_hf import hf_url_path, hf_url_source
from .manifest import ManifestError, _url


@dataclass(frozen=True)
class RepositoryReference:
    # Explicit provider: github or huggingface, never inferred from a bare path.
    provider: str
    # Provider repository identity, independent of the contained model IDs.
    repository: str
    # Authored branch, tag, or commit; None selects the provider's default branch.
    revision: str | None = None
    # Explicit file within a publication; None requests root entry-point discovery.
    path: str | None = None
    # Optional shorthand for --preset, independent of the repository revision.
    preset: str | None = None
    # Hub endpoint, including an explicitly configured HF_ENDPOINT when applicable.
    endpoint: str | None = None


def parse_repository(value: str) -> RepositoryReference | None:
    """Recognizes explicit providers and their URLs; leaves local paths untouched."""
    for provider in ("huggingface", "github"):
        prefix = provider + ":"
        if not value.startswith(prefix):
            continue
        coordinate, at, revision = value[len(prefix) :].partition("@")
        repository, colon, preset = coordinate.partition(":")
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            or any(part in (".", "..") for part in repository.split("/"))
            or (at and not revision)
            or (colon and not preset)
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ManifestError(
                f"expected {provider}:OWNER/REPOSITORY[:PRESET][@REVISION]"
            )
        endpoint = None
        if provider == "huggingface":
            endpoint = os.getenv("HF_ENDPOINT") or "https://huggingface.co"
            _url(endpoint, "Hugging Face endpoint")
            if urlsplit(endpoint).query:
                raise ManifestError("Hugging Face endpoint cannot contain a query")
            endpoint = endpoint.rstrip("/")
        return RepositoryReference(
            provider,
            repository,
            revision if at else None,
            preset=preset if colon else None,
            endpoint=endpoint,
        )
    if "://" not in value:
        return None
    file = hf_url_source(value)
    if file is not None:
        source, endpoint = file
        return RepositoryReference(
            "huggingface",
            source["repository"],
            source["revision"],
            source["path"],
            endpoint=endpoint,
        )
    hub = hf_url_path(value)
    parsed = urlsplit(value)
    if hub is not None:
        relative, endpoint = hub
        provider = "huggingface"
    elif parsed.scheme == "https" and parsed.netloc == "github.com":
        _url(value, "GitHub repository URL")
        relative, endpoint = parsed.path.removeprefix("/"), None
        provider = "github"
    else:
        return None
    match = re.fullmatch(
        r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:/tree/(.+?))?/?", relative
    )
    if (
        match is None
        or parsed.query
        or any(part in (".", "..") for part in match[1].split("/"))
    ):
        raise ManifestError(
            f"expected a {provider} repository URL, optionally with /tree/REVISION; "
            "use a raw file URL for a model card"
        )
    return RepositoryReference(
        provider, match[1], unquote(match[2]) if match[2] else None, endpoint=endpoint
    )
