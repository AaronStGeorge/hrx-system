# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Model and catalog discovery at one publication snapshot per acquisition."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin

from .catalog import ModelCatalog
from .fetch_reference import RepositoryReference, parse_repository
from .fetch_source import Document, FetchError, SourceNotFound, document_location
from .fetch_storage import ContentCache, atomic_write
from .manifest import ManifestError, ModelCard, parse_json


@dataclass(frozen=True)
class PublicationSnapshot:
    # Authored local document, file URL, or explicit repository reference.
    source: str
    # Entry document at its immutable repository coordinate, when applicable.
    document: Document
    # Validated root card or catalog, without reading unselected model cards.
    content: ModelCard | ModelCatalog
    # Selected Git commit, absent for local documents and ordinary HTTP cards.
    revision: str | None
    # Optional repository shorthand selection, with exactly --preset semantics.
    preset: str | None = None

    def summary(self) -> dict:
        return {
            "source": self.source,
            "location": self.document.location,
            "revision": self.revision,
        }

    def read_model(
        self, identity: str, cache: ContentCache, *, offline: bool
    ) -> tuple[Document, ModelCard]:
        if isinstance(self.content, ModelCard):
            document, card = self.document, self.content
        else:
            path = self.content.card_path(identity)
            location = (
                urljoin(self.document.location, quote(path, safe="/"))
                if self.document.remote
                else str(Path(self.document.location).parent / path)
            )
            document = cache.read_document(location, offline=offline)
            card = ModelCard.parse(document.text)
        if card.document["model"]["id"] != identity:
            raise ManifestError(
                "selected model card identity differs from the requested model"
            )
        return document, card


def installed_catalog() -> str:
    """Reads only the active installed SDK's explicitly authored association."""
    try:
        distribution = importlib.metadata.distribution("hrx-loom")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    # A source checkout on PYTHONPATH must not inherit another installed SDK's
    # release association merely because its distribution metadata is visible.
    if distribution is not None and Path(__file__).resolve().is_relative_to(
        Path(distribution.locate_file("loom/model")).resolve()
    ):
        text = distribution.read_text("loom.json")
        if text is not None:
            source = json.loads(text)["catalog"]
            if source:
                return source
    raise FetchError(
        "this SDK has no associated model catalog; use --catalog=LOCAL_JSON "
        "or --catalog=PROVIDER:OWNER/REPOSITORY[@REVISION]"
    )


def _commit(cache: ContentCache, reference: RepositoryReference) -> str:
    root = "https://api.github.com/repos/" + reference.repository + "/commits"
    location = (
        root + "/" + quote(reference.revision, safe="") + "?per_page=1"
        if reference.revision is not None
        else root + "?per_page=1"
    )
    document = parse_json(cache.read_document(location, offline=False).text)
    # Listing without sha selects the actual default branch, not a branch named
    # main. Get-commit resolves branches and annotated tags at the provider.
    if reference.revision is None:
        if not isinstance(document, list) or len(document) != 1:
            raise FetchError("GitHub returned no unique default-branch commit")
        document = document[0]
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("sha"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", document["sha"])
        or (
            reference.revision is not None
            and re.fullmatch(r"[0-9a-f]{40}", reference.revision)
            and document["sha"] != reference.revision
        )
    ):
        raise FetchError("GitHub returned an invalid immutable revision")
    return document["sha"]


def load_publication(
    source: str, cache: ContentCache, *, offline: bool, catalog: bool = False
) -> PublicationSnapshot:
    reference = parse_repository(source)
    if reference is None:
        if catalog and "://" in source:
            raise ManifestError(
                "catalog discovery requires a local JSON, a Hugging Face file URL, or "
                "an explicit repository reference; direct model URLs do not require "
                "a catalog"
            )
        location = document_location(source)
        document = cache.read_document(location, offline=offline)
        content = (ModelCatalog if catalog else ModelCard).parse(document.text)
        return PublicationSnapshot(location, document, content, None)

    path = reference.path or ("catalog.json" if catalog else None)
    key = json.dumps(
        [
            reference.provider,
            reference.endpoint,
            reference.repository,
            reference.revision,
            path,
            catalog,
        ]
    )
    pointer = cache.root / "publication" / hashlib.sha256(key.encode()).hexdigest()
    revision = reference.revision
    pinned = revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision)
    if offline and (not pinned or path is None):
        try:
            with pointer.open("rb") as stream:
                record = parse_json(stream.read(257).decode("utf-8"))
        except FileNotFoundError as error:
            raise FetchError(
                "--offline has no cached snapshot for this publication"
            ) from error
        except (UnicodeError, ManifestError) as error:
            raise FetchError("cached publication snapshot is invalid") from error
        if (
            not isinstance(record, dict)
            or set(record) != {"revision", "catalog"}
            or not isinstance(record["revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", record["revision"])
            or not isinstance(record["catalog"], bool)
            or (pinned and record["revision"] != revision)
            or (path is not None and record["catalog"] != catalog)
        ):
            raise FetchError("cached publication snapshot is invalid")
        revision, catalog = record["revision"], record["catalog"]
        path = path or ("catalog.json" if catalog else "model.json")
    elif not offline:
        if reference.provider == "huggingface":
            revision = cache.huggingface.resolve_revision(
                {
                    "repository": reference.repository,
                    "revision": reference.revision or "main",
                },
                offline=False,
            )
        elif not pinned or path is None:
            # Prove the repository/revision exists before interpreting a raw
            # file's 404 as permission to discover another root entry point.
            revision = _commit(cache, reference)

    def read(path):
        location = (
            cache.huggingface.file_url(
                {
                    "repository": reference.repository,
                    "revision": revision,
                    "path": path,
                },
                endpoint=reference.endpoint,
            )
            if reference.provider == "huggingface"
            else f"https://raw.githubusercontent.com/{reference.repository}/{revision}/"
            + quote(path, safe="/")
        )
        return cache.read_document(location, offline=offline)

    if path is None:
        try:
            document = read("model.json")
        except SourceNotFound:
            try:
                document = read("catalog.json")
            except SourceNotFound as error:
                raise FetchError(
                    "repository has neither a root model.json nor catalog.json; "
                    "specify a model-card file URL"
                ) from error
            catalog = True
    else:
        document = read(path)
    content = (ModelCatalog if catalog else ModelCard).parse(document.text)
    if not offline:
        atomic_write(
            pointer, json.dumps({"revision": revision, "catalog": catalog}).encode()
        )
    return PublicationSnapshot(source, document, content, revision, reference.preset)
