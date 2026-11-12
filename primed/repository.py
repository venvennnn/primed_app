"""Session files and the optional cloud gate.

The public demo uses one folder per browser session. Cloud Storage and
Firestore stay off until a project, a private bucket, and access rules exist.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from primed.config import data_root
from primed.workspace import new_workspace


class CloudNotConfigured(RuntimeError):
    pass


class LocalRepository:
    """JSON workspace plus clip files. Visitors do not share a folder."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / "workspace.json"

    def load(self) -> dict:
        if not self.path.exists():
            workspace = new_workspace()
            self.save(workspace)
            return workspace
        return json.loads(self.path.read_text())

    def save(self, workspace: dict) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(workspace))
        temporary.replace(self.path)

    def clip_dir(self, clip_id: str) -> Path:
        if not re.fullmatch(r"clip_[a-f0-9]{12}", clip_id):
            raise ValueError("Invalid clip id.")
        path = self.root / "clips" / clip_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_path(self, key: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid cache key.")
        folder = self.root / "cache"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{key}.json"


def session_dir(session_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise ValueError("Invalid session id.")
    return data_root() / session_id


def open_repository(session_directory: Path) -> LocalRepository:
    if os.environ.get("PRIMED_ENABLE_CLOUD") == "1":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        bucket = os.environ.get("PRIMED_GCS_BUCKET")
        if not project or not bucket:
            raise CloudNotConfigured(
                "Cloud storage is enabled but no project and private bucket are configured. "
                "Unset PRIMED_ENABLE_CLOUD to use the session folder."
            )
        raise CloudNotConfigured(
            "Firestore and Cloud Storage are not enabled in this demo. "
            "Unset PRIMED_ENABLE_CLOUD to keep the session folder."
        )
    return LocalRepository(session_directory)
