from pathlib import Path
from typing import Any, TypedDict

class VersionDict(TypedDict):
    """Type definition for the version dictionary returned by get_version_dict."""

    version: str
    full: str
    branch: str
    date: str
    dirty: bool
    error: str
    full_revisionid: str
    pieces: dict[str, Any]

def get_version_dict_wo_exec(
    cwd: str | Path,
    style: str = "pep440",
    tag_prefix: str = "v",
    parentdir_prefix: str | None = None,
    versionfile_source: str | None = None,
    verbose: bool = False,
    root: str | Path | None = None,
) -> VersionDict: ...
