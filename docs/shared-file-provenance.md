> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words pathlib dataclass shutil fsmonitor worktree umask -->

# Shared file writes and provenance

These helpers replace files and collect provenance facts. Each consuming project retains its manifest fields, JSON encoding, file selection, hash prefixes and rules for accepting a saved result.

The file-write and provenance APIs require 0.14.0 or later; permission options require 0.17.0 or later. They use the Python standard library. See the [installation and upgrade notes](README.md#upgrade-a-consuming-project).

## Write one complete file

`dse_research_utils.storage.files.atomic_write(path, write_temporary)` creates a temporary file beside the destination, calls the supplied writer, then replaces the destination with one rename. Readers can see the old file or the completed new file. Missing parent directories are created. A failure before replacement preserves the destination and removes the temporary file where possible; a cleanup failure is attached to the original exception.

The callback receives an absolute `Path` to an existing empty file. Write to that path and close all file handles before returning. The helper ignores the callback's return value. All destination suffixes are retained, so a writer can infer compression or format from `.csv.gz` or `.npy`.

```python
from pathlib import Path

import pandas as pd

from dse_research_utils.storage.files import atomic_write

frame = pd.DataFrame({"estimate": [0.2, 0.4]})
atomic_write(
    Path("output") / "estimates.csv.gz",
    lambda temporary: frame.to_csv(temporary, index=False),
)
```

Keep JSON encoding in the caller. This adapter accepts the project's existing encoder for values that JSON cannot encode by default. It also specifies key sorting, indentation and a final newline.

```python
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dse_research_utils.storage.files import atomic_write


def write_json_atomic(
    path: str, payload: dict[str, Any], *, encode_extra: Callable[[Any], Any]
) -> None:
    def write_temporary(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True, default=encode_extra)
            destination.write("\n")

    atomic_write(path, write_temporary)
```

Pass the existing encoder as `encode_extra`. Changing it is a separate change to the stored format. The file helper neither sanitises non-finite values nor changes enum, array or dataclass representations. The shared diagnostic writer continues to apply its own strict JSON sanitisation before writing.

On POSIX, the temporary file starts with owner-only read/write permissions. The destination takes the temporary file's final permissions and metadata. A callback can use `shutil.copy2` to retain source-file metadata or set an explicitly required mode. Existing destination permissions are not inherited. If a project previously created ordinary files, choose its sharing mode explicitly when adopting this helper.

An existing destination symlink is replaced as a directory entry; its target is unchanged. A callback must leave a regular file, not a symlink or directory. Created parent directories are retained after failure.

### Permission options in 0.17.0

Version 0.17.0 adds a keyword-only `mode` option. `mode=None` keeps the existing behaviour. An integer such as `mode=0o640` sets those permission bits after the callback. `mode="default"` uses the mode of an ordinary newly created file in the destination directory. An empty, exclusively created probe reads that mode without changing the process-wide `umask`. The probe is removed before replacement. A failed probe or permission change preserves the old destination.

```python
atomic_write(
    Path("output") / "estimates.csv",
    lambda temporary: frame.to_csv(temporary, index=False),
    mode="default",
)
```

The public `default_file_mode(directory)` helper exposes the same probe for callers that need the numeric mode. Its directory must already exist. Explicit `mode` values override permissions set by the callback, including a metadata-preserving copy. These options do not preserve access-control entries or inherit an existing destination's permissions. Consumers must select `v0.17.0` or a later tag before using them; see the [0.17.0 upgrade notes](migrating-to-0.17.md).

This operation does not coordinate competing writers. The last successful replacement wins, and a read-modify-write operation can still lose another writer's update. It does not commit several files as one transaction or guarantee durability after a power loss. The [directory promotion helper](consolidation-migration.md#promote-a-completed-directory) accepts an explicit lock and retains a backup for completed directory trees.

## Collect provenance facts

The functions in `dse_research_utils.metadata.provenance` collect raw facts without creating a manifest or choosing an acceptance policy.

| Function                               | Result                                                                                                                | Caller responsibility                                                     |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `sha256_file(path)`                    | Lowercase SHA-256 of the file's bytes, read in bounded chunks                                                         | Select files, maintain their order, and add any existing digest prefix    |
| `package_versions(distributions)`      | Requested distribution names mapped to installed versions or `None`; a label-to-distribution mapping is also accepted | Choose the package set and retain the manifest's missing-value convention |
| `git_snapshot(repo_root, timeout=5.0)` | A frozen `GitSnapshot` with commit, branch, detached/unborn state, dirty state and untracked-entry state              | Supply the intended directory and decide how to handle unavailable facts  |

```python
from pathlib import Path

from dse_research_utils.metadata.provenance import (
    git_snapshot,
    package_versions,
    sha256_file,
)

repo_root = Path("/path/to/consumer")
snapshot = git_snapshot(repo_root)
versions = package_versions(["dse-research-utils", "numpy"])
digest = sha256_file(repo_root / "input.csv")

# Keep the consumer's established manifest field names and digest prefix.
facts = {
    "commit": snapshot.commit,
    "dirty": snapshot.dirty,
    "git_state": snapshot.state,
    "git_error": snapshot.error,
    "packages": versions,
    "input_hash": f"sha256:{digest}",
}
```

This example field set is illustrative. Do not replace an existing manifest schema with it during migration. File-open and read errors from `sha256_file` propagate. Package lookup reads installed distribution metadata without importing the package; missing, unreadable or malformed metadata produces `None`.

`GitSnapshot.state` distinguishes `available`, `partial` and `unavailable`. A detached HEAD or a branch with no commits yet is a valid, available Git state. In those cases, the absent branch or commit is `None`, with `detached` or `unborn` set explicitly. Failed queries also retain `None` for unknown facts, including `dirty`; they do not report a clean checkout by default. A successful but incomplete response retains only the facts it could establish and records `incomplete_output`.

The Git helper runs one status command with a positive finite timeout. Dirty status includes staged, unstaged, unmerged and untracked changes, and excludes ignored files. It removes inherited repository/index overrides, disables Git's optional index locks and fsmonitor hook, and does not return paths, remote URLs or raw stderr. A directory within the intended worktree is accepted through Git's normal parent search. Bare repositories have no working-tree status and return an unavailable result.

Neither the Git query nor file hashing locks its inputs. Arrange stable inputs separately when a manifest requires an immutable snapshot. Do not treat these raw facts as proof that a fit is current, valid or approved for publication.

## Adoption and validation

Adopt these functions through the existing consumer wrappers so imports and return values remain stable. Keep JSON serialisation and hash inputs byte-for-byte equivalent during the first migration. The tests exercise native JSON, pandas compression, NumPy saving and metadata-preserving copies, as well as interruption, replacement failure, invalid staged files, symlinks and concurrent writers.

Before changing a consumer's provenance implementation, compare its existing manifest against a fixture with known inputs. Preserve its package set, source-file order, separators, digest prefix and missing-value representation. The shared library's display-oriented package-version functions still return `"Not found"` for unavailable packages.

Adding or moving library code can change a consumer's model implementation identity even when numerical results are unchanged. Follow the consumer's rules for resuming or publishing existing fits. Record any accepted identity migration explicitly; do not rewrite historical manifests to imply that old fits were produced by new code.
