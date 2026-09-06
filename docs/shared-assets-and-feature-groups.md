> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words srcset href src stylesheet srcsets preload iframe urllib symlink symlinks dcor dissimilarities dissimilarity linkage fcluster ndarray dtype float64 SciPy ELPD USBC VG LRP cutoff pathlib Nonempty endswith nonnegative int32 int64 DataFrame -->

# Shared report assets and feature groups

This part of PR #101 adds direct HTML asset inspection and feature grouping from an existing distance matrix. Both APIs are unreleased. They use dependencies already available in the library. They do not upload a report, select a scientific result for publication or choose a study's feature cutoff.

## Check the local report before checking an upload

`report.assets.inspect_local_assets` parses HTML rather than searching for quoted strings. It records direct resource references, such as images, scripts and stylesheets, and distinguishes them from ordinary navigation. Each reference records its source page, element, attribute, target, required status and local file state. Missing files remain visible.

The inspection root is the directory that corresponds to the upload prefix. It defaults to the entry page's directory. A reference such as `../figures/plot.png` can resolve inside an explicitly supplied root, but paths and symlinks that escape the root are invalid. The scanner preserves an internal symlink's referenced filename for matching against the uploader's inventory.

`inspection.required_paths` contains the inspected pages and required local references as sorted, raw paths relative to that root. It includes missing files. Call `check_uploaded_assets` before treating that list as a complete bundle. Passing the list back as the prospective upload inventory checks local failures without making requests.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from dse_research_utils.report.assets import (
    AssetFailure,
    check_uploaded_assets,
    inspect_local_assets,
    verify_published_assets,
)

with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "figure.png").write_bytes(b"example")
    (root / "index.html").write_text(
        "<img src='figure.png?v=2'><img src='missing.png'>", encoding="utf-8"
    )
    inspection = inspect_local_assets(root / "index.html")
    assert inspection.required_paths == ("figure.png", "index.html", "missing.png")
    failures = check_uploaded_assets(inspection, ["index.html", "figure.png"])
    assert failures == (AssetFailure("missing.png", "missing"),)
```

The upload check accepts raw filenames from `BlobUploadResult.relative_paths`, including Windows separators. It does not infer filenames from uploaded URLs. The entry page must be present at its inspected path; a nested `assets/index.html` cannot replace an entry at `index.html`. An inventory entry cannot hide a missing or unsupported local resource. Inspect and upload stable files because inspection is a snapshot.

Ordinary links remain available in `inspection.references`. Set `include_navigation=True` to require local downloads and page links too. VG's current publication checks treat existing local links to CSVs, images and HTML as required, so its initial adapter should select this option. External URLs, data URLs and same-page fragments are recorded but not requested. A successful local check says nothing about those references.

Set `follow_pages=True` to inspect referenced local `.html` and `.htm` pages recursively. Those pages become required even when ordinary navigation is otherwise optional. Invalid HTML paths that escape the root remain failures and are never opened. Repeated references and ordinary page cycles do not cause repeated scans. A new URL alias that resolves to an ancestor page produces an unsupported `cyclic_page_alias` finding. This stops symlink cycles without claiming to have checked resources under the new URL. Without recursive inspection, only the supplied page's direct references are inspected. Neither mode follows CSS imports, CSS image/font URLs, JavaScript imports or dynamically generated resources.

Nonempty `base href` and responsive `srcset` attributes currently produce explicit unsupported failures. Even `base href="./"` changes how query-only URLs resolve. These constructs can change which file a browser loads, so ignoring them could certify an incomplete bundle. Root-relative URLs such as `/figure.png` are invalid for local bundle matching because the browser resolves them at the website origin, rather than under the report's upload prefix. Consumer adapters must handle these findings before reporting success.

The scanner decodes HTML entities, separates URL path/query/fragment, then decodes the path once. Plus signs remain plus signs. URL path separators must be literal `/` characters. Encoded separators, repeated slashes and directory-shaped URLs ending in `/`, `/.` or `/..` are invalid for file matching because filesystem normalization could otherwise change the requested URL. For a filename containing the literal text `%20`, the HTML must contain `%2520`. The current VG test that links the literal filename `50%20.csv` as `href="50%20.csv"` is not a valid compatibility target: a browser asks for `50 .csv`. The shared inspection exposes that missing file. Tests retain both the mismatch and the correctly encoded form.

## Check HTTP availability separately

`verify_published_assets` accepts a directory URL and raw relative filenames. It preserves an already encoded upload prefix and encodes each filename once. It requests each distinct path once, in the supplied order. To check the entry page first, put it first explicitly.

```python
requests = []


def fetch_status(url: str, timeout: float) -> int:
    requests.append(url)
    return 404 if url.endswith("missing.png") else 200


failures = verify_published_assets(
    "https://example.org/report%20one/",
    ["index.html", "tables/a+b.csv", "missing.png"],
    fetch_status=fetch_status,
)
assert requests[1] == "https://example.org/report%20one/tables/a%2Bb.csv"
assert failures == (AssetFailure("missing.png", "http_status", 404),)
```

The default transport uses a bounded HTTP GET, closes the response without downloading its body and rejects redirects. A login page reached through a redirect cannot supply a successful status for the requested asset. Callers can inject a transport that returns an integer status and follows the same redirect rule. Failures record a relative path, an HTTP status or exception type; raw exception messages are excluded.

Run the local/upload inventory check first, then make HTTP requests only after it passes. A 200 response establishes HTTP availability, not correct bytes, content type, complete page rendering or permission to publish the scientific results. VG and LRP retain their report visibility rules and final success messages.

## Reuse a dissimilarity matrix for feature grouping

`ml.feature_groups.linkage_from_dissimilarity` builds a hierarchical clustering tree from a precomputed square matrix. It requires finite, nonnegative values, exact symmetry and a zero diagonal. It does not repair inputs, recompute dependence, impute values or change feature order. Labels attached to a DataFrame do not establish order for this numerical API; callers must retain the names corresponding to matrix rows and columns.

The supported methods are `single`, `complete`, `average` and `weighted`. Ward, centroid and median linkage need Euclidean distances, which an arbitrary dissimilarity matrix does not establish, so this API excludes them. Matrix values are converted to float64 before clustering. Zero or one feature returns an empty tree with shape `(0, 4)`.

`feature_groups_from_linkage` validates the tree, then cuts it at the caller's threshold. It returns a dictionary keyed by SciPy's numeric cluster labels. Groups appear in the order of their first input feature, and members retain input order. An empty sequence of feature names produces an empty dictionary; a singleton produces `{1: [name]}`. A merge exactly at the threshold is included.

```python
from dse_research_utils.ml.feature_groups import (
    feature_groups_from_linkage,
    linkage_from_dissimilarity,
)

names = ["reading", "language", "memory"]
matrix = [[0.0, 0.2, 0.8], [0.2, 0.0, 0.7], [0.8, 0.7, 0.0]]
tree = linkage_from_dissimilarity(matrix, method="average")
groups = feature_groups_from_linkage(names, tree, threshold=0.3)
assert list(groups.values()) == [["reading", "language"], ["memory"]]

# A consumer that assigns IDs in first-feature order can retain that convention.
consumer_groups = {
    f"cluster_{number:02d}": members
    for number, members in enumerate(groups.values(), start=1)
}
assert consumer_groups["cluster_01"] == ["reading", "language"]
```

USBC can retain its `cluster_01` identifiers through that adapter. LRP can keep the numeric keys for joins to its feature and cluster-importance tables. Its adapter should preserve the existing int32 dtype for the in-memory `cluster_id` column; constructing a DataFrame directly from Python integer keys otherwise infers int64. Numeric labels are local to a tree and cut; they are not stable identities across changed feature order, distances or thresholds. A tree stores positions and cannot detect names supplied in the wrong order.

Average linkage uses the average distance between merging clusters. Its threshold does not bound every pairwise distance inside a group or guarantee a minimum pairwise correlation. For distances 0.1, 0.2 and 0.8 among three features, average linkage merges all three at 0.5 even though one pair is 0.8 apart. Tests retain this distinction.

The existing `distance_corr_dissimilarity_linkage` helper delegates tree construction to the new function. Its normal average-linkage values and three-part return schema are preserved. Zero and singleton feature sets now return an empty tree. Direct use of `ml.feature_groups` does not import the optional distance-correlation package.

## Validate consumer adapters

Asset fixtures check missing files, upload omissions, raw and encoded names, recursive page paths, HTTP failures and the intentional changes from VG's scanner. Grouping fixtures retain member order, consumer identifiers and joins to importance tables. Tests use synthetic inputs and injected HTTP responses; they do not upload reports or access live publications.

Consumers should compare their own inventories and grouped output tables before migration. Keep their input cleanup, imputation, cutoff, feature roles, publication checks and saved-fit identity policies in their adapters. The [statistical array guide](shared-predictive-arrays.md) and [file/provenance guide](shared-file-provenance.md) describe the earlier additions in this PR.
