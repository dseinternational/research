> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words bfmi pathlib umask -->

# Upgrade to 0.17.0

Version 0.17.0 adds public diagnostic reductions and optional file-permission controls. It also fixes nullable missing-value handling in existing diagnostic reductions.

Python 3.14, the base dependency requirements and all extras are unchanged from `v0.16.2`. Existing calls to `atomic_write` remain valid, and the private `_bfmi_per_chain` compatibility wrapper remains available.

## Read sampling signals without choosing a publication rule

Import the public helpers from their module.

```python
from dse_research_utils.statistics.diagnostics import (
    bfmi_per_chain,
    diagnostic_extrema,
)
```

`bfmi_per_chain(trace)` returns energy diagnostics in chain-coordinate order. It reads the named `chain` and `draw` dimensions, so a transposed stored array has the same result. A draw-only energy array represents one chain. Missing or unreadable energy returns `None`; constant energy produces NaN for that chain. Additional dimensions are not flattened. BFMI measures how sampling chains move through energy levels. This function calculates the values without deciding whether they pass.

`diagnostic_extrema(summary)` accepts an existing, unrounded table with `r_hat`, `ess_bulk` and `ess_tail` columns. It returns the largest R-hat, the smallest bulk or tail effective sample size, and the row names with at least one unavailable diagnostic. R-hat checks agreement across chains. Effective sample size describes sampling precision in terms of an equivalent number of independent draws. Callers select variables, remove any permitted constant rows and rename columns before passing the table.

Absent columns, nullable missing values and values that cannot be converted to numbers become NaN. Extrema skip NaN and retain infinities. Entirely unavailable extrema are NaN, and an empty table raises `ValueError`. A caller must consider the unavailable row names separately when it applies a pass/fail rule. Rounding before reduction can hide a result just outside a threshold.

The nullable-value fix also applies to the existing diagnostic writer and sampling-quality helpers. Rows with missing diagnostics remain marked as unavailable. Keep each project's thresholds, exceptions and publication decisions explicit when adopting these functions.

## Choose file permissions explicitly

`storage.files.atomic_write` accepts a keyword-only `mode` argument. Omitting it, or passing `None`, retains the callback's final permissions. On POSIX, the temporary file starts with owner-only read/write permissions. An integer mode sets permission bits after the callback. `mode="default"` uses the mode of an ordinary new file in the destination directory.

```python
from pathlib import Path

from dse_research_utils.storage.files import atomic_write

atomic_write(
    Path("output") / "summary.json",
    lambda temporary: temporary.write_text('{"complete": true}\n', encoding="utf-8"),
    mode="default",
)
```

The public `default_file_mode(directory)` helper exposes the same mode probe. The directory must already exist. The probe creates and removes an empty file without changing the process-wide `umask`. A failed probe or permission change preserves the old destination. An explicit mode overrides permissions set by the callback, including copied metadata. It does not preserve access-control entries or inherit the old destination's mode. Windows applies its own permission semantics. See the [file-write guide](shared-file-provenance.md#permission-options-in-0170) for the full contract.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.17.0` as the target tag. Route local calculations and permission probes through the existing project adapters.

Test missing diagnostics, transposed energy arrays, constant chains, threshold boundaries and required new-file and replacement permissions. Render representative reports when their reporting code changes. Keep model specifications, variable selection, thresholds, exceptions and publication decisions in the consuming project.
