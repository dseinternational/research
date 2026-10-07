> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:ignore fonttools tqdm wrapt -->

# Upgrade to 0.15.1

Version 0.15.1 refreshes the repository lockfile. It changes no library API, dependency minimum or extra. A consuming project does not inherit this lockfile, so changing the library tag alone does not require these package updates.

## Lockfile changes in this release

| Package      | Previous lock | Updated lock |
| ------------ | ------------- | ------------ |
| build        | 1.6.0         | 1.6.1        |
| fonttools    | 4.64.0        | 4.65.0       |
| pure-eval    | 0.2.3         | 0.2.4        |
| ruff         | 0.16.6        | 0.16.7       |
| scikit-learn | 1.9.0         | 1.9.1        |
| tqdm         | 4.70.0        | 4.70.1       |
| uv           | 0.12.11       | 0.12.13      |
| wrapt        | 2.4.0         | 2.4.1        |

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.15.1` as the target tag. To adopt a particular lockfile update, request it explicitly in the consuming project, for example:

```bash
uv lock --upgrade-package dse-research-utils --upgrade-package scikit-learn
```

Review the versions actually resolved. Do not copy this repository's lockfile or add its build tooling as consumer dependencies. Apply the [0.15.0 notes](migrating-to-0.15.md) and earlier requirements if upgrading from an older version.
