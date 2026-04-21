# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from importlib.metadata import version

import pandas as pd

from dse_research_utils.console.sections import section_header
from dse_research_utils.console.tables import print_key_value_table


def get_package_version(pkg_name) -> str:
    try:
        return version(pkg_name)
    except Exception:
        return "Not found"


def get_package_versions(pkg_names: list[str], as_dataframe=False) -> dict | pd.DataFrame:
    versions = {}
    for pkg in pkg_names:
        versions[pkg] = get_package_version(pkg)
    if as_dataframe:
        return pd.DataFrame.from_dict(versions, orient="index", columns=["version"])
    return versions


def report_package_versions(pkg_names):
    versions = get_package_versions(pkg_names)
    section_header("Package Versions")
    print_key_value_table(versions, key_header="Package", value_header="Version")
