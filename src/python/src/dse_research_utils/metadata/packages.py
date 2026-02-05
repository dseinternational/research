# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
from importlib.metadata import version


def get_package_version(pkg_name) -> str:
    try:
        return version(pkg_name)
    except Exception:
        return "Not found"


def get_package_versions(pkg_names : list[str], as_dataframe=False) -> dict | pd.DataFrame:
    versions = {}
    for pkg in pkg_names:
        versions[pkg] = get_package_version(pkg)
    if as_dataframe:
        return pd.DataFrame.from_dict(versions, orient="index", columns=["version"])
    return versions

def report_package_versions(pkg_names):
    versions = get_package_versions(pkg_names)
    print("Package Versions:")
    for pkg, ver in versions.items():
        print(f"  {pkg}: {ver}")
