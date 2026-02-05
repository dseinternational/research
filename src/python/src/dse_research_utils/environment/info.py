# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Module for gathering and reporting environment information.
"""

import sys
import pandas as pd
import platform
import psutil


def get_environment_info(as_dataframe=False) -> dict | pd.DataFrame:
    vm = psutil.virtual_memory()
    environment = {
        "platform": platform.platform(),
        "platform_version": platform.version(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": int(round(vm.total / 1024 / 1024 / 1024, 0)),
        "python": sys.version,
    }
    if as_dataframe:
        df = pd.DataFrame(list(environment.items()), columns=["name", "value"])
        return df
    return environment


def report_environment_info():
    environment = get_environment_info()
    print("Environment:")
    for key, value in environment.items():
        print(f"  {key}: {value}")
    