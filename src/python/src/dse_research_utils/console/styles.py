# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Style token constants for the ``dse_research_utils.console`` subpackage.

Callers that need ad-hoc styling should import these constants rather than
hard-coding colour or weight strings, so that palette tweaks land in one
place. Every rendering helper exposes ``style=`` / ``border_style=`` kwargs
that default to one of these tokens.
"""

STYLE_BANNER_BORDER: str = "green"
STYLE_BANNER_TITLE: str = "bold green"
STYLE_BANNER_SUBTITLE: str = "dim"

STYLE_SECTION: str = "bold green"
STYLE_SUBSECTION: str = "bold yellow"

STYLE_TABLE_TITLE: str = "bold"
STYLE_TABLE_HEADER: str = "bold cyan"
STYLE_KEY_COLUMN: str = "cyan"
STYLE_VALUE_COLUMN: str = "white"

STYLE_MUTED: str = "dim"
STYLE_CHECK: str = "dim green"
