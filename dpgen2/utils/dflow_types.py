"""Compatibility aliases for types inspected by dflow at runtime."""

import typing

# dflow validates artifact schema types against ``typing.List`` objects.  The
# equivalent PEP 585 ``list`` form is rejected by current pydflow releases, so
# keep this runtime-only alias until dflow accepts both representations.
DflowList = typing.List  # noqa: UP006
