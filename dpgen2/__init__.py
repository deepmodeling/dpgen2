try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import (
        __version__,
    )


__all__ = ["__version__"]
