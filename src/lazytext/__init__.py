# -*- coding: utf-8 -*-

"""Top-level package for lazytext."""

__author__ = "lazytext"
__email__ = "jay.vala@msn.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.0"


def get_module_version():
    return __version__


from .example import Example  # noqa: F401
