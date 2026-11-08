"""Runnable examples shipped with the toolkit.

This file exists so the documented factory reference
``examples.custom_model.model:create_model`` resolves deterministically when a run is
started from the repository root. Without it, ``examples`` would be a namespace package
and any third-party distribution that ships a top-level ``examples`` package would win
the import instead (one is installed in the reference environment).

Consequence, stated rather than hidden: while the working directory is the repository
root, this package shadows such a distribution. Nothing in this project imports it, and
the alternative -- a name that depends on what else happens to be installed -- would make
the documented factory reference environment-dependent.
"""
