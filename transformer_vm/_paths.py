"""Canonical paths for package-internal resources."""

import os

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_PKG_ROOT, "data")
EXAMPLES_DIR = os.path.join(_PKG_ROOT, "examples")
MANIFEST = os.path.join(EXAMPLES_DIR, "manifest.yaml")
