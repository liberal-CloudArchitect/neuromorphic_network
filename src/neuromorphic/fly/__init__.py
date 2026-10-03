"""Exploratory fruit-fly connectome research branch (F track, see CR-013).

Results produced here are exploratory and never count as evidence for the
modular-network Gates.  Connectome structure is a dataset annotation, not a
functional emulation of a fly brain.
"""

from neuromorphic.fly.connectome import Connectome
from neuromorphic.fly.datasets import DatasetFile, DatasetIntegrityError, fetch, verify_file

__all__ = ["Connectome", "DatasetFile", "DatasetIntegrityError", "fetch", "verify_file"]
