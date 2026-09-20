"""AERIS benchmark and real-data bridge components."""

from .channels import diagnose_channels
from .cvb import project_cvb_behavior_table
from .manifest import build_data_manifest

__all__ = [
    "diagnose_channels",
    "project_cvb_behavior_table",
    "build_data_manifest",
]
