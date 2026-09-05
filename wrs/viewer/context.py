"""Viewer-facing alias for the process-wide wgpu device.

See utils.gpu_device -- the viewer and the batch collider share one device on
purpose, so the collider can bind buffers the viewer allocated.
"""
from wrs.utils.gpu_device import get_adapter, get_device

__all__ = ['get_adapter', 'get_device']
