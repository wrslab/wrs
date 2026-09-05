"""The one wgpu adapter/device for the process.

Both the viewer and the batch collider allocate on it, and they must share:
the collider binds mesh buffers as storage buffers, and a buffer cannot cross
devices.  Lives in utils rather than in either package so neither has to
import the other.
"""
import wgpu

_adapter = None
_device = None


def get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = wgpu.gpu.request_adapter_sync(
            power_preference='high-performance')
    return _adapter


def get_device():
    global _device
    if _device is None:
        _device = get_adapter().request_device_sync()
    return _device
