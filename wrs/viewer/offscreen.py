"""Headless rendering: one frame straight into a numpy RGBA array.

No window and no swapchain are involved, so this works in a script with no
display at all.
"""
import numpy as np
import wgpu
import wrs.viewer.context as wvc
import wrs.viewer.render as wvr

# sRGB, matching what World gets from get_preferred_format(): the hardware
# gamma-encodes on write, so an offscreen frame and the on-screen one come out
# identical. A non-sRGB target here would silently render 1.4x darker.
COLOR_FORMAT = wgpu.TextureFormat.rgba8unorm_srgb


class OffscreenTarget:

    def __init__(self, width, height, sample_count=4):
        self.width = width
        self.height = height
        self.sample_count = sample_count
        self.device = wvc.get_device()
        size = (width, height, 1)
        tu = wgpu.TextureUsage
        self.resolve_tex = self.device.create_texture(
            size=size, format=COLOR_FORMAT,
            usage=tu.RENDER_ATTACHMENT | tu.COPY_SRC)
        self.msaa_tex = self.device.create_texture(
            size=size, format=COLOR_FORMAT, sample_count=sample_count,
            usage=tu.RENDER_ATTACHMENT)
        self.depth_tex = self.device.create_texture(
            size=size, format=wvr.DEPTH_FORMAT, sample_count=sample_count,
            usage=tu.RENDER_ATTACHMENT)

    def read_rgba(self):
        row_bytes = self.width * 4
        data = self.device.queue.read_texture(
            {'texture': self.resolve_tex, 'mip_level': 0, 'origin': (0, 0, 0)},
            {'offset': 0, 'bytes_per_row': row_bytes,
             'rows_per_image': self.height},
            (self.width, self.height, 1))
        return np.frombuffer(data, np.uint8).reshape(self.height, self.width, 4)


def render_to_array(scene, camera, width=960, height=720,
                    sample_count=4, bg=(1.0, 1.0, 1.0, 1.0)):
    """Render one frame and return an (h, w, 4) uint8 array."""
    camera._rebuild_projmat(width, height)
    target = OffscreenTarget(width, height, sample_count)
    render = wvr.Render(camera, COLOR_FORMAT, sample_count)
    plan = render.prepare(scene, width, height)
    encoder = target.device.create_command_encoder()
    color_att = {
        'view': target.msaa_tex.create_view(),
        'resolve_target': target.resolve_tex.create_view(),
        'clear_value': bg,
        'load_op': wgpu.LoadOp.clear,
        'store_op': wgpu.StoreOp.store,
    }
    depth_att = {
        'view': target.depth_tex.create_view(),
        'depth_clear_value': 1.0,
        'depth_load_op': wgpu.LoadOp.clear,
        'depth_store_op': wgpu.StoreOp.store,
    }
    render_pass = encoder.begin_render_pass(
        color_attachments=[color_att], depth_stencil_attachment=depth_att)
    render.draw(render_pass, plan)
    render_pass.end()
    target.device.queue.submit([encoder.finish()])
    return target.read_rgba()


def save_png(path, rgba):
    """Minimal zlib/PNG writer so verification needs no image library."""
    import struct
    import zlib
    h, w, _ = rgba.shape
    raw = b''.join(b'\x00' + rgba[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        return (struct.pack('>I', len(payload)) + tag + payload +
                struct.pack('>I', zlib.crc32(tag + payload) & 0xffffffff))

    png = (b'\x89PNG\r\n\x1a\n' +
           chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress(raw, 6)) +
           chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)
    return path
