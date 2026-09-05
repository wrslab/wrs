"""wgpu counterpart of viewer.device_buffer.

The attribute layout lives in the render pipeline, so these objects only own
memory.  Mesh vertex/index buffers additionally carry the STORAGE flag, which
lets the GPU collider bind them directly as storage buffers -- possible only
because both sit on the one device (see context.get_device).
"""
import numpy as np
import wgpu
import wrs.viewer.context as wvc

_BU = wgpu.BufferUsage
_MESH_USAGE = _BU.VERTEX | _BU.STORAGE | _BU.COPY_DST
_INDEX_USAGE = _BU.INDEX | _BU.STORAGE | _BU.COPY_DST
_INSTANCE_USAGE = _BU.VERTEX | _BU.COPY_DST


class DeviceBufferBase:

    def __init__(self):
        self.device = wvc.get_device()
        self.count = 0


class MeshBuffer(DeviceBufferBase):

    def __init__(self, verts, faces, vert_normals):
        super().__init__()
        self.instance_count = 0
        self.tf_buf = None
        self.rgba_buf = None
        self._cached_tf = None
        self._cached_rgba = None
        self._tf_capacity = 0
        self._rgba_capacity = 0
        self._build(verts, faces, vert_normals)

    def update_instances(self, tf_arr, rgba_arr=None):
        """Upload per-instance model matrices / colors, skipping unchanged data.

        ``tf_arr`` is (N, 4, 4) already transposed by the caller, matching the
        column-major mat4x4 the shader reconstructs from locations 2..5.
        """
        self.instance_count = len(tf_arr)
        data = np.ascontiguousarray(tf_arr, dtype=np.float32)
        self.tf_buf, self._cached_tf, self._tf_capacity = self._upload(
            data, self.tf_buf, self._cached_tf, self._tf_capacity)
        if rgba_arr is not None:
            data = np.ascontiguousarray(rgba_arr, dtype=np.float32)
            self.rgba_buf, self._cached_rgba, self._rgba_capacity = self._upload(
                data, self.rgba_buf, self._cached_rgba, self._rgba_capacity)

    def bind(self, render_pass):
        render_pass.set_vertex_buffer(0, self.vertex_buf)
        render_pass.set_vertex_buffer(1, self.tf_buf)
        render_pass.set_vertex_buffer(2, self.rgba_buf)
        render_pass.set_index_buffer(self.index_buf, wgpu.IndexFormat.uint32)

    def draw_instanced(self, render_pass):
        if self.instance_count <= 0:
            return
        self.bind(render_pass)
        render_pass.draw_indexed(self.count, self.instance_count)

    def _upload(self, data, buf, cached, capacity):
        if buf is None or data.nbytes > capacity:
            capacity = int(data.nbytes * 1.5) + 256
            capacity += (-capacity) % 4
            buf = self.device.create_buffer(size=capacity,
                                            usage=_INSTANCE_USAGE)
            cached = None
        if cached is None or not np.array_equal(data, cached):
            self.device.queue.write_buffer(buf, 0, data)
            cached = data.copy()
        return buf, cached, capacity

    def _build(self, verts, faces, vert_normals):
        array = np.ascontiguousarray(
            np.hstack([verts, vert_normals]), dtype=np.float32)
        self.vertex_buf = self.device.create_buffer_with_data(
            data=array, usage=_MESH_USAGE)
        indices = np.ascontiguousarray(faces, dtype=np.uint32)
        self.index_buf = self.device.create_buffer_with_data(
            data=indices, usage=_INDEX_USAGE)
        self.count = indices.size


class PointCloudBuffer(DeviceBufferBase):
    """One instance per point; the quad corners come from Render's shared
    unit-quad buffer."""

    def __init__(self, vs, vrgbs):
        super().__init__()
        self.model_buf = self.device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | _BU.COPY_DST)
        self.model_bind_group = None  # filled in by Render on first draw
        self._build(vs, vrgbs)

    def set_model(self, tf):
        self.device.queue.write_buffer(
            self.model_buf, 0, np.ascontiguousarray(tf, dtype=np.float32))

    def draw(self, render_pass):
        if self.count <= 0:
            return
        render_pass.set_vertex_buffer(1, self.instance_buf)
        render_pass.draw(6, self.count)

    def _build(self, vs, vrgbs):
        self.count = len(vs)
        array = np.ascontiguousarray(
            np.hstack([vs, vrgbs]), dtype=np.float32)
        self.instance_buf = self.device.create_buffer_with_data(
            data=array, usage=_INSTANCE_USAGE)
