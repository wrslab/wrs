"""Renderer for the wgpu backend.

The GL renderer drives global state (glCullFace / glDepthMask / glEnable) and
draws immediately.  wgpu bakes all of that into immutable pipeline objects, so
the same four passes become four pipelines built once here:

    outline      cull front, depth write        -- inflated black back hull
    solid        cull back,  depth write
    transparent  cull back,  no depth write, alpha blend, back-to-front
    pcd          no cull,    depth write        -- instanced screen-space quads

Recording is split in two: prepare() does every queue.write_buffer while no
render pass is open (writing a buffer a live pass reads is a hazard), then
draw() only records draw calls.
"""
import numpy as np
import wgpu
import wrs.viewer.context as wvc
import wrs.viewer.shader as wvs

DEPTH_FORMAT = wgpu.TextureFormat.depth24plus
POINT_SIZE = 5.0

_GLOBALS_FLOATS = 40  # view 16 + proj 16 + view_pos 4 + viewport 2 + size 1 + pad

_MESH_VERTEX_LAYOUT = [
    {'array_stride': 24, 'step_mode': wgpu.VertexStepMode.vertex,
     'attributes': [
         {'format': wgpu.VertexFormat.float32x3, 'offset': 0, 'shader_location': 0},
         {'format': wgpu.VertexFormat.float32x3, 'offset': 12, 'shader_location': 1}]},
    {'array_stride': 64, 'step_mode': wgpu.VertexStepMode.instance,
     'attributes': [
         {'format': wgpu.VertexFormat.float32x4, 'offset': o, 'shader_location': 2 + i}
         for i, o in enumerate((0, 16, 32, 48))]},
    {'array_stride': 16, 'step_mode': wgpu.VertexStepMode.instance,
     'attributes': [
         {'format': wgpu.VertexFormat.float32x4, 'offset': 0, 'shader_location': 6}]},
]

_PCD_VERTEX_LAYOUT = [
    {'array_stride': 8, 'step_mode': wgpu.VertexStepMode.vertex,
     'attributes': [
         {'format': wgpu.VertexFormat.float32x2, 'offset': 0, 'shader_location': 0}]},
    {'array_stride': 24, 'step_mode': wgpu.VertexStepMode.instance,
     'attributes': [
         {'format': wgpu.VertexFormat.float32x3, 'offset': 0, 'shader_location': 1},
         {'format': wgpu.VertexFormat.float32x3, 'offset': 12, 'shader_location': 2}]},
]

_ALPHA_BLEND = {
    'color': {'src_factor': wgpu.BlendFactor.src_alpha,
              'dst_factor': wgpu.BlendFactor.one_minus_src_alpha,
              'operation': wgpu.BlendOperation.add},
    'alpha': {'src_factor': wgpu.BlendFactor.src_alpha,
              'dst_factor': wgpu.BlendFactor.one_minus_src_alpha,
              'operation': wgpu.BlendOperation.add},
}


class Render:

    def __init__(self, camera, color_format, sample_count=4):
        self.camera = camera
        self.device = wvc.get_device()
        self.color_format = color_format
        self.sample_count = sample_count
        self._groups_cache = None
        self._globals = np.zeros(_GLOBALS_FLOATS, dtype=np.float32)
        self._build_bind_layouts()
        self._build_pipelines()
        self._build_static_buffers()
        # per-frame transparent instance storage, grown on demand
        self._tp_tf_buf = None
        self._tp_rgba_buf = None
        self._tp_capacity = 0

    # ------------------------------------------------------------------ setup

    def _build_bind_layouts(self):
        uniform = {'type': wgpu.BufferBindingType.uniform}
        self.globals_layout = self.device.create_bind_group_layout(entries=[
            {'binding': 0,
             'visibility': wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             'buffer': uniform}])
        self.model_layout = self.device.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.VERTEX,
             'buffer': uniform}])
        self.globals_buf = self.device.create_buffer(
            size=_GLOBALS_FLOATS * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.globals_bind_group = self.device.create_bind_group(
            layout=self.globals_layout,
            entries=[{'binding': 0,
                      'resource': {'buffer': self.globals_buf,
                                   'offset': 0,
                                   'size': _GLOBALS_FLOATS * 4}}])

    def _pipeline(self, module, vs, fs, vertex_buffers,
                  cull, depth_write, blend, bind_layouts):
        return self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=bind_layouts),
            vertex={'module': module, 'entry_point': vs,
                    'buffers': vertex_buffers},
            primitive={'topology': wgpu.PrimitiveTopology.triangle_list,
                       'front_face': wgpu.FrontFace.ccw,
                       'cull_mode': cull},
            depth_stencil={'format': DEPTH_FORMAT,
                           'depth_write_enabled': depth_write,
                           'depth_compare': wgpu.CompareFunction.less},
            multisample={'count': self.sample_count},
            fragment={'module': module, 'entry_point': fs,
                      'targets': [{'format': self.color_format,
                                   'blend': blend}]})

    def _build_pipelines(self):
        mesh_mod = self.device.create_shader_module(
            code=wvs.mesh_wgsl, label='mesh')
        pcd_mod = self.device.create_shader_module(
            code=wvs.pcd_wgsl, label='pcd')
        g = [self.globals_layout]
        self.outline_pipeline = self._pipeline(
            mesh_mod, 'vs_outline', 'fs_outline', _MESH_VERTEX_LAYOUT,
            wgpu.CullMode.front, True, None, g)
        self.solid_pipeline = self._pipeline(
            mesh_mod, 'vs_mesh', 'fs_matte', _MESH_VERTEX_LAYOUT,
            wgpu.CullMode.back, True, None, g)
        self.transparent_pipeline = self._pipeline(
            mesh_mod, 'vs_mesh', 'fs_matte', _MESH_VERTEX_LAYOUT,
            wgpu.CullMode.back, False, _ALPHA_BLEND, g)
        self.pcd_pipeline = self._pipeline(
            pcd_mod, 'vs_pcd', 'fs_pcd', _PCD_VERTEX_LAYOUT,
            wgpu.CullMode.none, True, None, g + [self.model_layout])

    def _build_static_buffers(self):
        quad = np.array([-1, -1, 1, -1, 1, 1,
                         -1, -1, 1, 1, -1, 1], dtype=np.float32)
        self.quad_buf = self.device.create_buffer_with_data(
            data=quad, usage=wgpu.BufferUsage.VERTEX)

    # ---------------------------------------------------------------- prepare

    def prepare(self, scene, width, height):
        """Upload everything this frame needs; return the recorded draw plan."""
        self._update_globals(width, height)
        if scene.dirty or self._groups_cache is None:
            self._groups_cache = self._build_shader_groups(scene)
            scene.dirty = False
        solid = self._prepare_solid(self._groups_cache['mesh_solid'])
        transparent = self._prepare_transparent(
            self._groups_cache['mesh_transparent'])
        pcd = self._prepare_pcd(self._groups_cache['pcd'])
        return solid, transparent, pcd

    def _update_globals(self, width, height):
        self._globals[0:16] = self.camera.view_mat.T.ravel()
        self._globals[16:32] = self.camera.proj_mat.T.ravel()
        self._globals[32:35] = self.camera.pos
        self._globals[36] = width
        self._globals[37] = height
        self._globals[38] = POINT_SIZE
        self.device.queue.write_buffer(self.globals_buf, 0, self._globals)

    def _build_shader_groups(self, scene):
        groups = {'mesh_solid': {}, 'mesh_transparent': {}, 'pcd': {}}
        for sobj in scene:
            models = list(sobj.visuals)
            if sobj.toggle_render_collision:
                models += [c.to_render_model() for c in sobj.collisions]
            for model in models:
                device_buffer = model.get_device_buffer()
                if model.geom.fs is None:
                    key = 'pcd'
                elif model.alpha < 0.999:
                    key = 'mesh_transparent'
                else:
                    key = 'mesh_solid'
                groups[key].setdefault(device_buffer, []).append((model, sobj))
        return groups

    def _prepare_solid(self, group):
        plan = []
        for device, instance_list in group.items():
            n = len(instance_list)
            tf_arr = np.empty((n, 4, 4), np.float32)
            rgba_arr = np.empty((n, 4), np.float32)
            for i, (model, node) in enumerate(instance_list):
                tf_arr[i] = (node.tf @ model.loc_tf).T
                rgba_arr[i] = (*model.rgb, model.alpha)
            device.update_instances(tf_arr, rgba_arr)
            plan.append(device)
        return plan

    def _prepare_transparent(self, group):
        """Back-to-front, into one shared per-frame instance buffer.

        The GL path re-uploaded the same VBO between draws; here every upload
        happens before the pass opens, so each model instead gets its own slot
        and is drawn with a first_instance offset.
        """
        entries = []
        cam_pos = np.asarray(self.camera.pos, dtype=np.float32)
        for device, instance_list in group.items():
            for model, node in instance_list:
                tf = node.tf @ model.loc_tf
                d = tf[:3, 3] - cam_pos
                entries.append((float(d @ d), device, model, tf))
        if not entries:
            return []
        entries.sort(key=lambda x: x[0], reverse=True)
        n = len(entries)
        tf_arr = np.empty((n, 4, 4), np.float32)
        rgba_arr = np.empty((n, 4), np.float32)
        for i, (_, device, model, tf) in enumerate(entries):
            tf_arr[i] = tf.T
            rgba_arr[i] = (*model.rgb, model.alpha)
        self._ensure_transparent_capacity(n)
        self.device.queue.write_buffer(self._tp_tf_buf, 0, tf_arr)
        self.device.queue.write_buffer(self._tp_rgba_buf, 0, rgba_arr)
        return [(entries[i][1], i) for i in range(n)]

    def _ensure_transparent_capacity(self, n):
        if self._tp_tf_buf is not None and n <= self._tp_capacity:
            return
        self._tp_capacity = int(n * 1.5) + 8
        usage = wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
        self._tp_tf_buf = self.device.create_buffer(
            size=self._tp_capacity * 64, usage=usage)
        self._tp_rgba_buf = self.device.create_buffer(
            size=self._tp_capacity * 16, usage=usage)

    def _prepare_pcd(self, group):
        plan = []
        for device, instance_list in group.items():
            for model, node in instance_list:
                device.set_model((node.tf @ model.loc_tf).T)
                if device.model_bind_group is None:
                    device.model_bind_group = self.device.create_bind_group(
                        layout=self.model_layout,
                        entries=[{'binding': 0,
                                  'resource': {'buffer': device.model_buf,
                                               'offset': 0, 'size': 64}}])
                plan.append(device)
        return plan

    # ------------------------------------------------------------------- draw

    def draw(self, render_pass, plan):
        solid, transparent, pcd = plan
        render_pass.set_bind_group(0, self.globals_bind_group)
        if solid:
            render_pass.set_pipeline(self.outline_pipeline)
            for device in solid:
                device.draw_instanced(render_pass)
            render_pass.set_pipeline(self.solid_pipeline)
            for device in solid:
                device.draw_instanced(render_pass)
        if transparent:
            render_pass.set_pipeline(self.transparent_pipeline)
            for device, slot in transparent:
                render_pass.set_vertex_buffer(0, device.vertex_buf)
                render_pass.set_vertex_buffer(1, self._tp_tf_buf)
                render_pass.set_vertex_buffer(2, self._tp_rgba_buf)
                render_pass.set_index_buffer(device.index_buf,
                                             wgpu.IndexFormat.uint32)
                render_pass.draw_indexed(device.count, 1, 0, 0, slot)
        if pcd:
            render_pass.set_pipeline(self.pcd_pipeline)
            render_pass.set_vertex_buffer(0, self.quad_buf)
            for device in pcd:
                render_pass.set_bind_group(1, device.model_bind_group)
                device.draw(render_pass)
