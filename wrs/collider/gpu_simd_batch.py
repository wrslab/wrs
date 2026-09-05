"""GPU batch triangle-mesh collision on wgpu compute.

Same create_detector / build_batch / detect_collision_batch interface as
cpu_simd, so a caller only changes which module it imports.  Two things shape
the design:

* the per-pose work-group count is folded into a 2D grid, so a dispatch past
  the 65535-per-dimension limit still runs in one pass -- dense end-effector
  meshes would otherwise have nowhere to go;
* detect_collision_multi tests a whole list of placements in one call.  A call
  costs ~0.4 ms of fixed command-buffer work regardless of its size, so a
  planner that checks its candidates one at a time pays that per candidate;
  handing over the whole list pays it once.
"""
import os

import numpy as np
import wrs.utils.gpu_device as wugd
import wrs.collider.gpu_broadphase as wcgbp
import wrs.collider.gpu_collision_batch as wcgcb

_SHADER = os.path.join(os.path.dirname(__file__), 'shaders',
                       'broadphase_batch.wgsl')


class WgpuBatchDetector:

    def __init__(self):
        self.device = wugd.get_device()
        with open(_SHADER, encoding='utf-8') as f:
            module = self.device.create_shader_module(
                code=f.read(), label='broadphase_batch')
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[wcgbp.bind_group_layout(self.device)])
        self.bounds_pipeline = self.device.create_compute_pipeline(
            layout=layout, compute={'module': module, 'entry_point': 'bounds'})
        self.broad_pipeline = self.device.create_compute_pipeline(
            layout=layout, compute={'module': module, 'entry_point': 'broad'})
        self.narrow_pipeline = self.device.create_compute_pipeline(
            layout=layout, compute={'module': module, 'entry_point': 'narrow'})

    def detect_collision_batch(self, batch, eps=1e-9):
        """One placement, taken from the items' live transforms.

        :return: (points, pair_ids) of the contacts found, or None.
        """
        if batch.total_pairs == 0:
            return None
        tfs = batch.snapshot_transforms()[None, ...]
        collided = batch.broadphase().run(self._pipelines(), tfs, eps,
                                          want_points=True)
        if not collided[0]:
            return None
        count, points4 = batch.read_staged()
        if count == 0:
            return None
        return points4[:, :3], points4[:, 3].astype(np.uint32, copy=False)

    def _pipelines(self):
        return (self.bounds_pipeline, self.broad_pipeline,
                self.narrow_pipeline)

    def detect_collision_multi(self, batch, tfs_kn, eps=1e-9):
        """Many placements in one dispatch.

        :param tfs_kn: (K, n_items, 4, 4) world transforms, one set per pose,
            as collected with ``batch.snapshot_transforms()``.
        :return: bool array of length K, True where that pose collides.
        """
        n_poses = len(tfs_kn)
        if n_poses == 0:
            return np.zeros(0, dtype=bool)
        if batch.total_pairs == 0:
            return np.zeros(n_poses, dtype=bool)
        broadphase = batch.broadphase()
        pipelines = self._pipelines()
        per_run = wcgbp.max_poses_per_run(broadphase.tri_total)
        if n_poses <= per_run:
            return broadphase.run(pipelines, tfs_kn, eps)
        # the survivor lists are sized per pose, so a very long list is split
        out = np.zeros(n_poses, dtype=bool)
        for start in range(0, n_poses, per_run):
            stop = min(start + per_run, n_poses)
            out[start:stop] = broadphase.run(
                pipelines, tfs_kn[start:stop], eps)
        return out


def create_detector():
    return WgpuBatchDetector()


def build_batch(items, pairs):
    return wcgcb.WgpuCollisionBatch(items, pairs)
