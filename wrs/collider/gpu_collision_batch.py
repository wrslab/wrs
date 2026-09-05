"""wgpu storage buffers for the batched triangle collider.

The CPU-side packing all lives in CollisionBatch; this only owns GPU memory,
on the device from utils.gpu_device.

The geometry buffers here are shared and static; what varies per pose (the
transforms) is sized for K poses at once, because a call costs ~0.4 ms of
fixed command-buffer work whatever it computes and a caller amortises that by
submitting its whole candidate list.  Only the transforms are per-pose -- the
vertex, face and pair buffers are shared -- so K poses cost K * n_items * 64
bytes; a thousand placements of a gripper is a few hundred kilobytes, not a
thousand copies of the meshes.

The kernels reading these buffers, and the buffers only they need, live in
gpu_broadphase.  The results buffer here holds the contact count and points
(counter at 0, points at a 256-aligned offset, so one readback covers both),
which only a single-pose query fills.
"""
import numpy as np
import wgpu
import wrs.utils.gpu_device as wugd
from wrs.collider.collision_batch import CollisionBatch

_SU = wgpu.BufferUsage
_READ_ONLY = _SU.STORAGE | _SU.COPY_DST
_RESULTS = _SU.STORAGE | _SU.COPY_DST | _SU.COPY_SRC
_STAGING = _SU.MAP_READ | _SU.COPY_DST

MAX_POINTS = 200
POINTS_OFFSET = 256  # >= min_storage_buffer_offset_alignment


class WgpuCollisionBatch(CollisionBatch):

    def __init__(self, items, pairs=None):
        super().__init__(items, pairs)
        self.pair_prefix = self._build_pair_prefix()
        self._max_points = MAX_POINTS
        self.device = wugd.get_device()
        self.n_items = len(items)
        self.total_pairs = int(self.pair_prefix[-1])
        self.results_size = POINTS_OFFSET + self._max_points * 16
        self._build_static_buffers()
        self._pose_capacity = 0
        self._broadphase = None
        self._ensure_pose_capacity(1)

    def broadphase(self):
        """Lazily built broad/narrow driver for this batch."""
        if self._broadphase is None:
            import wrs.collider.gpu_broadphase as wcgbp
            self._broadphase = wcgbp.BroadPhase(self)
        return self._broadphase

    # -------------------------------------------------------------- per-call

    def snapshot_transforms(self):
        """The items' current world transforms, (n_items, 4, 4).

        Callers pose their mechanism, snapshot, repeat, then hand the stack to
        detect_collision_multi.
        """
        super().update_transforms()
        return self.tfs.copy()

    def upload_poses(self, tfs_kn):
        """``tfs_kn``: (K, n_items, 4, 4) world transforms, one set per pose."""
        tfs_kn = np.ascontiguousarray(tfs_kn, dtype=np.float32)
        if tfs_kn.shape[1:] != (self.n_items, 4, 4):
            raise ValueError(
                f'expected (K, {self.n_items}, 4, 4), got {tfs_kn.shape}')
        n_poses = tfs_kn.shape[0]
        self._ensure_pose_capacity(n_poses)
        self.device.queue.write_buffer(self._tfs_buf, 0, tfs_kn)
        return n_poses

    def clear_counter(self):
        self.device.queue.write_buffer(
            self._results_buf, 0, np.zeros(1, dtype=np.uint32))

    def copy_results_to_staging(self, encoder):
        encoder.copy_buffer_to_buffer(
            self._results_buf, 0, self._staging, 0, self.results_size)

    def read_staged(self):
        """Contact count and points of a single-pose query, from the staging
        copy."""
        self._staging.map_sync(wgpu.MapMode.READ)
        try:
            raw = np.frombuffer(self._staging.read_mapped(), np.uint8)
            count = int(raw[:4].view(np.uint32)[0])
            if count == 0:
                return 0, None
            count = min(count, self._max_points)
            points = raw[POINTS_OFFSET:POINTS_OFFSET + count * 16]
            return count, points.view(np.float32).reshape(-1, 4).copy()
        finally:
            self._staging.unmap()

    def release(self):
        for name in ('_verts_buf', '_faces_buf', '_desc_buf', '_pairs_buf',
                     '_results_buf', '_staging', '_tfs_buf'):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.destroy()
            setattr(self, name, None)
        self._broadphase = None

    # ----------------------------------------------------------------- setup

    def _storage(self, data, usage=_READ_ONLY):
        """Storage buffers may not be empty; pad degenerate inputs."""
        data = np.ascontiguousarray(data)
        if data.size == 0:
            data = np.zeros(4, dtype=data.dtype)
        return self.device.create_buffer_with_data(data=data, usage=usage)

    def _build_static_buffers(self):
        # the shaders read vec4 verts / uvec4 faces, so widen both
        vss4 = np.ones((self.vss.shape[0], 4), dtype=np.float32)
        vss4[:, :3] = self.vss
        fss4 = np.ones((self.fss.shape[0], 4), dtype=np.uint32)
        fss4[:, :3] = self.fss
        self._verts_buf = self._storage(vss4)
        self._faces_buf = self._storage(fss4)
        self._desc_buf = self._storage(self.geom_descs.astype(np.uint32))
        self._pairs_buf = self._storage(self.pairs.astype(np.uint32))
        self._results_buf = self.device.create_buffer(
            size=self.results_size, usage=_RESULTS)
        self._staging = self.device.create_buffer(
            size=self.results_size, usage=_STAGING)

    def _ensure_pose_capacity(self, n_poses):
        if n_poses <= self._pose_capacity:
            return
        self._tfs_buf = self.device.create_buffer(
            size=n_poses * self.n_items * 64, usage=_READ_ONLY)
        self._pose_capacity = n_poses

    def _build_pair_prefix(self):
        """index prefix for each pair's face count product"""
        counts = []
        for (i, j) in self.pairs:
            f_cnt_a = self.geom_descs[i][3]
            f_cnt_b = self.geom_descs[j][3]
            counts.append(np.uint64(f_cnt_a) * np.uint64(f_cnt_b))
        prefix = np.zeros(len(counts) + 1, dtype=np.uint32)
        if counts:
            prefix[1:] = np.cumsum(
                np.array(counts, dtype=np.uint64)).astype(np.uint32)
        return prefix
