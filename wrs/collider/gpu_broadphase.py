"""Two-kernel broad/narrow driver for the batched triangle collider.

Owns the buffers broadphase_batch.wgsl needs on top of what the batch already
holds (vertices, faces, descriptors, pairs, transforms are shared).  The layout
is:

* ``tri_prefix``  -- prefix over pairs of (f_cnt_a + f_cnt_b); broad's thread
  space is one triangle of one side of one pair, so this is how a thread finds
  which pair and which side it belongs to;
* ``surv_off``    -- where each (pair, side) survivor list starts inside a
  pose's region, so every list has room for its side's full triangle count and
  compaction can never overflow;
* ``counts``      -- survivors per (pose, pair, side), read back once per call;
* ``p2_prefix``   -- prefix over (pose, pair) of n_a * n_b, built on the CPU
  from those counts, which is what sizes the narrow dispatch exactly.

The counts readback is the one sync in the middle.  It costs the same ~0.4 ms
as any other round trip, but it is paid once for the whole pose list rather
than per pose, so it disappears against the batch.
"""
import math

import numpy as np
import wgpu

from wrs.collider.gpu_collision_batch import POINTS_OFFSET

WORKGROUP_SIZE = 256
PARAMS_SIZE = 48
# survivors buffer is n_poses * tri_total * 4 bytes; beyond this the caller
# splits its pose list instead
SURVIVOR_BYTES_BUDGET = 128 * 1024 * 1024
# narrow is one thread per surviving triangle pair, which for a batch of a few
# hundred grasp candidates runs into the billions.  Its pose list is split so
# no dispatch exceeds this: it keeps the thread index (and p2_prefix, a u32
# array) clear of overflow, and keeps a submit near 80 ms at the measured
# ~25 G threads/s, well under the OS GPU watchdog.  A single pose can never
# exceed it -- its worst case is the batch's own triangle-pair count.
NARROW_THREADS_PER_SUBMIT = 2_000_000_000

_SU = wgpu.BufferUsage
_READ = _SU.STORAGE | _SU.COPY_DST
_RW = _SU.STORAGE | _SU.COPY_DST | _SU.COPY_SRC
_STAGING = _SU.MAP_READ | _SU.COPY_DST
_UNIFORM = _SU.UNIFORM | _SU.COPY_DST

# an empty survivor box: min keys at the top of the order, max keys at the
# bottom, so nothing overlaps it until something is accumulated
_EMPTY_BOX = np.array([0xFFFFFFFF] * 3 + [0] * 3, dtype=np.uint32)

_layout_cache = {}


def bind_group_layout(device):
    cached = _layout_cache.get(id(device))
    if cached is not None:
        return cached
    read_only = {'type': wgpu.BufferBindingType.read_only_storage}
    storage = {'type': wgpu.BufferBindingType.storage}
    entries = []
    for binding in (0, 1, 2, 3, 4, 5, 6, 7, 10):
        entries.append({'binding': binding,
                        'visibility': wgpu.ShaderStage.COMPUTE,
                        'buffer': read_only})
    for binding in (8, 9, 11, 13, 14, 15):
        entries.append({'binding': binding,
                        'visibility': wgpu.ShaderStage.COMPUTE,
                        'buffer': storage})
    entries.append({'binding': 12,
                    'visibility': wgpu.ShaderStage.COMPUTE,
                    'buffer': {'type': wgpu.BufferBindingType.uniform}})
    layout = device.create_bind_group_layout(entries=sorted(
        entries, key=lambda e: e['binding']))
    _layout_cache[id(device)] = layout
    return layout


def max_poses_per_run(tri_total):
    return max(1, SURVIVOR_BYTES_BUDGET // max(1, tri_total * 4))


class BroadPhase:

    def __init__(self, batch):
        self.batch = batch
        self.device = batch.device
        self.limit = int(
            self.device.limits['max-compute-workgroups-per-dimension'])
        f_cnt = batch.geom_descs[:, 3].astype(np.int64)
        spans, offsets = [], []
        running = 0
        for (i, j) in batch.pairs:
            n_a, n_b = int(f_cnt[i]), int(f_cnt[j])
            offsets.extend((running, running + n_a))
            running += n_a + n_b
            spans.append(n_a + n_b)
        self.tri_total = running
        self.tri_prefix = np.zeros(len(spans) + 1, dtype=np.uint32)
        self.tri_prefix[1:] = np.cumsum(spans, dtype=np.uint64).astype(
            np.uint32)
        self.surv_off = np.asarray(offsets, dtype=np.uint32)
        self.n_pairs = len(batch.pairs)
        self._capacity = 0
        self._bound_tfs = None
        self._build_static_buffers()

    # ------------------------------------------------------------------ run

    def run(self, pipelines, tfs_kn, eps, want_points=False):
        """Return a bool array of length K: did that pose collide?

        ``want_points`` additionally records contact points into the batch's
        results buffer -- only meaningful for a single-pose query, since the
        points carry no pose index.
        """
        bounds_pipe, broad_pipe, narrow_pipe = pipelines
        n_poses = len(tfs_kn)
        batch = self.batch
        batch.upload_poses(tfs_kn)
        self._ensure_capacity(n_poses)
        self._upload_item_aabbs(tfs_kn)
        zeros = np.zeros(n_poses * self.n_pairs * 2, dtype=np.uint32)
        self.device.queue.write_buffer(self._counts_buf, 0, zeros)
        self.device.queue.write_buffer(
            self._flags_buf, 0, np.zeros(n_poses, dtype=np.uint32))
        self.device.queue.write_buffer(
            self._surv_box_buf, 0, np.tile(_EMPTY_BOX, n_poses * self.n_pairs * 2))
        if want_points:
            batch.clear_counter()
        self._want_points = want_points
        counts = self._run_broad((bounds_pipe, broad_pipe), n_poses, eps)
        work = counts[:, :, 0].astype(np.uint64) * counts[:, :, 1]
        if work.sum() == 0:
            return np.zeros(n_poses, dtype=bool)
        self._run_narrow(narrow_pipe, work, n_poses, eps)
        return self._read_flags(n_poses)

    def _run_broad(self, pipelines, n_poses, eps):
        bounds_pipe, broad_pipe = pipelines
        n_x, n_y = _fold(self.tri_total, WORKGROUP_SIZE, self.limit)
        n_counts = n_poses * self.n_pairs * 2
        base = 0
        # pose_base lives in the params buffer, and a queue write only orders
        # against submits -- so a chunk gets its own encoder and submit
        while base < n_poses:
            n_z = min(self.limit, n_poses - base)
            self._set_params(n_poses, n_x, 1, base, eps)
            encoder = self.device.create_command_encoder()
            # passes in one command buffer are ordered, so broad sees the boxes
            self._dispatch(encoder, bounds_pipe, n_x, n_y, n_z)
            self._dispatch(encoder, broad_pipe, n_x, n_y, n_z)
            base += n_z
            if base >= n_poses:
                encoder.copy_buffer_to_buffer(
                    self._counts_buf, 0, self._counts_staging, 0, n_counts * 4)
            self.device.queue.submit([encoder.finish()])
        self._counts_staging.map_sync(wgpu.MapMode.READ, 0, n_counts * 4)
        try:
            raw = np.frombuffer(
                self._counts_staging.read_mapped(0, n_counts * 4), np.uint32)
            return raw.reshape(n_poses, self.n_pairs, 2).copy()
        finally:
            self._counts_staging.unmap()

    def _pose_spans(self, work):
        """Pose ranges whose narrow thread counts each fit the budget."""
        per_pose = work.sum(axis=1)
        spans, start, running = [], 0, 0
        for i, w in enumerate(per_pose):
            if running and running + w > NARROW_THREADS_PER_SUBMIT:
                spans.append((start, i))
                start, running = i, 0
            running += int(w)
        spans.append((start, len(per_pose)))
        return spans

    def _run_narrow(self, pipeline, work, n_poses, eps):
        for start, stop in self._pose_spans(work):
            span = stop - start
            prefix = np.zeros(span * self.n_pairs + 1, dtype=np.uint64)
            prefix[1:] = np.cumsum(work[start:stop].ravel())
            total = int(prefix[-1])
            if total == 0:
                continue
            self.device.queue.write_buffer(
                self._p2_prefix_buf, 0, prefix.astype(np.uint32))
            n_x, n_y = _fold(total, WORKGROUP_SIZE, self.limit)
            # pose_base / p2_prefix live in buffers, and a queue write only
            # orders against submits -- so each span gets its own encoder
            self._set_params(n_poses, 0, n_x, start, eps, pose_span=span)
            encoder = self.device.create_command_encoder()
            self._dispatch(encoder, pipeline, n_x, n_y, 1)
            self.device.queue.submit([encoder.finish()])
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(
            self._flags_buf, 0, self._flags_staging, 0, _align4(n_poses * 4))
        if self._want_points:
            self.batch.copy_results_to_staging(encoder)
        self.device.queue.submit([encoder.finish()])

    def _dispatch(self, encoder, pipeline, n_x, n_y, n_z):
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, self.bind_group)
        compute_pass.dispatch_workgroups(n_x, n_y, n_z)
        compute_pass.end()

    def _read_flags(self, n_poses):
        n_bytes = _align4(n_poses * 4)
        self._flags_staging.map_sync(wgpu.MapMode.READ, 0, n_bytes)
        try:
            raw = np.frombuffer(
                self._flags_staging.read_mapped(0, n_bytes), np.uint32)
            return raw[:n_poses].astype(bool)
        finally:
            self._flags_staging.unmap()

    # --------------------------------------------------------------- uploads

    def _set_params(self, n_poses, row_groups, row_groups2, pose_base, eps,
                    pose_span=0):
        params = np.zeros(12, dtype=np.uint32)
        params[9] = 1 if getattr(self, '_want_points', False) else 0
        params[10] = self.batch._max_points
        params[0] = self.n_pairs
        params[1] = self.batch.n_items
        params[2] = self.tri_total
        params[3] = row_groups * WORKGROUP_SIZE
        params[4] = n_poses
        params[5] = row_groups2 * WORKGROUP_SIZE
        params[6] = pose_base
        params[7] = pose_span
        params[8] = np.float32(eps).view(np.uint32)
        self.device.queue.write_buffer(self._params_buf, 0, params)

    def _upload_item_aabbs(self, tfs_kn):
        """World AABB per (pose, item), from the items' local AABBs."""
        batch = self.batch
        centers = (batch.aabb_mins + batch.aabb_maxs) * 0.5
        halfs = (batch.aabb_maxs - batch.aabb_mins) * 0.5
        rotmats = tfs_kn[:, :, :3, :3]
        wd_centers = (np.einsum('knij,nj->kni', rotmats, centers) +
                      tfs_kn[:, :, :3, 3])
        wd_halfs = np.einsum('knij,nj->kni', np.abs(rotmats), halfs)
        k, n = tfs_kn.shape[0], tfs_kn.shape[1]
        packed = np.zeros((k, n, 2, 4), dtype=np.float32)
        packed[:, :, 0, :3] = wd_centers - wd_halfs
        packed[:, :, 1, :3] = wd_centers + wd_halfs
        self.device.queue.write_buffer(self._item_aabb_buf, 0, packed)

    # ----------------------------------------------------------------- setup

    def _build_static_buffers(self):
        dev = self.device
        self._tri_prefix_buf = dev.create_buffer_with_data(
            data=self.tri_prefix, usage=_READ)
        self._surv_off_buf = dev.create_buffer_with_data(
            data=self.surv_off, usage=_READ)
        self._params_buf = dev.create_buffer(size=PARAMS_SIZE, usage=_UNIFORM)

    def _ensure_capacity(self, n_poses):
        batch = self.batch
        if (n_poses <= self._capacity and
                self._bound_tfs is batch._tfs_buf):
            return
        dev = self.device
        capacity = max(n_poses, 1)
        n_counts = capacity * self.n_pairs * 2
        self._survivors_buf = dev.create_buffer(
            size=capacity * self.tri_total * 4, usage=_READ)
        self._counts_buf = dev.create_buffer(size=n_counts * 4, usage=_RW)
        self._counts_staging = dev.create_buffer(
            size=n_counts * 4, usage=_STAGING)
        self._item_aabb_buf = dev.create_buffer(
            size=capacity * batch.n_items * 32, usage=_READ)
        self._p2_prefix_buf = dev.create_buffer(
            size=(n_counts // 2 + 1) * 4, usage=_READ)
        self._surv_box_buf = dev.create_buffer(size=n_counts * 6 * 4, usage=_RW)
        self._flags_buf = dev.create_buffer(
            size=_align4(capacity * 4), usage=_RW)
        self._flags_staging = dev.create_buffer(
            size=_align4(capacity * 4), usage=_STAGING)
        self._capacity = capacity
        self._bound_tfs = batch._tfs_buf
        whole = _whole
        self.bind_group = dev.create_bind_group(
            layout=bind_group_layout(dev),
            entries=[
                {'binding': 0, 'resource': whole(batch._verts_buf)},
                {'binding': 1, 'resource': whole(batch._faces_buf)},
                {'binding': 2, 'resource': whole(batch._desc_buf)},
                {'binding': 3, 'resource': whole(batch._pairs_buf)},
                {'binding': 4, 'resource': whole(batch._tfs_buf)},
                {'binding': 5, 'resource': whole(self._tri_prefix_buf)},
                {'binding': 6, 'resource': whole(self._item_aabb_buf)},
                {'binding': 7, 'resource': whole(self._surv_off_buf)},
                {'binding': 8, 'resource': whole(self._counts_buf)},
                {'binding': 9, 'resource': whole(self._survivors_buf)},
                {'binding': 10, 'resource': whole(self._p2_prefix_buf)},
                {'binding': 11, 'resource': whole(self._flags_buf)},
                {'binding': 12, 'resource': whole(self._params_buf)},
                {'binding': 13, 'resource': whole(self._surv_box_buf)},
                {'binding': 14, 'resource': {'buffer': batch._results_buf,
                                             'offset': POINTS_OFFSET,
                                             'size': batch._max_points * 16}},
                {'binding': 15, 'resource': {'buffer': batch._results_buf,
                                             'offset': 0, 'size': 4}},
            ])


def _fold(n_items, group_size, limit):
    """Work groups covering n_items, folded near-square inside the limit."""
    groups = max(1, (n_items + group_size - 1) // group_size)
    n_x = min(groups, limit)
    if groups > limit:
        n_x = min(limit, int(math.isqrt(groups)) + 1)
    n_y = max(1, -(-groups // n_x))
    if n_y > limit:
        raise RuntimeError(f'{n_items} items exceed a {limit}x{limit} dispatch')
    return n_x, n_y


def _whole(buf):
    return {'buffer': buf, 'offset': 0, 'size': buf.size}


def _align4(n_bytes):
    return n_bytes + (-n_bytes) % 4
