"""Antipodal grasp planning: 2-point opposing pinch (force-closure) grasps
for a parallel-jaw gripper. ``antipodal``/``antipodal_iter`` sample the
target surface for antipodal contact pairs, align the jaw, and reject
gripper-vs-target collisions. See also polypodal (N-point) and monocontact
(single-contact / suction)."""
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.scene.geometry_ops as wsgop
import wrs.collider.cpu_simd as wccs
import wrs.grasp._common as wgc
from wrs.grasp.grasp import Grasp


def build_grasp_rotmat_batch(ray_dirs, open_dir):
    """
    ray_dirs: (N,3) unit vectors
    open_dir: gripper opening direction in TCP local frame -- (3,) shared by all
        candidates, or (N,3) one per candidate (e.g. a closure-dependent pinch).
    returns: rotmats (N,3,3)
    """
    y = ray_dirs / (np.linalg.norm(ray_dirs, axis=1, keepdims=True) + wum.eps)
    ref1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ref2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot1 = np.abs(y @ ref1)
    dot2 = np.abs(y @ ref2)
    use_ref1 = dot1 < dot2
    ref = np.where(use_ref1[:, None], ref1, ref2)
    z = ref - (np.sum(ref * y, axis=1, keepdims=True) * y)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + wum.eps)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + wum.eps)
    rot_base = np.stack([x, y, z], axis=2).astype(np.float32)
    open_dir = np.asarray(open_dir, dtype=np.float32)
    if np.linalg.norm(open_dir) < wum.eps:
        raise ValueError('open_dir must be non-zero')
    offset = wum.rotmat_between_vecs(
        open_dir, wuc.StandardAxis.Y).astype(np.float32)
    if offset.ndim == 2:                       # shared open_dir -> single offset
        return rot_base @ offset
    return np.einsum('nij,njk->nik', rot_base, offset)   # per-candidate


def _antipodal_candidates(
        tgt_vs, tgt_fs, tgt_fns,
        density, normal_tol_deg,
        roll_step_deg, clearance):
    normal_cos_th = np.cos(np.deg2rad(normal_tol_deg))
    roll_step = np.deg2rad(roll_step_deg)
    n_samples = wsgop.sample_count_from_area(tgt_vs, tgt_fs, density)
    pts, nrms, _ = wsgop.sample_surface(tgt_vs, tgt_fs, n_samples)
    v0 = tgt_vs[tgt_fs[:, 0]]
    v1 = tgt_vs[tgt_fs[:, 1]]
    v2 = tgt_vs[tgt_fs[:, 2]]
    origins = pts
    directions = -nrms
    hit_t, hit_id = wsgop.ray_triangles_batch_far(
        origins, directions, v0, v1, v2, eps=float(wum.eps))
    valid = hit_id >= 0
    if not np.any(valid):
        return None
    origins_v = origins[valid]
    directions_v = directions[valid]
    hit_t_v = hit_t[valid]
    hit_id_v = hit_id[valid]
    hit_pos = origins_v + hit_t_v[:, None] * directions_v
    hit_n = tgt_fns[hit_id_v]
    p_all = origins_v
    n_all = nrms[valid]
    q_all = hit_pos
    nq_all = hit_n
    dot_nn = np.einsum("ij,ij->i", n_all, -nq_all)
    n_norm = np.linalg.norm(n_all, axis=1) + wum.eps
    nq_norm = np.linalg.norm(nq_all, axis=1) + wum.eps
    cos_vals = dot_nn / (n_norm * nq_norm)
    jaw_width = np.linalg.norm(q_all - p_all, axis=1) + 2 * clearance
    center = (p_all + q_all) * 0.5
    return center, jaw_width, n_all, nq_all, cos_vals, normal_cos_th, roll_step


def antipodal_iter(gripper, target_sobj,
                   density=0.02, normal_tol_deg=20,
                   roll_step_deg=30, clearance=0.002,
                   score_weights=(0.7, 0.3),
                   exclude_regions=None, pre_open=0.5):
    """
    Generator: yields (grasp, collided) -- a Grasp and its collision flag.
    :param gripper: a parallel jaw -- a real gripper, or a dexterous hand's
        ``hand.as_jaw('pinch')`` (a JawView). Both implement the jaw protocol
        (jaw_range / set_opening / grasp_center_tcp / grip_at / contact_pattern).
    :param target_sobj: target object to grasp
    :param density: surface sampling density
    :param normal_tol_deg: normal tolerance in degrees
    :param roll_step_deg: roll angle step in degrees
    :param clearance: additional clearance for jaw width
    :param score_weights: (normal_align_weight, jaw_close_weight)
    :param exclude_regions: optional list of convex regions (each a
        list of (point, normal) half-space constraints) carved out of
        the surface before contact-pair sampling. The original target
        mesh is still used for gripper-vs-target collision checks, so
        grasps near a clipped feature will still be rejected if the
        gripper would collide with that feature.
    :param pre_open: how far OPEN the jaw is at the pre-grasp pose, as a
        fraction in [0, 1] of the room between the grasp width and the
        max opening (pre_jw = jw + pre_open*(jaw_max - jw)). 0 keeps it
        at the grasp width; the default 0.5 opens half-way so the
        collision check reflects the wider hand swept in on approach.
    Uses gripper.contact_pattern to confirm this is a single-contact
        model and to compensate jaw width for contact depth along the
        gripper opening axis. TCP is aligned to the two-contact midpoint.
    :return: yields (grasp, collided), where ``grasp`` is a
        :class:`~wrs.grasp.grasp.Grasp` in the target's LOCAL frame:
        - ``grasp.pose`` is the GRASP CENTER (grasp_center tcp) frame closed on
          the object -- origin at the contact-pair midpoint, +z the approach
          axis (exactly the (tgt_pos, tgt_rotmat) grip_at expects).
        - ``grasp.pre_pose`` is that pose retreated along the approach axis; its
          collision check uses the jaw opened part-way (see pre_open).
        - ``grasp.tcp`` / ``grasp.qpos`` / ``grasp.pre_qpos`` are the frozen tcp
          loc_tf and the hand configs at the grasp and pre-grasp openings.
        - ``grasp.provenance`` records the mode (if a dexterous hand) and width.
        ``collided`` is True if the gripper collides with the target at the
        grasp or pre-grasp pose (both are collision-checked).
    """
    gripper = gripper.clone()
    # Plan in the target's LOCAL (zero-pose) frame. Clone the target and zero
    # its pose so the contact sampling (local geom) and the gripper-vs-target
    # collision check (CollisionBatch uses target.tf) live in the SAME frame.
    # The returned grasps are therefore in the target's local frame; the caller
    # maps them onto the placed object (e.g. target_sobj.tf @ pose).
    target_sobj = target_sobj.clone()
    target_sobj.set_pos_rotmat(
        pos=np.zeros(3, dtype=np.float32), rotmat=np.eye(3, dtype=np.float32))
    tgt_vs, tgt_fs, tgt_fns = wccs.cols_to_vffns(
        target_sobj.collisions)
    if exclude_regions:
        tgt_vs, tgt_fs = wsgop.clip_mesh(
            tgt_vs, tgt_fs, exclude_regions)
        if len(tgt_fs) == 0:
            return []
        v0 = tgt_vs[tgt_fs[:, 0]]
        v1 = tgt_vs[tgt_fs[:, 1]]
        v2 = tgt_vs[tgt_fs[:, 2]]
        tgt_fns = np.cross(v1 - v0, v2 - v0)
        tgt_fns = tgt_fns / (
            np.linalg.norm(tgt_fns, axis=1, keepdims=True) + wum.eps)
        tgt_fns = tgt_fns.astype(np.float32)
    cand = _antipodal_candidates(
        tgt_vs, tgt_fs, tgt_fns, density,
        normal_tol_deg, roll_step_deg, clearance)
    if cand is None:
        return []
    (center, jaw_width, n_all, nq_all,
     cos_vals, normal_cos_th, roll_step) = cand
    contact_pattern = np.asarray(gripper.contact_pattern, dtype=np.float32)
    if contact_pattern.ndim != 2 or contact_pattern.shape != (1, 3):
        raise ValueError(
            'antipodal requires gripper.contact_pattern to be (1, 3)'
        )
    open_dir = gripper.open_dir / (np.linalg.norm(gripper.open_dir) + wum.eps)
    contact_depth = abs(float(contact_pattern[0] @ open_dir))
    jaw_width = jaw_width - 2.0 * contact_depth
    jaw_min, jaw_max = gripper.jaw_range
    jaw_mid = 0.5 * (jaw_min + jaw_max)
    jaw_span = jaw_max - jaw_min + wum.eps
    mask = ((cos_vals >= normal_cos_th) &
            (jaw_width >= jaw_min) &
            (jaw_width <= jaw_max))
    if not np.any(mask):
        return []
    center_sel = center[mask]
    jaw_sel = jaw_width[mask]
    n_sel = n_all[mask]
    nq_sel = nq_all[mask]
    ray_dirs = -n_sel
    # per-candidate opening axis when the gripper provides one (e.g. a curling
    # pinch whose pad-opposition axis depends on the closure width); else the
    # single constant open_dir (a true parallel jaw).
    if hasattr(gripper, 'open_dir_at'):
        open_dir_sel = np.asarray(gripper.open_dir_at(jaw_sel), dtype=np.float32)
    else:
        open_dir_sel = open_dir
    rot_base = build_grasp_rotmat_batch(ray_dirs, open_dir_sel)
    angles = np.arange(0.0, 2 * np.pi, roll_step)
    if open_dir_sel.ndim == 2:
        roll_axes = np.einsum('nij,nj->ni', rot_base, open_dir_sel)
    else:
        roll_axes = np.einsum('nij,j->ni', rot_base, open_dir_sel)
    roll_rots = wum.rotmat_from_axangle(roll_axes[:, None, :], angles[None, :])
    rot_all = roll_rots @ rot_base[:, None, :, :]
    pose_tf = np.tile(np.eye(4, dtype=np.float32),
                      (rot_all.shape[0], rot_all.shape[1], 1, 1))
    pose_tf[:, :, :3, :3] = rot_all
    pose_tf[:, :, :3, 3] = center_sel[:, None, :]
    pose_all = pose_tf.reshape(-1, 4, 4)
    jaw_all = np.repeat(jaw_sel, len(angles))
    normal_align = (1.0 + np.einsum("ij,ij->i", n_sel, -nq_sel) /
                    (np.linalg.norm(n_sel, axis=1) *
                     np.linalg.norm(nq_sel, axis=1) + wum.eps)) * 0.5
    jaw_close = 1.0 - np.abs(jaw_sel - jaw_mid) / jaw_span
    score = score_weights[0] * normal_align + score_weights[1] * jaw_close
    score_all = np.repeat(score, len(angles))
    order = np.argsort(score_all)[::-1]
    pose_all = pose_all[order]
    jaw_all = jaw_all[order]
    score_all = score_all[order]
    # prepare collision batch
    detector, batch = wgc.build_ee_target_detector(gripper, target_sobj)
    # retreat distance (grasp-center tcp offset; a JawView's shifts with the
    # closure, so take it at the widest opening)
    tcp_len = np.linalg.norm(
        gripper.grasp_center_tcp(gripper.jaw_range[1]).loc_tf[:3, 3])
    retreat_dist = 0.5 * tcp_len
    # Pose the gripper at every candidate first and snapshot the link
    # transforms, then check the whole list in one dispatch. A collision call
    # costs a fixed ~0.4 ms of command-buffer work whatever it computes, so
    # checking candidates one at a time pays that per candidate; only the
    # transforms are per-pose, so the stack below is a few hundred bytes each.
    pre_jw_all = jaw_all + pre_open * (jaw_max - jaw_all)
    pre_pose_all = pose_all.copy()
    pre_pose_all[:, :3, 3] -= retreat_dist * pose_all[:, :3, 2]
    tfs = np.empty((2 * len(pose_all), batch.n_items, 4, 4), dtype=np.float32)
    for i, (pose, pre_pose, jw, pre_jw) in enumerate(
            zip(pose_all, pre_pose_all, jaw_all, pre_jw_all)):
        gripper.grip_at(pose[:3, 3], pose[:3, :3], jw)
        tfs[2 * i] = batch.snapshot_transforms()
        # the pre-grasp pose is checked with the jaw OPENED PART-WAY toward its
        # max (pre_open in [0, 1]): on approach the fingers should clear the
        # object, not already be closed to the grasp width. 0 -> same as grasp,
        # 1 -> full open. The check thus reflects the wider hand actually swept
        # in.
        gripper.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], pre_jw)
        tfs[2 * i + 1] = batch.snapshot_transforms()
    hit = detector.detect_collision_multi(batch, tfs)
    collided_all = hit[0::2] | hit[1::2]
    for pose, pre_pose, jw, pre_jw, sc, collided in zip(
            pose_all, pre_pose_all, jaw_all, pre_jw_all, score_all,
            collided_all):
        # from_jaw reads only the gripper's local tcp and qpos, so it does not
        # need the gripper still standing at this candidate
        grasp = Grasp.from_jaw(gripper, pose, pre_pose, jw, pre_jw, float(sc))
        yield grasp, bool(collided)


def antipodal(gripper, target_sobj,
              density=0.02, normal_tol_deg=20,
              roll_step_deg=30, clearance=0.002,
              score_weights=(0.7, 0.3),
              exclude_regions=None, pre_open=0.5):
    """
    Collects non-colliding grasps only.
    :param gripper: gripper instance
    :param target_sobj: target object to grasp
    :param density: surface sampling density
    :param normal_tol_deg: normal tolerance in degrees
    :param roll_step_deg: roll angle step in degrees
    :param clearance: additional clearance for jaw width
    :param score_weights: (normal_align_weight, jaw_close_weight)
    :param exclude_regions: optional list of convex regions to carve
        out of the contact-sampling surface (forwarded to
        antipodal_iter). The full mesh is still used for collision
        checks.
    :param pre_open: jaw opening at the pre-grasp pose as a fraction of
        the room to max (default 0.5 = half-open); forwarded to
        antipodal_iter.
    :return: list of :class:`~wrs.grasp.grasp.Grasp` for the collision-free
        grasps, best score first, in the target's LOCAL frame. Each freezes the
        grasp-center (tcp) frame, its pre-grasp frame, the tcp loc_tf and the
        hand qpos at both openings. See antipodal_iter for the field detail.

        Every candidate is checked, so slice the result for a shorter list --
        it is score-sorted, so ``antipodal(...)[:n]`` is the best n. To spend
        less time, sample fewer candidates (``density`` / ``roll_step_deg``).
    """
    return [grasp for grasp, collided in antipodal_iter(
        gripper, target_sobj,
        density, normal_tol_deg, roll_step_deg,
        clearance, score_weights,
        exclude_regions=exclude_regions, pre_open=pre_open)
        if not collided]
