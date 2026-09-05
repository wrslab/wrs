"""Monocontact: single-contact ("one contact pad") surface-approach
planning for suction / tip tools.

Naming convention in this package -- the suffix encodes the *mechanism*,
the prefix the *contact count*:

    -podal   : opposing pinch, force-closure (the pads press against
               each other).  antipodal (2), polypodal (N).
    -contact : same-side adhesion / press, NOT force-closure (suction,
               magnetic, a tip pressing or inserting).  monocontact (1),
               and a future polycontact (N, e.g. a suction-cup array).

A monocontact grasp therefore has a single contact and no opposition --
the tool is held against the surface by suction / adhesion, or simply
presses on / inserts into it. The planner samples the target surface
and, for each sampled point, aligns the tool's contact axis (the tcp
local +z) with the inward surface normal, optionally rolls about that
axis, then rejects tool-vs-target collisions. It needs only a tcp to
align and ``runtime_lnks`` for collision, so it works for any
single-contact end effector (suction cup, screwdriver tip, probe, ...).

Outputs are (pose_4x4, pre_pose_4x4, score) tuples; ``pose`` places the
tcp origin at the contact point with +z pointing into the surface, and
``pre_pose`` is the same pose retreated along the approach axis.
"""
import numpy as np

import wrs.utils.math as wum
import wrs.scene.geometry_ops as wsgop
import wrs.collider.cpu_simd as wccs
import wrs.grasp._common as wgc
from wrs.grasp.grasp import Grasp


def monocontact_iter(tool, target_sobj, tcp='tip',
                     density=0.02, roll_step_deg=90, retreat=None,
                     approach_bias=(0.0, 0.0, 1.0), exclude_regions=None):
    """
    Generator: yields (grasp, collided) -- a Grasp and its collision flag.
    :param tool: single-contact end effector (must expose ``tcp(name)``,
        ``runtime_lnks`` and ``set_pos_rotmat``)
    :param target_sobj: target object to contact
    :param tcp: name of the contact tcp to align (default 'tip')
    :param density: surface sampling density (smaller -> denser)
    :param roll_step_deg: roll step about the approach axis, in degrees.
        For an axisymmetric suction cup one roll is enough; finer steps
        only matter for an asymmetric tool body's collisions.
    :param retreat: pre-pose retreat distance along the approach axis.
        Defaults to half the tcp offset length.
    :param approach_bias: world direction favoured by the score; the
        default world +z rewards top-facing surfaces (a suction seal
        approached from above). Set to None to score all contacts equally.
    :param exclude_regions: optional convex regions carved out of the
        contact-sampling surface (full mesh still used for collisions).
    :return: yields (grasp, collided), where ``grasp`` is a
        :class:`~wrs.grasp.grasp.Grasp` in the target's LOCAL frame: ``pose``
        aligns the tcp +z into the surface at the contact point, ``pre_pose`` is
        that pose retreated along the approach axis, ``tcp`` is the named contact
        tcp's loc_tf, and ``qpos`` / ``pre_qpos`` are the tool's (fixed) config.
        ``collided`` is True if the tool collides at either pose.
    """
    tool = tool.clone()
    tcp_loc = np.asarray(tool.tcp(tcp).loc_tf, dtype=np.float32)
    if retreat is None:
        retreat = 0.5 * float(np.linalg.norm(tcp_loc[:3, 3]))
    tcp_loc_inv = np.linalg.inv(tcp_loc)

    # Plan in the target's LOCAL (zero-pose) frame: clone the target and zero
    # its pose so contact sampling (local geom) and the tool-vs-target collision
    # check (CollisionBatch uses target.tf) share one frame. Returned poses are
    # in the target's local frame; the caller maps them onto the placed object.
    target_sobj = target_sobj.clone()
    target_sobj.set_pos_rotmat(
        pos=np.zeros(3, dtype=np.float32), rotmat=np.eye(3, dtype=np.float32))
    tgt_vs, tgt_fs, _ = wccs.cols_to_vffns(target_sobj.collisions)
    if exclude_regions:
        tgt_vs, tgt_fs = wsgop.clip_mesh(tgt_vs, tgt_fs, exclude_regions)
        if len(tgt_fs) == 0:
            return
    n_samples = wsgop.sample_count_from_area(tgt_vs, tgt_fs, density)
    pts, nrms, _ = wsgop.sample_surface(tgt_vs, tgt_fs, n_samples)
    nrms = nrms / (np.linalg.norm(nrms, axis=1, keepdims=True) + wum.eps)

    # approach (tool +z) points into the surface, opposite the outward normal
    approach = -nrms
    rot_base = wum.frame_from_normal(approach)                  # (N,3,3), z=approach
    roll_step = np.deg2rad(roll_step_deg)
    angles = np.arange(0.0, 2.0 * np.pi, roll_step)
    roll_rots = wum.rotmat_from_axangle(
        approach[:, None, :], angles[None, :])                 # (N,K,3,3)
    rot_all = roll_rots @ rot_base[:, None, :, :]               # (N,K,3,3)

    pose_tf = np.tile(np.eye(4, dtype=np.float32),
                      (rot_all.shape[0], rot_all.shape[1], 1, 1))
    pose_tf[:, :, :3, :3] = rot_all
    pose_tf[:, :, :3, 3] = pts[:, None, :]
    pose_all = pose_tf.reshape(-1, 4, 4)

    if approach_bias is None:
        score = np.zeros(len(nrms), dtype=np.float32)
    else:
        bias = np.asarray(approach_bias, dtype=np.float32)
        bias = bias / (np.linalg.norm(bias) + wum.eps)
        score = 0.5 * (1.0 + nrms @ bias)        # top-facing -> 1, down -> 0
    score_all = np.repeat(score, len(angles))
    order = np.argsort(score_all)[::-1]
    pose_all = pose_all[order]
    score_all = score_all[order]

    # tool-vs-target collision batch
    detector, batch = wgc.build_ee_target_detector(tool, target_sobj)

    # Place the tool at every candidate first and snapshot the link transforms,
    # then check the whole list in one dispatch. A collision call costs a fixed
    # ~0.4 ms of command-buffer work whatever it computes, so checking
    # candidates one at a time pays that per candidate.
    # pre-contact (retreated) pose: move back along +outward normal
    pre_pose_all = pose_all.copy()
    pre_pose_all[:, :3, 3] -= retreat * pose_all[:, :3, 2]
    tfs = np.empty((2 * len(pose_all), batch.n_items, 4, 4), dtype=np.float32)
    for i, (pose, pre_pose) in enumerate(zip(pose_all, pre_pose_all)):
        base_tf = pose @ tcp_loc_inv
        tool.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        tfs[2 * i] = batch.snapshot_transforms()
        pre_base = pre_pose @ tcp_loc_inv
        tool.set_pos_rotmat(pre_base[:3, 3], pre_base[:3, :3])
        tfs[2 * i + 1] = batch.snapshot_transforms()
    hit = detector.detect_collision_multi(batch, tfs)
    collided_all = hit[0::2] | hit[1::2]
    for pose, pre_pose, sc, collided in zip(
            pose_all, pre_pose_all, score_all, collided_all):
        # from_tool reads only the tool's local tcp and qs, so it does not need
        # the tool still standing at this candidate
        grasp = Grasp.from_tool(tool, pose, pre_pose, tcp_name=tcp,
                                score=float(sc))
        yield grasp, bool(collided)


def monocontact(tool, target_sobj, tcp='tip',
                density=0.02, roll_step_deg=90, retreat=None,
                approach_bias=(0.0, 0.0, 1.0),
                exclude_regions=None):
    """
    Collects non-colliding single-contact grasps only.
    :param tool: single-contact end effector
    :param target_sobj: target object to contact
    :param tcp: name of the contact tcp to align (default 'tip')
    :param density: surface sampling density
    :param roll_step_deg: roll step about the approach axis, in degrees
    :param retreat: pre-pose retreat distance (defaults to half tcp length)
    :param approach_bias: world direction favoured by the score
    :param exclude_regions: convex regions carved out of contact sampling
    :return: list of :class:`~wrs.grasp.grasp.Grasp`, in the target's LOCAL
        frame, best score first.

        Every candidate is checked, so slice the result for a shorter list --
        it is score-sorted, so ``monocontact(...)[:n]`` is the best n. To spend
        less time, sample fewer candidates (``density`` / ``roll_step_deg``).
    """
    return [grasp for grasp, collided in monocontact_iter(
        tool, target_sobj, tcp, density, roll_step_deg,
        retreat, approach_bias, exclude_regions=exclude_regions)
        if not collided]
