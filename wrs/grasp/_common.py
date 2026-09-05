"""Shared grasp-domain helpers (depend on end-effector runtime links and
the collider backends, so they live in the grasp package rather than in
the generic geometry/math utilities)."""
import wrs.collider.gpu_simd_batch as wcgsb


def build_ee_target_detector(ee, target_sobj):
    """Collision detector + batch checking every link of an end effector
    against a target object.

    Runs on the wgpu compute backend, which needs no window and splits
    oversized dispatches into chunks -- so unlike the old GL path there is no
    case left where this has to degrade to the CPU collider.

    :param ee: end effector exposing ``runtime_lnks``
    :param target_sobj: the object the ee is tested against
    :return: (detector, batch); a placement collides when
        ``detector.detect_collision_batch(batch) is not None``.
    """
    items = ee.runtime_lnks + [target_sobj]
    tgt_idx = len(items) - 1
    pairs = [(i, tgt_idx) for i in range(len(ee.runtime_lnks))]
    detector = wcgsb.create_detector()
    batch = wcgsb.build_batch(items, pairs)
    return detector, batch
