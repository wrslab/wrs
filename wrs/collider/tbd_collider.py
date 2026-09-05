import wrs.utils.constant as wuc
import wrs.collider.cpu_simd as wccs
import wrs.collider.gpu_simd_batch as wcgsb


def is_collided(sobj_a, sobj_b, eps=1e-9, max_points=1000):
    """
    detect collision between two scene objects
    automatically selects the appropriate collision detection method:
    - single mesh collision: GPU SIMD (with CPU fallback)
    - other shapes: special collision detection (TODO)
    :param sobj_a, sobj_b: SceneObject instances
    :param eps: numerical tolerance (default 1e-9)
    :param max_points: max collision points for mesh detection (default 200)
    :return: (K,3) collision points or None
    """
    batch = wcgsb.build_batch((sobj_a, sobj_b), pairs=[(0,1)])
    if batch is None:
        return None
    mesh_types = (wuc.CollisionType.MESH, wuc.CollisionType.CVXHULL)
    is_single_mesh_a = (len(sobj_a.collisions) == 1 and
                        sobj_a._collision_type in mesh_types)
    is_single_mesh_b = (len(sobj_b.collisions) == 1 and
                        sobj_b._collision_type in mesh_types)
    if is_single_mesh_a and is_single_mesh_b:
        col_a = sobj_a.collisions[0]
        col_b = sobj_b.collisions[0]
        try:
            return wcgsb.detect_collision(
                col_a, sobj_a.tf, col_b, sobj_b.tf,
                eps=eps, max_points=max_points)
        except Exception as e:
            import sys
            print(f"GPU collision detection failed: {e}", file=sys.stderr)
            print("Falling back to CPU collision detection...", file=sys.stderr)
            return wccs.detect_collision(
                col_a, sobj_a.tf, col_b, sobj_b.tf, eps=eps)
    else:
        raise NotImplementedError(
            f"Collision detection not yet implemented for:\n"
            f"  Object A: {len(sobj_a.collisions)} collision(s), "
            f"type={sobj_a._collision_type}\n"
            f"  Object B: {len(sobj_b.collisions)} collision(s), "
            f"type={sobj_b._collision_type}\n"
            f"Only single mesh collision is currently supported.")
