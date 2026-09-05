import builtins
import numpy as np

import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.viewer.world as wvw
import wrs.scene.scene_object_primitive as wssop
import wrs.geom.geometry as wgg
import wrs.robots.manipulators.kawasaki.rs007l.rs007l as wrmkr7
import wrs.robots.end_effectors.onrobot.or_2fg7.or_2fg7 as wreo2fg7


if __name__ == "__main__":
    base = wvw.World(cam_pos=(1.8, 1.2, 1.4), cam_lookat_pos=(0.0, 0.0, 0.5))
    builtins.base = base
    scene = base.scene
    wssop.frame().add_to_scene(scene)

    robot = wrmkr7.RS007L(rotmat=wum.rotmat_from_euler(0, 0, -np.pi / 2))
    robot.add_to_scene(scene)
    builtins.robot = robot

    gripper = wreo2fg7.OR2FG7()
    gripper.set_opening(0.03)
    gripper.add_to_scene(scene)

    # loc_tf is flange->ee_base transform
    loc_tf = wum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotmat=np.eye(3, dtype=np.float32),
    )
    robot.mount(gripper, robot.runtime_lnks[-1], loc_tf, update=True)

    tgt_pos = np.array([0.55, 0.10, 0.35], dtype=np.float32)
    ico_geom = wgg.gen_icosphere_geom(radius=1.0, n_subs=1)
    dirs = ico_geom.vs.copy()
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + wum.eps)

    n_success = 0
    for i, d in enumerate(dirs):
        # Use each icosphere vertex direction as target TCP +Z direction.
        tgt_rotmat = wum.rotmat_from_normal(d)
        wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat, color_mat=wuc.CoordColor.DYO, alpha=0.2).add_to_scene(scene)
        _s = robot.ik(tgt_pos, tgt_rotmat, tcp=gripper.tcp('grasp_center'), max_solutions=1)
        qs = _s[0] if _s else None
        if qs is None:
            continue
        n_success += 1
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.add_to_scene(base.scene)
        tmp_robot.alpha = .3
        wssop.frame(
            pos=tmp_robot.tcp('flange').tf[:3, 3],
            rotmat=tmp_robot.tcp('flange').tf[:3, :3],
            color_mat=wuc.CoordColor.MYC,
        ).add_to_scene(scene)
    print(f"IK success: {n_success}/{len(dirs)}")

    base.run()
