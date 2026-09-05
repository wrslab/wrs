import builtins
import numpy as np

import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.viewer.world as wvw
import wrs.scene.scene_object_primitive as wssop
import wrs.robots.manipulators.universal_robots.ur3.ur3 as wrmuu3
import wrs.robots.end_effectors.onrobot.or_2fg7.or_2fg7 as wreo2fg7
import wrs.robots.end_effectors.openarm_gripper.oa_gripper as wreeogog


if __name__ == "__main__":
    base = wvw.World(cam_pos=(1.6, 1.0, 1.4), cam_lookat_pos=(0.0, 0.0, 0.45))
    builtins.base = base
    scene = base.scene
    wssop.frame().add_to_scene(scene)

    robot = wrmuu3.UR3()
    robot.add_to_scene(scene)
    builtins.robot = robot

    # gripper = wreo2fg7.OR2FG7()
    gripper = wreeogog.OAGripper()
    gripper.set_opening(0.03)
    gripper.add_to_scene(scene)

    # loc_tf is flange->ee_base transform
    loc_tf = wum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotmat=np.eye(3, dtype=np.float32),
    )
    robot.mount(gripper, robot.runtime_lnks[-1], loc_tf, update=True)

    tgt_pos = np.array([0.35, -0.20, 0.2], dtype=np.float32)
    tgt_rotmat = (
        wum.rotmat_from_axangle(wuc.StandardAxis.Z, np.pi / 6.0)
        @ wum.rotmat_from_axangle(wuc.StandardAxis.Y, np.pi)
    )
    wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat, color_mat=wuc.CoordColor.DYO).add_to_scene(scene)

    _s = robot.ik(tgt_pos, tgt_rotmat, tcp=gripper.tcp('grasp_center'), max_solutions=1)
    qs = _s[0] if _s else None
    print("ik:", qs)
    if qs is not None:
        robot.fk(qs=qs)
        wssop.frame(
            pos=robot.tcp('flange').tf[:3, 3],
            rotmat=robot.tcp('flange').tf[:3, :3],
            color_mat=wuc.CoordColor.MYC,
        ).add_to_scene(scene)

    base.run()
