import builtins
import numpy as np
from wrs import wuc, wum, wvw, wssop
import wrs.robots.end_effectors.openarm_gripper.oa_gripper as wreeogog
import wrs.robots.manipulators.openarm.openarm as wrmoo

base = wvw.World(cam_pos=(1.2, .575, 1.2), cam_lookat_pos=(0, 0, .4),
                 toggle_auto_cam_orbit=True)
oframe = wssop.frame().add_to_scene(base.scene)
robot = wrmoo.OpenArm()
robot.add_to_scene(base.scene)
builtins.robot = robot  # for debug access
builtins.base = base
robot.rgt_arm.fk((0, 0, 0, 0, 0, 0, 0))
robot.lft_arm.fk((0, 0, 0, 0, 0, 0, 0))
lft_gripper = wreeogog.OAGripper()
rgt_gripper = wreeogog.OAGripper()
lft_gripper.add_to_scene(base.scene)
rgt_gripper.add_to_scene(base.scene)
robot.rgt_arm.mount(rgt_gripper, robot.rgt_arm.runtime_lnks[-1], update=True)
robot.lft_arm.mount(lft_gripper, robot.lft_arm.runtime_lnks[-1], update=True)
lft_tcp_tf = robot.lft_arm.tcp('flange').tf
lft_tcp_frame = wssop.frame(rotmat=lft_tcp_tf[:3, :3], pos=lft_tcp_tf[:3, 3],
                            color_mat=wuc.CoordColor.MYC)
lft_tcp_frame.add_to_scene(base.scene)
rgt_tcp_tf = robot.rgt_arm.tcp('flange').tf
rgt_tcp_frame = wssop.frame(rotmat=rgt_tcp_tf[:3, :3], pos=rgt_tcp_tf[:3, 3],
                            color_mat=wuc.CoordColor.MYC)
rgt_tcp_frame.add_to_scene(base.scene)
# robot.body.alpha=0.3
# base.run()

tgt_pos = (0.4, 0.1, 0.4)
# tgt_rotmat = (wum.rotmat_from_axangle(wuc.StandardAxis.X, wum.pi / 2) @
#               wum.rotmat_from_axangle(wuc.StandardAxis.Y, wum.pi / 2))
tgt_rotmat = (wum.rotmat_from_axangle(wuc.StandardAxis.Z, 0) @
              wum.rotmat_from_axangle(wuc.StandardAxis.Y, wum.pi))
wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat).add_to_scene(base.scene)

prev_qs = robot.lft_arm.qs.copy()
for y in range(1, 5):
    for z in range(1, 5):
        tgt_pos = (0.2, y * 0.1, z * 0.1)
        _s = robot.lft_arm.ik(
            tgt_pos, tgt_rotmat, max_solutions=1,
            ref_qs=robot.lft_arm.chain('main').extract_active_qs(
                np.asarray(prev_qs, dtype=np.float32)))
        qs = _s[0] if _s else None
        if qs is not None:
            prev_qs = qs
            tmp_lft_arm = robot.lft_arm.clone()
            tmp_lft_arm.fk(qs=qs)
            tmp_lft_arm.add_to_scene(base.scene)
base.run()
