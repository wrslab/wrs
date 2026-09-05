import builtins
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

lft_gripper = wreeogog.OAGripper()
lft_gripper.add_to_scene(base.scene)

rgt_gripper = wreeogog.OAGripper()
rgt_gripper.add_to_scene(base.scene)

robot.rgt_arm.mount(rgt_gripper, robot.rgt_arm.runtime_lnks[-1], update=True)
robot.lft_arm.mount(lft_gripper, robot.lft_arm.runtime_lnks[-1], update=True)
robot.body.alpha = 0.3

tgt_pos = (0.4, 0.2, 0.4)
tgt_rotmat = (wum.rotmat_from_axangle(wuc.StandardAxis.X, wum.pi / 2) @
              wum.rotmat_from_axangle(wuc.StandardAxis.Y, wum.pi / 2))
# a circle from around tgt_pos and in the plane dtermined by wuc.StandardAxis.X
tgt_pos_list = []
radius = 0.1
num_points = 8
for i in range(num_points):
    angle = (2 * wum.pi / num_points) * i
    x = tgt_pos[0]
    y = tgt_pos[1] + radius * wum.cos(angle)
    z = tgt_pos[2] + radius * wum.sin(angle)
    tgt_pos_list.append((x, y, z))
    wssop.frame(pos=(x, y, z), rotmat=tgt_rotmat).add_to_scene(base.scene)

qs_list = robot.lft_arm.ik(tgt_pos, tgt_rotmat, tcp=lft_gripper.tcp('grasp_center'))
tcp_tf = robot.lft_arm.tcp('flange').tf
tcp_frame = wssop.frame(rotmat=tcp_tf[:3, :3], pos=tcp_tf[:3, 3],
                        color_mat=wuc.CoordColor.MYC)
tcp_frame.add_to_scene(base.scene)
for qs in qs_list:
    tmp_lft_arm = robot.lft_arm.clone()
    tmp_lft_arm.fk(qs=qs)
    tmp_lft_arm.add_to_scene(base.scene)
base.run()
