import numpy as np
from wrs import wum, wuc, wvw, wssop, khi_rs007l, or_2fg7

base = wvw.World(cam_pos=(2, 1, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)
wssop.frame().add_to_scene(base.scene)
robot = khi_rs007l.RS007L(pos=(.5, 0, 0))
robot.add_to_scene(base.scene)

tgt_pos = np.array([0, .5, .3])
tgt_rotmat = wum.rotmat_from_euler(wum.pi, 0, 0)
wssop.frame(rotmat=tgt_rotmat, pos=tgt_pos).add_to_scene(base.scene)
qs_list = robot.ik(tgt_pos, tgt_rotmat)
robot_ik = robot.clone()
robot_ik.rgb = wuc.BasicColor.LIME
robot_ik.fk(qs=qs_list[0])
robot_ik.add_to_scene(base.scene)
wd_tcp_rotmat = robot_ik.tcp('flange').tf[:3, :3]
wd_tcp_pos = robot_ik.tcp('flange').tf[:3, 3]
wssop.frame(rotmat=wd_tcp_rotmat, pos=wd_tcp_pos,
            color_mat=wuc.CoordColor.MYC).add_to_scene(base.scene)

robot2 = robot.clone()
robot2.set_pos_rotmat(pos=(-.5, 0, 0))
gripper = or_2fg7.OR2FG7()
robot2.mount(gripper, robot2.runtime_lnks[-1], update=True)
robot2.add_to_scene(base.scene)
qs2_list = robot2.ik(tgt_pos, tgt_rotmat, tcp=gripper.tcp('grasp_center'))

box = wssop.cylinder(spos=(-.3, 0, .3), epos=(.3, 0, .1), radius=.03)
box.add_to_scene(base.scene)
gripper.close()
gripper.hold(box)
gripper.open()
gripper.release(box)
# base.run()

robot2_ik = robot2.clone()
robot2_ik.rgb = wuc.BasicColor.YELLOW
robot2_ik.fk(qs=qs2_list[0])
robot2_ik.add_to_scene(base.scene)
base.run()

# # engage later
# robot.mount(gripper, robot.runtime_lnks[-1], update=True)
# robot.fk(qs=[0, -wum.pi / 4, 0, -wum.pi / 2, 0, wum.pi / 3])
# base.run()
