import numpy as np
from wrs import wum, wvw, wssop, khi_rs007l

base = wvw.World(cam_pos=(1.6, .3, .7), cam_lookat_pos=(0, 0, .45),
                toggle_auto_cam_orbit=False)
# world origin
oframe = wssop.frame().add_to_scene(base.scene)
base_pos1 = np.array([0, 0.5, 0])
base_rotmat = wum.rotmat_from_euler(0, 0, -np.pi / 2)
# robot 1 (left robot)
robot1 = khi_rs007l.RS007L()
robot1.add_to_scene(base.scene)
robot1.set_pos_rotmat(pos=base_pos1, rotmat=base_rotmat)
robot1.toggle_render_collision = True
# robot 2 (right robot)
robot2 = robot1.clone()
base_pos2 = np.array([0, -0.5, 0])
robot2.add_to_scene(base.scene)
robot2.set_pos_rotmat(pos=base_pos2, rotmat=base_rotmat)
robot2.toggle_render_collision = False
# goal1
tgt1_rotmat = wum.rotmat_from_euler(-wum.pi / 2, 0, 0)
tgt1_pos = np.array([0.3, 0, 0.5])
g1frame = wssop.frame(rotmat=tgt1_rotmat, pos=tgt1_pos)
g1frame.add_to_scene(base.scene)
qs1_list = robot1.ik(tgt1_pos, tgt1_rotmat)
for qs in qs1_list:
    tmp_robot = robot1.clone()
    tmp_robot.fk(qs=qs)
    tmp_robot.add_to_scene(base.scene)
# goal2
tgt2_rotmat = wum.rotmat_from_euler(wum.pi / 2, 0, 0)
tgt2_pos = np.array([0.3, 0, 0.5])
g2frame = wssop.frame(rotmat=tgt2_rotmat, pos=tgt2_pos)
g2frame.add_to_scene(base.scene)
qs2_list = robot2.ik(tgt2_pos, tgt2_rotmat)
for qs in qs2_list:
    tmp_robot = robot2.clone()
    tmp_robot.fk(qs=qs)
    tmp_robot.add_to_scene(base.scene)
base.run()
