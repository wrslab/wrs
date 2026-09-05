import numpy as np
import wrs.physics.mj_env as mj
from wrs import wum, wvw, wssop, wuc, wsso, khi_rs007l

base = wvw.World(cam_pos=(3.5, 1, 3.5),
                 cam_lookat_pos=(0, 0, .5))

robot = khi_rs007l.RS007L()
robot.add_to_scene(base.scene)
robot.toggle_render_collision = True
robot.set_pos_rotmat(pos=(0, 0, 1))
robot.fk(qs=[0, 0, -np.pi / 4, 0, 0, 0])

for i in range(1, 15):
    tmp_robot = robot.clone()
    tmp_robot.is_floating = True
    tmp_robot.set_pos_rotmat(pos=(0, 0, 1 + i * 1.5))
    tmp_robot.add_to_scene(base.scene)

plane_bottom = wssop.plane()
plane_bottom.toggle_render_collision = False
plane_bottom.add_to_scene(base.scene)

mjenv = mj.MJEnv(scene=base.scene)
# mjenv.sync_mechstates_to_mujoco()
mjenv.save("scene.xml")
base.schedule_interval(mjenv.step)
base.run()
