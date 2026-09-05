import builtins
import numpy as np
import wrs.physics.mj_env as wpme
from wrs import wum, wvw, wssop, wuc, wsso, khi_rs007l

oframe = wssop.frame()
bunny = wsso.SceneObject.from_file(
    "bunny.stl", collision_type=wuc.CollisionType.MESH,
    is_floating=True)
bunny.rgb = wuc.ExtendedColor.PINK
bunny.alpha = 0.4
# bunny.toggle_render_collision = True
base = wvw.World(cam_pos=(3.5, 1, 3.5),
                 cam_lookat_pos=(0, 0, .5),
                 toggle_auto_cam_orbit=False)
setattr(builtins, "base", base)
oframe.add_to_scene(base.scene)
bunny.add_to_scene(base.scene)
for i in np.linspace(5.5, 15.5, 1):
    tmp_bunny = bunny.clone()
    tmp_bunny.pos = (.5, 0, i)
    tmp_bunny.add_to_scene(base.scene)
# container
plane_bottom = wssop.plane()
plane_bottom.toggle_render_collision = True
plane_bottom.add_to_scene(base.scene)
wall_left = wssop.box(pos=(.5, .35, .2),
                      xyz_lengths=(0.71, 0.01, 0.4),
                      collision_type=wuc.CollisionType.AABB,
                      alpha=wuc.ALPHA.TRANSPARENT,
                      is_floating=False)
wall_left.add_to_scene(base.scene)
wall_right = wssop.box(pos=(.5, -.35, .2),
                       xyz_lengths=(0.71, 0.01, 0.4),
                       collision_type=wuc.CollisionType.AABB,
                       alpha=wuc.ALPHA.TRANSPARENT,
                       is_floating=False)
wall_right.add_to_scene(base.scene)
wall_front = wssop.box(pos=(.85, 0, .2),
                       xyz_lengths=(0.01, 0.71, 0.4),
                       collision_type=wuc.CollisionType.AABB,
                       alpha=wuc.ALPHA.TRANSPARENT,
                       is_floating=False)
wall_front.add_to_scene(base.scene)
wall_back = wssop.box(pos=(.15, 0, .2),
                      xyz_lengths=(0.01, 0.71, 0.4),
                      collision_type=wuc.CollisionType.AABB,
                      alpha=wuc.ALPHA.TRANSPARENT,
                      is_floating=False)
wall_back.add_to_scene(base.scene)

# robot
base_rotmat = wum.rotmat_from_euler(0, 0, -np.pi / 2)
base_pos = np.array([0, 0.7, 0])
robot1 = khi_rs007l.RS007L()
robot1.is_floating=True
robot1.add_to_scene(base.scene)
robot1.set_pos_rotmat(pos=base_pos, rotmat=base_rotmat)
robot1.toggle_render_collision = True
robot1.fk(qs=[0, 0, -np.pi / 4, 0, 0, 0])
robot1.alpha = 0.1

mjenv = wpme.MJEnv(scene=base.scene)
# mjenv.sync.push_qpos()
# mjenv.sync_mechstates_to_mujoco()
mjenv.save("scene.xml")
base.schedule_interval(mjenv.step)
base.run()
