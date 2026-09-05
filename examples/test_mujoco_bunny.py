import cProfile
import builtins
import numpy as np
import wrs.physics.mj_env as wpme
from wrs import wvw, wssop, wuc, wsso

oframe = wssop.frame()
bunny = wsso.SceneObject.from_file(
    "bunny.stl", collision_type=wuc.CollisionType.MESH,
    is_floating=True)
bunny.rgb = wuc.ExtendedColor.PINK
base = wvw.World(cam_pos=(.75, 1.5, 1.5),
                 toggle_auto_cam_orbit=False)
builtins.base = base
oframe.add_to_scene(base.scene)
for i in np.linspace(.5, 10.5, 50):
    tmp_bunny = bunny.clone()
    tmp_bunny.pos = (0, 0, i)
    tmp_bunny.add_to_scene(base.scene)
plane_ground = wssop.plane()
plane_ground.add_to_scene(base.scene)
wall_left = wssop.box(name="wall", pos=(.0, .35, .2),
                      xyz_lengths=(0.71, 0.01, 0.4),
                      collision_type=wuc.CollisionType.AABB,
                      alpha=wuc.ALPHA.TRANSPARENT, is_floating=False)
wall_left.add_to_scene(base.scene)
wall_right = wssop.box(name="wall", pos=(.0, -.35, .2),
                       xyz_lengths=(0.71, 0.01, 0.4),
                       collision_type=wuc.CollisionType.AABB,
                       alpha=wuc.ALPHA.TRANSPARENT, is_floating=False)
wall_right.add_to_scene(base.scene)
wall_front = wssop.box(name="wall", pos=(.35, 0, .2),
                       xyz_lengths=(0.01, 0.71, 0.4),
                       collision_type=wuc.CollisionType.AABB,
                       alpha=wuc.ALPHA.TRANSPARENT, is_floating=False)
wall_front.add_to_scene(base.scene)
wall_back = wssop.box(name="wall", pos=(-.35, 0, .2),
                      xyz_lengths=(0.01, 0.71, 0.4),
                      collision_type=wuc.CollisionType.AABB,
                      alpha=wuc.ALPHA.TRANSPARENT, is_floating=False)
wall_back.add_to_scene(base.scene)
mjenv = wpme.MJEnv(scene=base.scene)
base.schedule_interval(mjenv.step)
base.run()
