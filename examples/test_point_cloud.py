"""Point cloud alongside a mesh and a translucent collision volume -- one
scene per render pipeline the viewer has."""
import numpy as np

from wrs import wvw, wuc, wssop, wsso

base = wvw.World(cam_pos=(.3, .3, .3))

wssop.frame().add_to_scene(base.scene)

bunny = wsso.SceneObject.from_file("bunny.stl",
                                   collision_type=wuc.CollisionType.CAPSULE)
bunny.toggle_render_collision = True
bunny.add_to_scene(base.scene)

rng = np.random.default_rng(0)
dirs = rng.normal(size=(3000, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
shell = wssop.point_cloud(vs=(dirs * .11 + np.array([0, .05, .04])).astype(np.float32),
                          vrgbs=((dirs + 1.) * .5).astype(np.float32))
shell.add_to_scene(base.scene)

base.run()
