from wrs import wum, wvw, wuc, wssop, wsso

base = wvw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)

oframe = wssop.frame()
bunny = wsso.SceneObject.from_file("bunny.stl", collision_type=wuc.CollisionType.CAPSULE)
bunny.toggle_render_collision = True
oframe.add_to_scene(base.scene)
bunny.add_to_scene(base.scene)

bunny2 = bunny.clone()
bunny2.pos = bunny.pos + wum.np.array([0, 0.1, 0])
bunny2.add_to_scene(base.scene)
base.run()
