from wrs import wvw, xyt, wssop, wuc

base = wvw.World(cam_pos=(3, 1, 2), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
oframe = wssop.frame()
oframe.add_to_scene(base.scene)
xyt_bot = xyt.XYThetaRobot()
xyt_bot.rgb=wuc.ExtendedColor.LAWN_GREEN
xyt_bot.add_to_scene(base.scene)
base.run()