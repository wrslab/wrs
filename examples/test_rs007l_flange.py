if __name__ == '__main__':
    from wrs import wvw, wssop, wsso

    base = wvw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)
    wsso.SceneObject.from_file("link6.stl").add_to_scene(base.scene)
    wssop.frame(length_scale=.3, radius_scale=.3).add_to_scene(base.scene)
    base.run()