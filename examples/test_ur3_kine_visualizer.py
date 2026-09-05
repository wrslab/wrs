import numpy as np

import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.viewer.world as wvw
import wrs.scene.scene_object_primitive as wssop
import wrs.robots.base.kine_visualizer as wrbkv
import wrs.robots.manipulators.universal_robots.ur3.ur3 as wrmuu3
import wrs.robots.manipulators.kawasaki.rs007l.rs007l as wrmkr7


if __name__ == '__main__':
    base = wvw.World(cam_pos=(1.6, 1.2, 1.1), cam_lookat_pos=(0.0, 0.0, 0.3))
    scene = base.scene
    wssop.frame().add_to_scene(scene)

    robot = wrmuu3.UR3()
    # robot = wrmkr7.RS007L()
    robot.add_to_scene(scene)
    robot.alpha = 0.3

    # qs = np.array([0.6, -1.1, 1.2, -0.9, 0.8, 0.3], dtype=np.float32)
    # robot.fk(qs=qs)

    jviz = wrbkv.KineVisualizer(
        robot, chain=robot.chain('main'))
    jviz.add_to_scene(scene)
    
    wssop.frame(pos=robot.tcp('flange').tf[:3, 3], rotmat=robot.tcp('flange').tf[:3, :3],
                color_mat=wuc.CoordColor.MYC).add_to_scene(scene)
    base.run()

    tgt_pos = (0.35, -0.2, 0.35)
    tgt_rotmat = (wum.rotmat_from_axangle(wuc.StandardAxis.Z, np.pi / 6.0) @
                  wum.rotmat_from_axangle(wuc.StandardAxis.Y, np.pi))
    wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat).add_to_scene(scene)

    base.run()