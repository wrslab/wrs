import numpy as np

import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.viewer.world as wvw
import wrs.scene.scene_object_primitive as wssop
import wrs.robots.base.kine_visualizer as wrbkv
import wrs.robots.manipulators.denso.cvr038.cvr038 as wrmdc


if __name__ == '__main__':
    base = wvw.World(cam_pos=(1.6, 1.2, 1.1), cam_lookat_pos=(0.0, 0.0, 0.3))
    scene = base.scene
    wssop.frame().add_to_scene(scene)

    robot = wrmdc.CVR038()
    robot.add_to_scene(scene)
    robot.alpha = 0.3

    # qs = np.array([0.2, -0.6, 1.0, -0.8, 0.7, 0.3], dtype=np.float32)
    # robot.fk(qs=qs)

    jviz = wrbkv.KineVisualizer(robot, chain=robot.chain('main'))
    jviz.add_to_scene(scene)

    flange_tf = robot.tcp('flange').tf
    wssop.frame(pos=flange_tf[:3, 3], rotmat=flange_tf[:3, :3],
                color_mat=wuc.CoordColor.MYC).add_to_scene(scene)
    base.run()

    tgt_pos = (0.25, 0.15, 0.25)
    tgt_rotmat = (wum.rotmat_from_axangle(wuc.StandardAxis.Z, np.pi / 6.0) @
                  wum.rotmat_from_axangle(wuc.StandardAxis.Y, np.pi))
    wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat).add_to_scene(scene)

    base.run()