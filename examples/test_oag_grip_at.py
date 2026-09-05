import numpy as np
import wrs.utils.math as wum
from wrs import wvw, wssop
from wrs.robots.end_effectors.openarm_gripper.oa_gripper import OAGripper

base = wvw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
wssop.frame().add_to_scene(base.scene)

gripper = OAGripper()
gripper.add_to_scene(base.scene)

# target pose (example)
tgt_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
tgt_rotmat = wum.rotmat_from_euler(
    np.deg2rad(-45.0), 0.0, np.deg2rad(30.0))
tgt_jw = 0.04  # within [0.0, 0.088]

base_tf = gripper.grip_at(tgt_pos, tgt_rotmat, tgt_jw)

# optional: draw frames
wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat).add_to_scene(base.scene)
wssop.frame(pos=base_tf[:3, 3],
            rotmat=base_tf[:3, :3]).add_to_scene(base.scene)

base.run()