"""Two Lite6 arms mounted on a shared base to form a dual-arm setup."""
import numpy as np

from wrs import wum, wuc, wvw, wssop, xarm_lite6


if __name__ == "__main__":
    base = wvw.World(cam_pos=(1.5, 1.5, 1.2),
                     cam_lookat_pos=(0.0, 0.0, 0.35))

    # shared torso/base plate at z=0
    plate = wssop.box(xyz_lengths=(0.5, 0.3, 0.02),
                      pos=(0.0, 0.0, 0.0),
                      rgb=wuc.BasicColor.GRAY)
    plate.add_to_scene(base.scene)

    # left arm: mounted on +Y side, tilted outward 30 deg around X
    left_rotmat = wum.rotmat_from_axangle(wuc.StandardAxis.X, np.deg2rad(-30))
    left = xarm_lite6.Lite6(pos=(0.0, 0.15, 0.01), rotmat=left_rotmat)
    left.add_to_scene(base.scene)

    # right arm: mounted on -Y side, mirror tilt
    right_rotmat = wum.rotmat_from_axangle(wuc.StandardAxis.X, np.deg2rad(30))
    right = xarm_lite6.Lite6(pos=(0.0, -0.15, 0.01), rotmat=right_rotmat)
    right.add_to_scene(base.scene)

    # simple "ready" pose for both
    ready = np.array([0.0, -0.5, 1.2, 0.0, 0.4, 0.0], dtype=np.float32)
    left.fk(qs=ready)
    right.fk(qs=ready)

    wssop.frame().add_to_scene(base.scene)
    base.run()
