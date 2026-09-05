"""Inspect the L1 left-arm kinematic structure: stick (skeleton) model +
coordinate frames at every left-arm joint, with axis 1 and axis 2 drawn as
extended lines and their two closest points marked -- to eyeball why the
joint-1 / joint-2 axes miss each other by ~1.6 mm.

Robot is shown at ZERO config so the static chain origins/axes line up with
the meshes.
"""
import os
import sys
import builtins

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import wrs.scene.scene_object_primitive as wssop              # noqa: E402
import wrs.viewer.world as wvw                                # noqa: E402
import wrs.robots.base.kine_visualizer as wrbkv               # noqa: E402
import wrs.robots.humanoids.linx.l1.l1 as l1mod               # noqa: E402


def closest_points(o1, d1, o2, d2):
    """Closest points between line1 (o1,d1) and line2 (o2,d2)."""
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    r = o1 - o2
    b = float(d1 @ d2)
    d = float(d1 @ r)
    e = float(d2 @ r)
    denom = 1.0 - b * b
    if abs(denom) < 1e-9:          # parallel
        t, s = 0.0, e
    else:
        t = (b * e - d) / denom
        s = (e - b * d) / denom
    return o1 + t * d1, o2 + s * d2


def main():
    base = wvw.World(cam_pos=(0.9, 0.6, 1.4), cam_lookat_pos=(0.1, 0.15, 0.9))
    builtins.base = base
    wssop.frame().add_to_scene(base.scene)

    robot = l1mod.L1O6()
    robot.fk(qs=np.zeros(robot.qs.shape, dtype=np.float32))   # zero config
    robot.alpha = 0.2
    robot.left_hand.alpha = 0.2
    robot.right_hand.alpha = 0.2
    robot.add_to_scene(base.scene)

    # skeleton / stick model
    kv = wrbkv.KineVisualizer(robot, alpha=0.9)
    kv.add_to_scene(base.scene)

    # left-arm joint origins/axes in WORLD (chain origins are in the chain-base
    # frame = waist_link2; map them through that link's world tf).
    ch = robot.chain('left_arm')
    base_tf = robot.lnk('waist_link2').tf
    o_w, a_w = [], []
    for i in range(len(ch.jnts)):
        o = base_tf[:3, :3] @ np.asarray(ch.origins[i], np.float32) + base_tf[:3, 3]
        a = base_tf[:3, :3] @ np.asarray(ch.axes[i], np.float32)
        a = a / np.linalg.norm(a)
        o_w.append(o.astype(np.float32))
        a_w.append(a.astype(np.float32))
        # coordinate frame at each joint origin
        wssop.frame(pos=o).add_to_scene(base.scene)

    # draw every axis as an extended line; highlight axis1 (red) and axis2 (cyan)
    L = 0.25
    hi_rgb = {0: (0.9, 0.1, 0.1), 1: (0.1, 0.8, 0.9)}
    for i, (o, a) in enumerate(zip(o_w, a_w)):
        rgb = hi_rgb.get(i, (0.5, 0.5, 0.5))
        rad = 0.004 if i in hi_rgb else 0.002
        wssop.cylinder(spos=o - L * a, epos=o + L * a, radius=rad,
                       rgb=rgb).add_to_scene(base.scene)

    # closest approach between axis 1 and axis 2
    p1, p2 = closest_points(o_w[0], a_w[0], o_w[1], a_w[1])
    gap = float(np.linalg.norm(p1 - p2))
    wssop.sphere(pos=p1, radius=0.006, rgb=(0.9, 0.1, 0.1)).add_to_scene(base.scene)
    wssop.sphere(pos=p2, radius=0.006, rgb=(0.1, 0.8, 0.9)).add_to_scene(base.scene)
    wssop.linsegs(np.array([[p1, p2]], dtype=np.float32), radius=0.0015,
                  srgbs=np.array([1.0, 1.0, 0.0], dtype=np.float32)
                  ).add_to_scene(base.scene)

    print("left-arm joint origins (world, zero config):")
    for i, (o, a) in enumerate(zip(o_w, a_w)):
        print(f"  j{i + 1}: o={np.round(o, 4)}  axis={np.round(a, 3)}")
    print(f"axis1-axis2 closest points: red={np.round(p1, 4)} "
          f"cyan={np.round(p2, 4)}")
    print(f"axis1-axis2 gap = {gap * 1000:.3f} mm")
    print("red=axis1  cyan=axis2  yellow seg=gap  (robot at zero config)")

    base.run()


if __name__ == "__main__":
    main()
