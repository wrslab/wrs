"""Plan O6 left-hand antipodal grasps on the cylinder loaded from cylinder.stl.
Counterpart to o6scnobjplanning.py (an wssop.cylinder primitive of the same
size) -- run both and compare with o6cmpplanning.py.

Headless: ONE_HEADLESS=1   Viewer keys: N = next grasp pair
"""
import os

import wrs.utils.constant as wuc
import wrs.scene.scene_object as wsso
from _o6cylplan import plan_save_show, _THIS

CYL_STL = os.path.join(_THIS, "cylinder.stl")
OUT_JSON = os.path.join(_THIS, "o6_cyl_stl_grasps.json")


def main():
    cyl = wsso.SceneObject.from_file(
        CYL_STL, collision_type=wuc.CollisionType.MESH, is_floating=True,
        rgb=(0.6, 0.7, 0.5))
    plan_save_show(cyl, OUT_JSON, "stl")


if __name__ == "__main__":
    main()
