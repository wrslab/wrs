import numpy as np
import mujoco
import wrs.collider.mj_collider as wcm
from wrs import wum, wuc, wvw, wssop, khi_rs007l, or_2fg7

# Test with BOX instead of PLANE
print("="*60)
print("Testing PLANE vs BOX for ground collision")
print("="*60)

base = wvw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)

robot = khi_rs007l.RS007L(pos=(.5, 0, 0))
robot.add_to_scene(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.add_to_scene(base.scene)
robot.mount(gripper, robot.runtime_lnks[-1], update=True)

print("\n--- Test 1: Using PLANE ---")
ground_plane = wssop.plane(pos=(0, 0, 0))
ground_plane.add_to_scene(base.scene)

mjc_plane = wcm.MJCollider()
mjc_plane.append(robot)
mjc_plane.append(ground_plane)
mjc_plane.actors = [robot]
mjc_plane.compile(margin=0.0)

qs_test = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
result_plane = mjc_plane.is_collided(qs_test)
print(f"  Robot at z=0, ground plane at z=0")
print(f"  Collision detected: {result_plane}")

print("\n--- Test 2: Using BOX (large flat box) ---")
# Create a large flat box as ground
ground_box = wssop.box(
    xyz_lengths=(100, 100, 0.002),  # Very thin box
    pos=(0, 0, -0.001),  # Position slightly below z=0
    collision_type=wuc.CollisionType.AABB
)
ground_box.add_to_scene(base.scene)

mjc_box = wcm.MJCollider()
mjc_box.append(robot)
mjc_box.append(ground_box)
mjc_box.actors = [robot]
mjc_box.compile(margin=0.0)

result_box = mjc_box.is_collided(qs_test)
print(f"  Robot at z=0, ground box at z=-0.001")
print(f"  Collision detected: {result_box}")

print("\n" + "="*60)
print("CONCLUSION:")
if result_plane and result_box:
    print("  Both PLANE and BOX work correctly")
elif result_box and not result_plane:
    print("  BOX works, but PLANE does NOT work!")
    print("  RECOMMENDATION: Use gen_box() for ground instead of gen_plane()")
elif result_plane and not result_box:
    print("  PLANE works, BOX does not")
else:
    print("  Neither PLANE nor BOX detected collision")
    print("  There may be a deeper issue with collision detection")
print("="*60)
