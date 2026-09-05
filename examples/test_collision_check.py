import numpy as np
import wrs.collider.mj_collider as wcm
from wrs import wum, wvw, wssop, khi_rs007l, or_2fg7

# Simple test to verify robot-ground collision detection
base = wvw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)
wssop.frame().add_to_scene(base.scene)

robot = khi_rs007l.RS007L(pos=(.5, 0, 0.01))
robot.add_to_scene(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.add_to_scene(base.scene)
robot.mount(gripper, robot.runtime_lnks[-1], update=True)

# create ground plane at z=0
ground = wssop.plane(pos=(0, 0, 0))
ground.add_to_scene(base.scene)

# setup mj collider
mjc = wcm.MJCollider()
mjc.append(robot)
mjc.append(ground)
mjc.actors = [robot]
mjc.compile(margin=0.0)

# Test 1: Initial pose (should be collision-free with slight z offset)
qs_safe = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
print(f"Test 1 - Safe pose [0,0,0,0,0,0]: collided = {mjc.is_collided(qs_safe)}")

# Test 2: Joint angles that cause collision with ground
qs_collision = np.array([0, np.pi/3, 0, 0, 0, 0], dtype=np.float32)
print(f"Test 2 - Collision pose [0,π/3,0,0,0,0]: collided = {mjc.is_collided(qs_collision)}")

# Test 3: Lower robot base to z=0 (should definitely collide)
robot.pos = (.5, 0, 0)
qs_low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
print(f"Test 3 - Robot at z=0: collided = {mjc.is_collided(qs_low)}")

print("\nIf Test 3 shows 'collided = True', collision detection is working correctly!")

base.run()
