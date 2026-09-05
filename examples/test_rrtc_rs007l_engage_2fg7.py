import builtins
import numpy as np

from wrs import wum, wvw, wuc, wssop, wcm, wmppc, wmpr, khi_rs007l, or_2fg7

base = wvw.World(cam_pos=(-2, 2, 2), cam_lookat_pos=(0, 0, 0.5), toggle_auto_cam_orbit=False)
builtins.base = base
oframe = wssop.frame()
oframe.add_to_scene(base.scene)
robot = khi_rs007l.RS007L()
robot.set_pos_rotmat(pos=(0, 0, 0.01))
gripper = or_2fg7.OR2FG7()
# gripper.open()
robot.mount(gripper, robot.runtime_lnks[-1], update=True)
robot.add_to_scene(base.scene)
builtins.robot = robot

box = wssop.box(xyz_lengths=(2, 0.02, 0.3), pos=(.0, -0.3, .7),
                collision_type=wuc.CollisionType.AABB)
box.add_to_scene(base.scene)
box2 = wssop.box(xyz_lengths=(0.3, 0.02, 2), pos=(-.5, -0.3, 0.3),
                 collision_type=wuc.CollisionType.AABB)
box2.add_to_scene(base.scene)
box3 = wssop.box(xyz_lengths=(0.02, 2, 0.3), pos=(.3, 0.0, 1),
                 collision_type=wuc.CollisionType.AABB)
box3.add_to_scene(base.scene)
box4 = box2.clone()
box4.pos = (-.5, .3, .5)
box4.add_to_scene(base.scene)
box5 = box.clone()
box5.pos = (0, .3, 1)
box5.add_to_scene(base.scene)
plane_ground = wssop.plane()
plane_ground.add_to_scene(base.scene)

collider = wcm.MJCollider()
collider.append(robot)
collider.append(box)
collider.append(box2)
collider.append(box3)
collider.append(box4)
collider.append(box5)
collider.append(plane_ground)
collider.actors = [robot]
collider.compile()

pln_ctx = wmppc.PlanningContext(collider=collider)
planner = wmpr.RRTConnectPlanner(
    pln_ctx=pln_ctx, extend_step_size=np.pi / 36)
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-wum.pi / 2, -wum.pi / 4, wum.pi / 2, -wum.pi / 2, wum.pi / 4, wum.pi / 3])
robot1 = robot.clone()
robot1.fk(qs=start)
robot1.rgba = (1, 0, 0, 0.5)
robot1.add_to_scene(base.scene)
robot2 = robot.clone()
robot2.fk(qs=goal)
robot2.rgba = (0, 0, 1, 0.5)
robot2.add_to_scene(base.scene)
counter = [0]
#
# rrt_iter = planner.solve_iter(start=start, goal=goal,
#                               verbose=True, max_iters=100000)
# rrt_states = []
# final_path = None
#
# def update_pose(dt):
#     global final_path
#     if final_path is not None:
#         return
#     try:
#         status, data, tree = next(rrt_iter)
#         if status == "extend_start" or status == "extend_goal":
#             qs = data
#             robot.fk(qs=qs)
#         elif status == "success":
#             final_path = data
#             print("RRT finished, path length =", len(final_path))
#         elif status == "failed":
#             print("RRT failed")
#     except StopIteration:
#         pass
#
# base.schedule_interval(update_pose, interval=0.05)
# base.run()

state_list = planner.solve(
    start=start, goal=goal, verbose=False, max_iters=100000)

def update_pose(dt, counter):
    if counter[0] < len(state_list):
        robot.fk(qs=state_list[counter[0]])
        counter[0] += 1
    else:
        counter[0] = 0

base.schedule_interval(update_pose, interval=0.1, counter=counter)
base.run()
