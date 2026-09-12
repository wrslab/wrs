"""Agent harness demo: an LLM plans a collision-free reach behind a wall.

Headless benchmark task for :mod:`wrs.agent`: the scene is an RS007L with a
2FG7, a wall between the arm and a target pose; the model must compose the
skills (reachable / moveto / Recipe) into a script whose `motion` routes the
tcp to the target. Needs LLM credentials (ANTHROPIC_API_KEY or an `ant auth
login` profile):

    py -3.12 examples/test_agent_moveto.py
"""
import numpy as np

from wrs import wum, wuc, wssop, khi_rs007l, or_2fg7
from wrs.agent import solve

TARGET_POS = np.array([0., .55, .25])
TARGET_ROTMAT = wum.rotmat_from_euler(wum.pi, 0, 0)   # grasp axis straight down


def build_scene():
    robot = khi_rs007l.RS007L()
    gripper = or_2fg7.OR2FG7()
    robot.mount(gripper, robot.runtime_lnks[-1], update=True)
    wall = wssop.box(xyz_lengths=(.8, .02, .5), pos=(0, .35, .25),
                     collision_type=wuc.CollisionType.AABB)
    collider = robot.build_collider(objects=(wall,))
    namespace = {
        'np': np,
        'robot': robot,               # RS007L, SingleArmManipulation verbs
        'gripper': gripper,           # OR2FG7 mounted at the flange
        'collider': collider,         # robot + wall, compiled
        'tcp': gripper.tcp('grasp_center'),
        'target_pos': TARGET_POS.copy(),
        'target_rotmat': TARGET_ROTMAT.copy(),
    }
    description = (
        "- `robot`: a Kawasaki RS007L arm (all manipulation skills), at the\n"
        "  origin, at its zero config.\n"
        "- `gripper`: an OR2FG7 gripper mounted on the arm's flange.\n"
        "- `collider`: the compiled collision world (robot + a wall of size\n"
        "  0.8 x 0.02 x 0.5 m standing at (0, 0.35, 0.25), between the arm\n"
        "  and the target).\n"
        "- `tcp`: the gripper's grasp_center TCP (pass as tcp=).\n"
        "- `target_pos`, `target_rotmat`: the pose the grasp_center must\n"
        "  reach, behind the wall.\n"
        "- `np`: numpy.")
    return namespace, description


def check(outcome):
    """Accept only a plan that actually ends at the target pose."""
    robot = outcome.namespace['robot']
    robot.fk(qs=outcome.motion.robot_qpos_list[-1])
    tf = np.asarray(outcome.namespace['tcp'].tf, dtype=np.float32)
    err = float(np.linalg.norm(tf[:3, 3] - TARGET_POS))
    if err > 0.01:
        return (f'the motion ends with the tcp {err:.3f} m away from '
                f'target_pos -- it must end AT the target pose')
    return None


if __name__ == '__main__':
    solution = solve('Move the gripper grasp_center to (target_pos, '
                     'target_rotmat) without colliding with the wall.',
                     build_scene, check=check)
    print(solution)
    if solution.ok:
        print(f'{len(solution.motion.robot_qpos_list)} waypoints')
        print(solution.code)
