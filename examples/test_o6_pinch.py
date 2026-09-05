"""Antipodal grasp planning for the Linkerbot O6 dexterous hand's PINCH.

``antipodal`` is written for a parallel-jaw gripper, but a thumb-index pinch is
exactly a 2-point opposing grasp. The O6 hand presents itself as a parallel jaw
through ``hand.as_jaw('pinch')`` -- a ``JawView`` that exposes the gripper
interface antipodal needs (jaw_range, set_opening, open_dir_at, grasp_center_at,
grip_at). No separate adapter class.

``test_o6_tripod`` reuses ``main`` with ``'tripod'`` (index + middle). Shows the
best pinch on a small bunny solid, plus the next few as translucent ghosts.
Keys: N = next grasp as the solid one, R = re-plan.
"""
import os

import wrs.utils.constant as wuc
import wrs.viewer.world as wvw
import wrs.geom.loader as wgl
import wrs.scene.render_model as wsrm
import wrs.scene.scene_object as wsso
import wrs.scene.scene_object_primitive as wssop
import wrs.robots.end_effectors.linkerbot.o6.o6 as o6
from wrs.grasp.antipodal import antipodal

BUNNY_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'bunny_small.stl')


def make_bunny():
    bunny = wsso.SceneObject(collision_type=wuc.CollisionType.MESH, is_floating=True)
    bunny.add_visual(
        wsrm.RenderModel(geom=wgl.load_geometry(BUNNY_STL), rgb=(0.85, 0.7, 0.6)),
        auto_make_collision=True)
    return bunny


def main(primitive='pinch'):
    base = wvw.World(cam_pos=(0.28, 0.22, 0.18), cam_lookat_pos=(0.0, 0.0, 0.02))
    # length_scale shrinks the shaft; radius_scale must shrink the head too,
    # else the (unscaled) head fills the whole arrow and only tips show.
    wssop.frame(length_scale=0.2, radius_scale=0.25).add_to_scene(base.scene)

    # present the O6 left hand as a parallel jaw for this grasp primitive
    grasper = o6.O6Left().as_jaw(primitive)
    bunny = make_bunny()
    bunny.add_to_scene(base.scene)

    state = {'grasps': [], 'cur': 0, 'solid': None, 'ghosts': []}

    def clear():
        if state['solid'] is not None:
            state['solid'].remove_from_scene(base.scene)
            state['solid'] = None
        for gh in state['ghosts']:
            gh.remove_from_scene(base.scene)
        state['ghosts'] = []

    def show(idx):
        clear()
        grasps = state['grasps']
        if not grasps:
            return
        idx %= len(grasps)
        state['cur'] = idx
        # a few lower-ranked candidates as translucent ghosts
        for g in grasps[1:6]:
            pose = g.pose
            jw = g.provenance["jaw_width"]
            gh = grasper.clone()
            gh.grip_at(pose[:3, 3], pose[:3, :3], jw)
            gh.alpha = 0.2
            gh.add_to_scene(base.scene)
            state['ghosts'].append(gh)
        # the selected grasp solid
        g = grasps[idx]
        pose = g.pose
        jw = g.provenance["jaw_width"]
        solid = grasper.clone()
        solid.grip_at(pose[:3, 3], pose[:3, :3], jw)
        solid.add_to_scene(base.scene)
        state['solid'] = solid

    def replan():
        grasps = antipodal(grasper, bunny, density=0.0008, normal_tol_deg=25,
                           roll_step_deg=30, clearance=0.003)[:40]
        state['grasps'] = grasps
        print(f'antipodal {primitive} grasps: {len(grasps)} '
              f'(N = next, R = re-plan)')
        show(0)

    replan()

    import wrs.viewer.key as key

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            show(state['cur'] + 1)
        if base.input_manager.is_key_pressed_edge(key.R):
            replan()

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == '__main__':
    main('pinch')
