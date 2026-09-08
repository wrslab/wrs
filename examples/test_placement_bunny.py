import builtins
import wrs.viewer.key as key
from wrs import wvw, wssop, wsso, wuc
import wrs.geom.fitting as wgf
import wrs.geom.surface as wgs
import wrs.grasp.placement as wgp

base = wvw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)
builtins.base = base
bunny = wsso.SceneObject.from_file(
    "bunny.stl", collision_type=wuc.CollisionType.MESH)
bunny.alpha = 0.3
bunny.add_to_scene(base.scene)
oframe = wssop.frame(length_scale=0.25)
oframe.add_to_scene(base.scene)

plane_ground = wssop.plane()
plane_ground.add_to_scene(base.scene)

geom = bunny.collisions[0].geom
geom_hull = wgf.convex_hull(geom)
facets = wgs.segment_surface(geom_hull)
stable_poses = wgp.compute_stable_poses(
    geom_hull.vs, geom_hull.fs, facets,
    com=None, stable_thresh=10.0)
if not stable_poses:
    print("No stable poses found")

cur_idx = 0
cur_vis = None

def show_pose(idx):
    global cur_vis
    pos, rotmat, seg_id, ratio, segs3d = stable_poses[idx]
    bunny.pos = pos
    bunny.rotmat = rotmat
    print(f"seg={seg_id}, ratio={ratio:.6f}")
    # optional: visualize current segment
    if cur_vis is not None:
        cur_vis.remove_from_scene(base.scene)
        cur_vis = None
    cur_vis = wssop.linsegs(segs3d, radius=.002)
    # fs_sub = geom_hull.fs[facets[seg_id]]
    # cur_vis = wssop.gen_mesh(geom_hull.vs, fs_sub, rgb=(1, 0, 0), alpha=0.6)
    cur_vis.add_to_scene(base.scene)
    cur_vis.pos=pos
    cur_vis.rotmat=rotmat

def update(dt):
    global cur_idx
    if not stable_poses:
        return
    if base.is_key_pressed_edge(key.SPACE):
        show_pose(cur_idx)
        cur_idx = (cur_idx + 1) % len(stable_poses)

base.schedule_interval(update, interval=0.02)
base.run()
