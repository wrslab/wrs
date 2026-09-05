import wrs.geom.fitting as wgf
import wrs.geom.surface as wgs
from wrs import wvw, wssop, wsso, wuc, key

base = wvw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)

oframe = wssop.frame()
bunny = wsso.SceneObject.from_file(
    "bunny.stl", collision_type=wuc.CollisionType.MESH)
oframe.add_to_scene(base.scene)
bunny.add_to_scene(base.scene)
bunny.alpha=.3

geom_hull = wgf.convex_hull(bunny.collisions[0].geom)
segmented = wgs.segment_surface(geom_hull, normal_tol_deg=5)
for fids in segmented:
    wssop.mesh(geom_hull.vs, geom_hull.fs[fids],
               rgb=(1, 0, 0)).add_to_scene(base.scene)
    break

counter = [0]
def draw_segmented(dt, geom, segmented, counter):
    if counter[0] >= len(segmented):
        return
    if base.input_manager.is_key_pressed(key.SPACE):
        fids = segmented[counter[0]]
        wssop.mesh(geom.vs, geom.fs[fids],
                   rgb=(1, 0, 0)).add_to_scene(base.scene)
        counter[0] += 1

base.schedule_interval(draw_segmented, 0.01, geom_hull, segmented, counter)
base.run()