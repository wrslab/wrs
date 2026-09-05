import numpy as np
import wrs.utils.math as wum
from wrs import wvw, wuc, wsso, wsrm, or_2fg7, wcm

base = wvw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
# wssop.gen_frame().add_to_scene(base.scene)
# gripper
gripper = or_2fg7.OR2FG7()
gripper.add_to_scene(base.scene)
# # box (object to collide)
# box = wssop.gen_box(xyz_lengths=(0.06, 0.06, 0.06),
#                     rgb=wuc.BasicColor.ORANGE,
#                     collision_type=wuc.CollisionType.AABB,
#                     is_floating=True)
# box.set_pos_rotmat(pos=np.array([0.0, 0.0, 0.05]), rotmat=np.eye(3))
# box.add_to_scene(base.scene)
# bunny (object to collide)
bunny = wsso.SceneObject.from_file(
    "bunny.stl",
    collision_type=wuc.CollisionType.MESH)
bunny.add_to_scene(base.scene)
# target grasp pose (scene coordinates)
tgt_pos = np.array([0.00257776, -0.01682014, 0.14954071], dtype=np.float32)
tgt_rotmat = np.array([[0.92450643, 0.38116568, 0.00000005],
                       [0.32940900, -0.79897177, 0.50312352],
                       [0.19177355, -0.46514106, -0.86421424]], dtype=np.float32)
tgt_jw = 0.03603375
# move gripper to target
gripper.grip_at(tgt_pos, tgt_rotmat, tgt_jw)
# collider
collider = wcm.MJCollider()
collider.append(gripper)
collider.append(bunny)
collider.actors = [gripper]
collider.compile()
# collision check
collided = collider.is_collided(gripper.qs)
print("Collided:", collided)
collider._mjenv.save("collided.xml")
model = collider._mjenv.model
body = collider._mjenv.sync.sobj2bdy[bunny]
bid = model.body(body.name).id
geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == bid]
geom_id = geom_ids[0]
mesh_id = model.geom_dataid[geom_id]
vadr = model.mesh_vertadr[mesh_id]
vnum = model.mesh_vertnum[mesh_id]
fadr = model.mesh_faceadr[mesh_id]
fnum = model.mesh_facenum[mesh_id]
verts = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3).copy()
faces = model.mesh_face[fadr:fadr + fnum].reshape(-1, 3).copy()
bunny_mj = wsso.SceneObject()
bunny_mj.add_visual(wsrm.RenderModel(
    geom=(verts, faces),
    rgb=wuc.BasicColor.GREEN,
    alpha=0.3), auto_make_collision=False)
bunny_mj.add_to_scene(base.scene)
base.run()
