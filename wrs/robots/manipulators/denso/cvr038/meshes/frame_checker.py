import argparse
import os

import wrs.scene.scene_object as wsso
import wrs.scene.scene_object_primitive as wssop
import wrs.utils.constant as wuc
import wrs.viewer.world as wvw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load one CVR038 link mesh and show world frame.')
    parser.add_argument('--link', default='j4.stl',
                        help='Mesh file name under this folder, e.g. base_link.stl, j1.stl ... j6.stl')
    args = parser.parse_args()

    mesh_path = os.path.join(os.path.dirname(__file__), args.link)
    if not os.path.exists(mesh_path):
        raise ValueError(f'Link mesh not found: {mesh_path}')

    base = wvw.World(cam_pos=(0.9, 0.6, 0.6), cam_lookat_pos=(0.0, 0.0, 0.15))
    scene = base.scene
    wssop.frame().add_to_scene(scene)

    link_sobj = wsso.SceneObject.from_file(
        path=mesh_path,
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    link_sobj.add_to_scene(scene)

    base.run()
