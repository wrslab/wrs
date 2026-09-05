import os
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
import wrs.robots.end_effectors.ee_mixins as wreb


class ORSD(wrbmb.MechBase, wreb.PointMixin):
    """Minimal OnRobot ScrewDriver implementation."""

    @classmethod
    def _build_structure(cls):
        structure = wrbms.MechStruct()
        mesh_dir = structure.default_mesh_dir
        # Single link - screwdriver body
        lnk = wrbms.Link.from_file(
            os.path.join(mesh_dir, "or_screwdriver.stl"),
            collision_type=wuc.CollisionType.MESH,
            rgb=wuc.ExtendedColor.SILVER)
        structure.add_lnk(lnk)
        structure.compile()
        return structure

    def __init__(self):
        # TCP at tip, Z-axis along screwdriver axis
        tcp_pos = np.array([0.16855, 0, 0.09509044], dtype=np.float32)
        tcp_rotmat = wum.rotmat_from_axangle(wuc.StandardAxis.Y, np.pi / 2)
        super().__init__()
        self.add_tcp('tip', self.runtime_root_lnk,
                     wum.tf_from_pos_rotmat(tcp_pos, tcp_rotmat))
        self._is_activated = False


if __name__ == '__main__':
    import builtins
    import wrs.viewer.world as wd
    import wrs.scene.scene_object_primitive as wssop
    import wrs.robots.manipulators.kawasaki.rs007l.rs007l as rs007l

    base = wd.World(cam_pos=(2,0.5,2), cam_lookat_pos=(0, 0, .75))
    builtins.base=base
    # world frame
    wssop.frame().add_to_scene(base.scene)
    manipulator = rs007l.RS007L()
    manipulator.add_to_scene(base.scene)
    manipulator.alpha=.3
    builtins.robot = manipulator
    screwdriver = ORSD()
    screwdriver.add_to_scene(base.scene)
    manipulator.mount(screwdriver, manipulator.runtime_lnks[-1],
                      wum.tf_from_pos_rotmat(pos=(0.0, 0.0, 0.05)), update=True)
    tgt_pos = (0.3, 0.5, 0.5)
    tgt_rotmat = wum.rotmat_from_axangle(wuc.StandardAxis.Y, np.pi)
    wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat, color_mat=wuc.CoordColor.DYO).add_to_scene(base.scene)
    _sols = manipulator.ik(tgt_pos, tgt_rotmat, tcp=screwdriver.tcp('tip'), max_solutions=1)
    qs = _sols[0] if _sols else None
    print(qs)
    manipulator.fk(qs)
    _tf = screwdriver.tcp('tip').tf
    wssop.frame(pos=_tf[:3, 3],
                rotmat=_tf[:3, :3],
                color_mat=wuc.CoordColor.MYC).add_to_scene(base.scene)
    wssop.frame(pos=screwdriver.pos, rotmat=screwdriver.rotmat).add_to_scene(base.scene)
    base.run()
