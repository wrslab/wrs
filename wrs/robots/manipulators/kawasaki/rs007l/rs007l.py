import os
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
import wrs.robots.base.kine.anaik as wrbka
from wrs.manipulation.arm import SingleArmManipulation


def prepare_mechstruct():
    structure = wrbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    # 7 links
    base_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    base_lnk.set_inertia(mass=11.0)
    lnk1 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link1.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk1.set_inertia(mass=8.118)
    lnk2 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link2.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_rotmat=wum.rotmat_from_euler(0, np.pi / 2, 0),
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk2.set_inertia(mass=6.826)
    lnk3 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link3.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_rotmat=wum.rotmat_from_euler(0, np.pi / 2, 0),
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk3.set_inertia(mass=5.236)
    lnk4 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link4.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=np.array([0.0, 0.0, 0.3852], dtype=np.float32),
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk4.set_inertia(mass=5.066)
    lnk5 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link5.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_rotmat=wum.rotmat_from_euler(0, np.pi / 2, 0),
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk5.set_inertia(mass=1.625)
    lnk6 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link6.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk6.set_inertia(mass=0.625)
    # 6 joints
    jnt_bl_l1 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=-wuc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.36], dtype=np.float32),
        lmt_lo=-np.pi,
        lmt_up=np.pi,
    )
    jnt_l1_l2 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=wuc.StandardAxis.Z,
        rotmat=wum.rotmat_from_euler(0, -np.pi / 2, 0),
        lmt_lo=-3 / 4 * np.pi,
        lmt_up=3 / 4 * np.pi,
    )
    jnt_l2_l3 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=-wuc.StandardAxis.Z,
        pos=np.array([0.455, 0.0, 0.0], dtype=np.float32),
        lmt_lo=-7 / 8 * np.pi,
        lmt_up=7 / 8 * np.pi,
    )
    jnt_l3_l4 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.0925, 0.0, 0.0], dtype=np.float32),
        rotmat=wum.rotmat_from_euler(0, np.pi / 2, 0),
        lmt_lo=-10 / 9 * np.pi,
        lmt_up=10 / 9 * np.pi,
    )
    jnt_l4_l5 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=-wuc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.3825], dtype=np.float32),
        rotmat=wum.rotmat_from_euler(0, -np.pi / 2, 0),
        lmt_lo=-25 / 36 * np.pi,
        lmt_up=25 / 36 * np.pi,
    )
    jnt_l5_l6 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.078, 0.0, 0.0], dtype=np.float32),
        rotmat=wum.rotmat_from_euler(0, np.pi / 2, 0),
        lmt_lo=-2 * np.pi,
        lmt_up=2 * np.pi,
    )
    # add links
    structure.add_lnk(base_lnk)
    structure.add_lnk(lnk1)
    structure.add_lnk(lnk2)
    structure.add_lnk(lnk3)
    structure.add_lnk(lnk4)
    structure.add_lnk(lnk5)
    structure.add_lnk(lnk6)
    # add joints
    structure.add_jnt(jnt_bl_l1)
    structure.add_jnt(jnt_l1_l2)
    structure.add_jnt(jnt_l2_l3)
    structure.add_jnt(jnt_l3_l4)
    structure.add_jnt(jnt_l4_l5)
    structure.add_jnt(jnt_l5_l6)
    # ignore collisions pairs
    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    # order joints for quick access
    structure.compile()
    return structure


class RS007L(wrbmb.MechBase, SingleArmManipulation):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos, is_floating=False)
        c = self.structure.compiled
        self.add_chain('main', c.root_lnk, c.tip_lnks[0],
                       solver=wrbka.S456X12)
        self.add_tcp('flange', self.runtime_lnks[-1])


if __name__ == "__main__":
    import builtins
    import wrs.viewer.world as wvw
    import wrs.scene.scene_object_primitive as wssop

    base = wvw.World(cam_pos=[2.0, 1.0, 1.0], cam_lookat_pos=[0.0, 0.0, 0.5])
    robot = RS007L()
    print("=== RS007L IK Test ===")
    builtins.base = base
    builtins.robot = robot  # for debug access

    tgt_pos = (0.4, 0.1, 0.4)
    tgt_rotmat = wum.rotmat_from_axangle(
        wuc.StandardAxis.Z, 0
    ) @ wum.rotmat_from_axangle(wuc.StandardAxis.Y, wum.pi)
    wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat).add_to_scene(base.scene)

    # all_qs = robot.ik(tgt_pos, tgt_rotmat, max_solutions=8)
    # for qs in all_qs:
    #     tmp_robot = robot.clone()
    #     tmp_robot.fk(qs=qs)
    #     tmp_robot.add_to_scene(base.scene)
    # test ik to nearest solution
    prev_qs = np.zeros(6)
    _s = robot.ik(tgt_pos, tgt_rotmat, max_solutions=1,
                  ref_qs=robot.chain('main').extract_active_qs(
                      np.asarray(prev_qs, dtype=np.float32)))
    qs = _s[0] if _s else None
    if qs is not None:
        print("Found IK solution:", qs)
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.add_to_scene(base.scene)
    base.run()
