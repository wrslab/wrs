import os
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
from wrs.manipulation.arm import SingleArmManipulation


def prepare_mechstruct():
    structure = wrbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    fr_white = wuc.BasicColor.WHITE
    # 8 links: base (link0) + link1..link7
    base_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link0.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    base_lnk.set_inertia(mass=2.3)
    lnk1 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link1.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk1.set_inertia(mass=2.74)
    lnk2 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link2.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk2.set_inertia(mass=2.74)
    lnk3 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link3.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk3.set_inertia(mass=2.38)
    lnk4 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link4.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk4.set_inertia(mass=2.38)
    lnk5 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link5.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk5.set_inertia(mass=2.74)
    lnk6 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link6.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk6.set_inertia(mass=1.55)
    lnk7 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "link7.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk7.set_inertia(mass=0.54)
    # 7 revolute joints (kinematics from FR3 URDF)
    jnt_bl_l1 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.333], dtype=np.float32),
        lmt_lo=-2.8973,
        lmt_up=2.8973,
    )
    jnt_l1_l2 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=wuc.StandardAxis.Z,
        rotmat=wum.rotmat_from_euler(-np.pi / 2, 0, 0),
        lmt_lo=-1.8326,
        lmt_up=1.8326,
    )
    jnt_l2_l3 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.0, -0.316, 0.0], dtype=np.float32),
        rotmat=wum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-2.8972,
        lmt_up=2.8972,
    )
    jnt_l3_l4 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.0825, 0.0, 0.0], dtype=np.float32),
        rotmat=wum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-3.0718,
        lmt_up=-0.1222,
    )
    jnt_l4_l5 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=wuc.StandardAxis.Z,
        pos=np.array([-0.0825, 0.384, 0.0], dtype=np.float32),
        rotmat=wum.rotmat_from_euler(-np.pi / 2, 0, 0),
        lmt_lo=-2.8798,
        lmt_up=2.8798,
    )
    jnt_l5_l6 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=wuc.StandardAxis.Z,
        rotmat=wum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=0.4364,
        lmt_up=4.6251,
    )
    jnt_l6_l7 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk6,
        child_lnk=lnk7,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.088, 0.0, 0.0], dtype=np.float32),
        rotmat=wum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-3.0543,
        lmt_up=3.0543,
    )
    # add links
    structure.add_lnk(base_lnk)
    structure.add_lnk(lnk1)
    structure.add_lnk(lnk2)
    structure.add_lnk(lnk3)
    structure.add_lnk(lnk4)
    structure.add_lnk(lnk5)
    structure.add_lnk(lnk6)
    structure.add_lnk(lnk7)
    # add joints
    structure.add_jnt(jnt_bl_l1)
    structure.add_jnt(jnt_l1_l2)
    structure.add_jnt(jnt_l2_l3)
    structure.add_jnt(jnt_l3_l4)
    structure.add_jnt(jnt_l4_l5)
    structure.add_jnt(jnt_l5_l6)
    structure.add_jnt(jnt_l6_l7)
    # ignore collisions pairs
    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    structure.ignore_collision(lnk5, lnk7)
    # order joints for quick access
    structure.compile()
    return structure


class FR3(wrbmb.MechBase, SingleArmManipulation):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(
            rotmat=rotmat,
            pos=pos,
            is_floating=False,
            home_qs=[0.0, -np.pi / 4, 0.0,
                     -3 * np.pi / 4, 0.0, np.pi / 2,
                     np.pi / 4],
        )
        c = self.structure.compiled
        self.add_chain('main', c.root_lnk, c.tip_lnks[0])
        # link7 -> link8 (flange) fixed offset: +0.107 m along z
        self.add_tcp('flange', self.runtime_lnks[-1],
                     wum.tf_from_pos_rotmat(pos=(0.0, 0.0, 0.107)))


def fr3_with_hand(rotmat=None, pos=None, jaw_width=0.08):
    """Convenience factory: FR3 arm with the stock Franka Hand attached."""
    from wrs.robots.end_effectors.fr3_gripper.fr3_gripper import FR3Gripper

    arm = FR3(rotmat=rotmat, pos=pos)
    hand = FR3Gripper()
    hand.set_opening(jaw_width)
    # panda_hand_joint: hand rotated -pi/4 about Z relative to link7
    loc_tf = wum.tf_from_pos_rotmat(
        rotmat=wum.rotmat_from_euler(0, 0, -np.pi / 4)
    )
    arm.mount(hand, arm.runtime_lnks[-1], loc_tf, update=True)
    return arm, hand


if __name__ == "__main__":
    import builtins
    import wrs.viewer.world as wvw
    import wrs.scene.scene_object_primitive as wssop

    base = wvw.World(cam_pos=[2.0, 1.0, 1.0], cam_lookat_pos=[0.0, 0.0, 0.5])
    arm, hand = fr3_with_hand()
    builtins.base = base
    builtins.arm = arm
    builtins.hand = hand
    arm.add_to_scene(base.scene)
    wssop.frame().add_to_scene(base.scene)
    arm.toggle_tcp('flange')

    base.run()
