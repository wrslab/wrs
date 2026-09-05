import os
import numpy as np

import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.kine.anaik as wrbka
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
from wrs.manipulation.arm import SingleArmManipulation


def prepare_mechstruct():
    structure = wrbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    # UR3 kinematics from Universal_Robots_ROS2_Description:
    # config/ur3/default_kinematics.yaml
    shoulder_xyz = np.array([0.0, 0.0, 0.1519], dtype=np.float32)
    upper_arm_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    forearm_xyz = np.array([-0.24365, 0.0, 0.0], dtype=np.float32)
    wrist1_xyz = np.array([-0.21325, 0.0, 0.11235], dtype=np.float32)
    wrist2_xyz = np.array([0.0, -0.08535, -1.750557762378351e-11],
                          dtype=np.float32)
    wrist3_xyz = np.array([0.0, 0.0819, -1.679797079540562e-11],
                          dtype=np.float32)
    shoulder_rpy = (0.0, 0.0, 0.0)
    upper_arm_rpy = (1.570796327, 0.0, 0.0)
    forearm_rpy = (0.0, 0.0, 0.0)
    wrist1_rpy = (0.0, 0.0, 0.0)
    wrist2_rpy = (1.570796327, 0.0, 0.0)
    wrist3_rpy = (1.570796326589793, np.pi, np.pi)

    # UR ROS2 mesh offsets (config/ur3/visual_parameters.yaml).
    base_mesh_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    shoulder_mesh_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper_arm_mesh_pos = np.array([0.0, 0.0, 0.1198], dtype=np.float32)
    forearm_mesh_pos = np.array([0.0, 0.0, 0.0275], dtype=np.float32)
    wrist1_mesh_pos = np.array([0.0, 0.0, -0.085], dtype=np.float32)
    wrist2_mesh_pos = np.array([0.0, 0.0, -0.083], dtype=np.float32)
    wrist3_mesh_pos = np.array([0.0, -0.00255, -0.082], dtype=np.float32)
    base_mesh_rpy = (0.0, 0.0, np.pi)
    shoulder_mesh_rpy = (0.0, 0.0, np.pi)
    upper_arm_mesh_rpy = (np.pi / 2.0, 0.0, -np.pi / 2.0)
    forearm_mesh_rpy = (np.pi / 2.0, 0.0, -np.pi / 2.0)
    wrist1_mesh_rpy = (np.pi / 2.0, 0.0, 0.0)
    wrist2_mesh_rpy = (0.0, 0.0, 0.0)
    wrist3_mesh_rpy = (np.pi / 2.0, 0.0, 0.0)
    ur_blue = wuc.ExtendedColor.ORIENTAL_BLUE
    ur_gray = wuc.ExtendedColor.MOON_GRAY

    base_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "base.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=base_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*base_mesh_rpy),
        rgb=ur_gray,
    )
    lnk1 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "shoulder.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=shoulder_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*shoulder_mesh_rpy),
        rgb=ur_blue,
    )
    lnk2 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "upperarm.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=upper_arm_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*upper_arm_mesh_rpy),
        rgb=ur_gray,
    )
    lnk3 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "forearm.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=forearm_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*forearm_mesh_rpy),
        rgb=ur_gray,
    )
    lnk4 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "wrist1.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=wrist1_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*wrist1_mesh_rpy),
        rgb=ur_blue,
    )
    lnk5 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "wrist2.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=wrist2_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*wrist2_mesh_rpy),
        rgb=ur_blue,
    )
    lnk6 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "wrist3.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=wrist3_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*wrist3_mesh_rpy),
        rgb=ur_gray,
    )

    # 6 revolute joints, following UR kinematic convention in this framework
    jnt_b_l1 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=wuc.StandardAxis.Z,
        pos=shoulder_xyz,
        rotmat=wum.rotmat_from_euler(*shoulder_rpy),
        lmt_lo=-2.0 * np.pi,
        lmt_up=2.0 * np.pi,
    )
    jnt_l1_l2 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=wuc.StandardAxis.Z,
        pos=upper_arm_xyz,
        rotmat=wum.rotmat_from_euler(*upper_arm_rpy),
        lmt_lo=-2.0 * np.pi,
        lmt_up=2.0 * np.pi,
    )
    jnt_l2_l3 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=wuc.StandardAxis.Z,
        pos=forearm_xyz,
        rotmat=wum.rotmat_from_euler(*forearm_rpy),
        lmt_lo=-2.0 * np.pi,
        lmt_up=2.0 * np.pi,
    )
    jnt_l3_l4 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=wuc.StandardAxis.Z,
        pos=wrist1_xyz,
        rotmat=wum.rotmat_from_euler(*wrist1_rpy),
        lmt_lo=-2.0 * np.pi,
        lmt_up=2.0 * np.pi,
    )
    jnt_l4_l5 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=wuc.StandardAxis.Z,
        pos=wrist2_xyz,
        rotmat=wum.rotmat_from_euler(*wrist2_rpy),
        lmt_lo=-2.0 * np.pi,
        lmt_up=2.0 * np.pi,
    )
    jnt_l5_l6 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=wuc.StandardAxis.Z,
        pos=wrist3_xyz,
        rotmat=wum.rotmat_from_euler(*wrist3_rpy),
        lmt_lo=-2.0 * np.pi,
        lmt_up=2.0 * np.pi,
    )

    for lnk in [base_lnk, lnk1, lnk2, lnk3, lnk4, lnk5, lnk6]:
        structure.add_lnk(lnk)
    for jnt in [jnt_b_l1, jnt_l1_l2, jnt_l2_l3,
                jnt_l3_l4, jnt_l4_l5, jnt_l5_l6]:
        structure.add_jnt(jnt)

    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    structure.compile()
    return structure


class UR3(wrbmb.MechBase, SingleArmManipulation):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(
            rotmat=rotmat, pos=pos, is_floating=False,
            home_qs=[0, -np.pi / 2, np.pi / 2, 0, 0, 0]
        )
        c = self.structure.compiled
        self.add_chain('main', c.root_lnk, c.tip_lnks[0],
                       solver=wrbka.P234X56)
        self.add_tcp('flange', self.runtime_lnks[-1])


if __name__ == "__main__":
    import wrs.viewer.world as wvw
    import wrs.scene.scene_object_primitive as wssop
    import builtins

    base = wvw.World(cam_pos=(2, 1, 1), cam_lookat_pos=(0, 0, 0.5))
    builtins.base = base
    scene = base.scene
    oframe = wssop.frame()
    oframe.add_to_scene(scene)
    robot = UR3()
    robot.add_to_scene(scene)
    base.run()
    # robot.alpha=0.3
    builtins.robot = robot
    wssop.frame(
        pos=robot.tcp('flange').tf[:3, 3],
        rotmat=robot.tcp('flange').tf[:3, :3],
        color_mat=wuc.CoordColor.MYC,
    ).add_to_scene(scene)

    tgt_pos = (0.4, 0.2, 0.2)
    tgt_rotmat = wum.rotmat_from_axangle(
        wuc.StandardAxis.Z, np.pi / 6.0
    ) @ wum.rotmat_from_axangle(wuc.StandardAxis.Y, np.pi)
    wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat).add_to_scene(scene)

    all_qs = robot.ik(tgt_pos, tgt_rotmat, max_solutions=8)
    for qs in all_qs:
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.add_to_scene(base.scene)
    base.run()
