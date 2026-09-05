import os
import numpy as np

import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
from wrs.manipulation.arm import SingleArmManipulation
import wrs.robots.manipulators.fanuc.crx5ia.ik as wrmfci


def prepare_mechstruct():
    structure = wrbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    # CRX5ia kinematics from crx5ia_urdf_macro.xacro:
    # https://github.com/fanuc/crx_urdf
    l_base = 0.185
    l_1 = 0.0
    l_2 = 0.410
    l_3 = 0.0
    l_4 = 0.430
    l_5 = 0.130
    l_6 = 0.145

    # Joint positions in parent link frame (from URDF joint <origin> xyz)
    j1_xyz = np.array([0.0, 0.0, l_base], dtype=np.float32)
    j2_xyz = np.array([l_1, 0.0, 0.0], dtype=np.float32)
    j3_xyz = np.array([0.0, 0.0, l_2], dtype=np.float32)
    j4_xyz = np.array([0.0, 0.0, l_3], dtype=np.float32)
    j5_xyz = np.array([l_4, 0.0, 0.0], dtype=np.float32)
    j6_xyz = np.array([0.0, -l_5, 0.0], dtype=np.float32)

    # Joint axes (from URDF joint <axis> xyz)
    # Note: StandardAxis only has positive X, Y, Z.
    # Negative axes are specified via rotmat or by using the axis parameter directly.
    # For negative axes, we pass the numpy array directly.
    j1_axis = wuc.StandardAxis.Z
    j2_axis = wuc.StandardAxis.Y
    j3_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # -Y
    j4_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # -X
    j5_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # -Y
    j6_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # -X

    # Mesh positions relative to joint frame (from URDF link collision <origin> xyz)
    # STL files use collision origin, not visual origin.
    base_mesh_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    j1_mesh_pos = np.array([0.0, 0.0, -l_base], dtype=np.float32)
    j2_mesh_pos = np.array([-l_1, 0.0, -l_base], dtype=np.float32)
    j3_mesh_pos = np.array([-l_1, 0.0, -(l_base + l_2)], dtype=np.float32)
    j4_mesh_pos = np.array([-l_1, 0.0, -(l_base + l_2 + l_3)], dtype=np.float32)
    j5_mesh_pos = np.array([-(l_1 + l_4), 0.0, -(l_base + l_2 + l_3)],
                           dtype=np.float32)
    # J6 collision origin is "0 0 0" (different from visual origin)
    j6_mesh_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Mesh rotations: collision origin rpy from URDF
    # Most links have rpy="0 0 -pi/2" for collision, visual has "0 0 0"
    base_mesh_rpy = (0.0, 0.0, 0.0)
    j1_mesh_rpy = (0.0, 0.0, -np.pi / 2.0)
    j2_mesh_rpy = (0.0, 0.0, -np.pi / 2.0)
    j3_mesh_rpy = (0.0, 0.0, -np.pi / 2.0)
    j4_mesh_rpy = (0.0, 0.0, -np.pi / 2.0)
    j5_mesh_rpy = (0.0, 0.0, -np.pi / 2.0)
    j6_mesh_rpy = (0.0, 0.0, 0.0)

    # FANUC CRX5ia color (approximate FANUC orange)
    crx_orange = (1.0, 0.4, 0.0)
    crx_gray = (0.6, 0.6, 0.6)

    # Create links with meshes
    base_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "base.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=base_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*base_mesh_rpy),
        rgb=crx_gray,
    )
    lnk1 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j1.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=j1_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*j1_mesh_rpy),
        rgb=crx_orange,
    )
    lnk2 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j2.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=j2_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*j2_mesh_rpy),
        rgb=crx_orange,
    )
    lnk3 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j3.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=j3_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*j3_mesh_rpy),
        rgb=crx_orange,
    )
    lnk4 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j4.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=j4_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*j4_mesh_rpy),
        rgb=crx_orange,
    )
    lnk5 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j5.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=j5_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*j5_mesh_rpy),
        rgb=crx_orange,
    )
    lnk6 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j6.stl"),
        collision_type=wuc.CollisionType.MESH,
        loc_pos=j6_mesh_pos,
        loc_rotmat=wum.rotmat_from_euler(*j6_mesh_rpy),
        rgb=crx_gray,
    )

    # Joint limits (from URDF, converted to radians)
    # J1: +/- 200 deg
    # J2: +/- 179.9 deg
    # J3: -68 deg to 248 deg
    # J4: +/- 190 deg
    # J5: +/- 179.9 deg
    # J6: +/- 225 deg
    j1_lo = np.radians(-200)
    j1_up = np.radians(200)
    j2_lo = np.radians(-179.9)
    j2_up = np.radians(179.9)
    j3_lo = np.radians(-68)
    j3_up = np.radians(248)
    j4_lo = np.radians(-190)
    j4_up = np.radians(190)
    j5_lo = np.radians(-179.9)
    j5_up = np.radians(179.9)
    j6_lo = np.radians(-225)
    j6_up = np.radians(225)

    # 6 revolute joints
    # CRX5ia has mixed joint axes (Z, Y, -Y, -X, -Y, -X)
    # Joint rotations: from URDF joint <origin> rpy (all "0 0 0")
    j1_rpy = (0.0, 0.0, 0.0)
    j2_rpy = (0.0, 0.0, 0.0)
    j3_rpy = (0.0, 0.0, 0.0)
    j4_rpy = (0.0, 0.0, 0.0)
    j5_rpy = (0.0, 0.0, 0.0)
    j6_rpy = (0.0, 0.0, 0.0)

    jnt_b_l1 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=j1_axis,
        pos=j1_xyz,
        rotmat=wum.rotmat_from_euler(*j1_rpy),
        lmt_lo=j1_lo,
        lmt_up=j1_up,
    )
    jnt_l1_l2 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=j2_axis,
        pos=j2_xyz,
        rotmat=wum.rotmat_from_euler(*j2_rpy),
        lmt_lo=j2_lo,
        lmt_up=j2_up,
    )
    jnt_l2_l3 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=j3_axis,
        pos=j3_xyz,
        rotmat=wum.rotmat_from_euler(*j3_rpy),
        lmt_lo=j3_lo,
        lmt_up=j3_up,
    )
    jnt_l3_l4 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=j4_axis,
        pos=j4_xyz,
        rotmat=wum.rotmat_from_euler(*j4_rpy),
        lmt_lo=j4_lo,
        lmt_up=j4_up,
    )
    jnt_l4_l5 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=j5_axis,
        pos=j5_xyz,
        rotmat=wum.rotmat_from_euler(*j5_rpy),
        lmt_lo=j5_lo,
        lmt_up=j5_up,
    )
    jnt_l5_l6 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=j6_axis,
        pos=j6_xyz,
        rotmat=wum.rotmat_from_euler(*j6_rpy),
        lmt_lo=j6_lo,
        lmt_up=j6_up,
    )

    for lnk in [base_lnk, lnk1, lnk2, lnk3, lnk4, lnk5, lnk6]:
        structure.add_lnk(lnk)
    for jnt in [jnt_b_l1, jnt_l1_l2, jnt_l2_l3,
                jnt_l3_l4, jnt_l4_l5, jnt_l5_l6]:
        structure.add_jnt(jnt)

    # Ignore collision between non-adjacent links
    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    structure.compile()
    return structure


class CRX5ia(wrbmb.MechBase, SingleArmManipulation):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(
            rotmat=rotmat, pos=pos, is_floating=False, home_qs=[0, 0, 0, 0, 0, 0]
        )
        c = self.structure.compiled
        self.add_chain('main', c.root_lnk, c.tip_lnks[0],
                       solver=wrmfci.CRX5iaAnalyticIK)
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
    robot = CRX5ia()
    robot.add_to_scene(scene)
    base.run()
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
