import os
import numpy as np

import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
from wrs.manipulation.arm import SingleArmManipulation
import wrs.robots.manipulators.denso.cvr038.ik as wrmdci


def prepare_mechstruct():
    structure = wrbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    # Source:
    # https://github.com/DENSORobot/denso_robot_ros2
    # denso_robot_descriptions/robots/cobotta/urdf/denso_robot_kinematics.xacro
    # (DAE meshes converted to STL and stored in this folder.)

    # Link meshes
    base_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk1 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j1.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk2 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j2.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk3 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j3.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk4 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j4.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk5 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j5.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )
    lnk6 = wrbms.Link.from_file(
        os.path.join(mesh_dir, "j6.stl"),
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.BEIGE,
    )

    # Cobotta CVR038A1-NV6-NN kinematic layout from official xacro.
    jnt_b_l1 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        lmt_lo=-2.61799387799149,
        lmt_up=2.61799387799149,
    )
    jnt_l1_l2 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=wuc.StandardAxis.Y,
        pos=np.array([0.0, 0.0, 0.18], dtype=np.float32),
        lmt_lo=-1.0471975511966,
        lmt_up=1.74532925199433,
    )
    jnt_l2_l3 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=wuc.StandardAxis.Y,
        pos=np.array([0.0, 0.0, 0.165], dtype=np.float32),
        lmt_lo=0.314159265358979,
        lmt_up=2.44346095279206,
    )
    jnt_l3_l4 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=wuc.StandardAxis.Z,
        pos=np.array([-0.012, 0.020, -0.345], dtype=np.float32),
        lmt_lo=-2.96705972839036,
        lmt_up=2.96705972839036,
    )
    jnt_l4_l5 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=wuc.StandardAxis.Y,
        pos=np.array([0.0, -0.020, 0.5225], dtype=np.float32),
        lmt_lo=-1.65806278939461,
        lmt_up=2.35619449019234,
    )
    jnt_l5_l6 = wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=wuc.StandardAxis.Z,
        pos=np.array([0.0, -0.0445, 0.042], dtype=np.float32),
        lmt_lo=-2.96705972839036,
        lmt_up=2.96705972839036,
    )

    for lnk in [base_lnk, lnk1, lnk2, lnk3, lnk4, lnk5, lnk6]:
        structure.add_lnk(lnk)
    for jnt in [jnt_b_l1, jnt_l1_l2, jnt_l2_l3, jnt_l3_l4, jnt_l4_l5, jnt_l5_l6]:
        structure.add_jnt(jnt)

    # Typical adjacent-link ignore pairs.
    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    structure.compile()
    return structure


class CVR038(wrbmb.MechBase, SingleArmManipulation):
    """6-DOF single-arm: a MechBase configured with one root->tip chain and a
    flange tcp. No ManipulatorBase -- 'arm' is just this configuration."""

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos, is_floating=False)
        c = self.structure.compiled
        # chain + its analytic solver, declared together
        self.add_chain('main', c.root_lnk, c.tip_lnks[0],
                       solver=wrmdci.CVR038PencilIK)
        self.add_tcp('flange', self.runtime_lnks[-1])


def cvr038_with_gripper(rotmat=None, pos=None, jaw_width=0.03):
    """Convenience factory: CVR038 arm with the stock COBOTTA gripper attached."""
    from wrs.robots.end_effectors.denso.cvr038_gripper import CVR038Gripper

    arm = CVR038(rotmat=rotmat, pos=pos)
    gripper = CVR038Gripper()
    gripper.set_opening(jaw_width)
    arm.mount(gripper, arm.runtime_lnks[-1], update=True)
    return arm, gripper


if __name__ == "__main__":
    import wrs.scene.scene_object_primitive as wssop
    import wrs.robots.base.kine_visualizer as wrbkv
    import wrs.viewer.world as wvw

    base = wvw.World(cam_pos=(1.8, 1.0, 1.2),
                     cam_lookat_pos=(0.0, 0.0, 0.45))
    robot, gripper = cvr038_with_gripper()
    robot.add_to_scene(base.scene)
    robot.alpha=0.3
    kv = wrbkv.KineVisualizer(robot, chain=robot.chain('main'), alpha=0.8)
    kv.add_to_scene(base.scene)

    compiled = robot.structure.compiled
    joint_frames = []
    for jidx in robot.chain('main').jnt_ids_in_structure:
        parent_lidx = compiled.plidx_of_jidx[jidx]
        parent_lnk = robot.runtime_lnks[parent_lidx]
        frame = wssop.frame(length_scale=0.35, radius_scale=0.45)
        robot.mount(frame, parent_lnk, compiled.jtf0_by_idx[jidx], update=True)
        joint_frames.append(frame)

    base.run()
