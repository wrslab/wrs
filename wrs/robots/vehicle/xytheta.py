import numpy as np
import wrs.utils.constant as wuc
import wrs.scene.render_model_primitive as wsrmp
import wrs.robots.base.mech_base as wrbmb
import wrs.robots.base.mech_structure as wsrbms


def prepare_mechstruct():
    structure = wsrbms.MechStruct()
    wd_lnk = wsrbms.Link()
    dummy_xlnk = wsrbms.Link()
    dummy_ylnk = wsrbms.Link()
    body_lnk = wsrbms.Link(collision_type=wuc.CollisionType.AABB)
    body_lnk.add_visual(wsrmp.gen_box_rmodel((.2, .2, .2)))
    body_lnk.set_inertia(mass=0.1)
    joint_x = wsrbms.Joint(jnt_type=wuc.JntType.PRISMATIC,
                           parent_lnk=wd_lnk,
                           child_lnk=dummy_xlnk,
                           axis=wuc.StandardAxis.X,
                           lmt_lo=-5.0, lmt_up=5.0)
    joint_y = wsrbms.Joint(jnt_type=wuc.JntType.PRISMATIC,
                           parent_lnk=dummy_xlnk,
                           child_lnk=dummy_ylnk,
                           axis=wuc.StandardAxis.Y,
                           lmt_lo=-5.0, lmt_up=5.0)
    joint_t = wsrbms.Joint(jnt_type=wuc.JntType.REVOLUTE,
                           parent_lnk=dummy_ylnk,
                           child_lnk=body_lnk,
                           axis=wuc.StandardAxis.Z,
                           lmt_lo=-np.pi, lmt_up=np.pi)
    structure.add_lnk(wd_lnk)
    structure.add_lnk(dummy_xlnk)
    structure.add_lnk(dummy_ylnk)
    structure.add_lnk(body_lnk)
    structure.add_jnt(joint_x)
    structure.add_jnt(joint_y)
    structure.add_jnt(joint_t)
    structure.compile()
    return structure


class XYThetaRobot(wrbmb.MechBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat, pos)
