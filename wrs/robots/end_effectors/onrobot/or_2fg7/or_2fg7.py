import os
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
import wrs.robots.end_effectors.ee_mixins as wreb


def prepare_ms():
    structure = wrbms.MechStruct()
    mesh_dir=structure.default_mesh_dir
    # 3 links
    base_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        loc_rotmat=wum.rotmat_from_euler(0, 0, np.pi / 2),
        loc_pos=None,
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.SILVER)
    lf_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "inward_left_finger_link.stl"),
        loc_rotmat=wum.rotmat_from_euler(0, 0, np.pi / 2),
        loc_pos=None,
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.STEEL_BLUE)
    rf_lnk = wrbms.Link.from_file(
        os.path.join(mesh_dir, "inward_right_finger_link.stl"),
        loc_rotmat=wum.rotmat_from_euler(0, 0, np.pi / 2),
        loc_pos=None,
        collision_type=wuc.CollisionType.MESH,
        rgb=wuc.ExtendedColor.SALMON_PINK)
    # 1 joint
    jnt_lf = wrbms.Joint(
        jnt_type=wuc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=lf_lnk,
        axis=wuc.StandardAxis.Y,
        pos=np.array([0, -0.019, 0], dtype=np.float32),
        lmt_lo=0.0, lmt_up=0.019)
    jnt_rf = wrbms.Joint(
        jnt_type=wuc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=rf_lnk,
        axis=-wuc.StandardAxis.Y,
        pos=np.array([0, 0.019, 0], dtype=np.float32),
        mmc=(jnt_lf, 1.0, 0.0),
        lmt_lo=0.0, lmt_up=0.019)
    # add lnks
    structure.add_lnk(base_lnk)
    structure.add_lnk(lf_lnk)
    structure.add_lnk(rf_lnk)
    # add jnts
    structure.add_jnt(jnt_lf)
    structure.add_jnt(jnt_rf)
    # ignore collision between fingers
    structure.ignore_collision(lf_lnk, rf_lnk)
    # order joints for quick access
    structure.compile()
    return structure


class OR2FG7(wrbmb.MechBase, wreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        super().__init__()
        self.add_tcp('grasp_center', self.runtime_root_lnk,
                     wum.tf_from_pos_rotmat(pos=(0, 0, 0.15)))
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self.jaw_range = np.array([0.005, 0.038], dtype=np.float32)  # min, max
        self.open_dir = wuc.StandardAxis.Y
        self.set_opening(0.005)

    def set_opening(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f"jaw_width {jaw_width} out of range {self.jaw_range}")
        self.fk(qs=[jaw_width * 0.5, jaw_width * 0.5])

    def clone(self):
        new = super().clone()
        new.contact_pattern = self.contact_pattern.copy()
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir
        new.set_opening(self.qs[0]*2)
        return new
