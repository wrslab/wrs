import numpy as np

import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.scene.scene_object as wsso
import wrs.scene.render_model_primitive as wsrmp
import wrs.scene.scene_object_primitive as wssop


class KineVisualizer:
    def __init__(self, mech, chain=None,
                 axis_length=None, axis_radius=None, link_radius=0.01,
                 stator_rgb=wuc.ExtendedColor.DEEP_GRAY,
                 rotor_rgb=wuc.ExtendedColor.SILVER_GRAY,
                 link_rgb=wuc.ExtendedColor.GOLD2,
                 alpha=1.0):
        # chain given -> draw just that chain's joints; None -> draw the whole
        # mechanism (all joints in the structure).
        self.mech = mech
        self.chain = chain
        self.link_radius = float(link_radius)
        if axis_radius is None:
            axis_radius = self.link_radius * 1.5
        if axis_length is None:
            axis_length = self.link_radius * 1.2
        self.axis_length = float(axis_length)
        self.axis_radius = float(axis_radius)
        self.stator_rgb = np.asarray(stator_rgb, dtype=np.float32)
        self.rotor_rgb = np.asarray(rotor_rgb, dtype=np.float32)
        self.link_rgb = np.asarray(link_rgb, dtype=np.float32)
        self.alpha = float(alpha)
        self._objects = []

    def _joint_ids(self):
        comp = self.mech.structure.compiled
        if self.chain is not None:
            return list(self.chain.jnt_ids_in_structure)
        return [comp.jidx_map[j] for j in self.mech.structure.jnts]

    def _joint_frames(self, jids):
        # Joint world frames read from the mechanism's already-computed
        # per-link world tfs. Correct for branched trees and for chains not
        # rooted at the base; a serial re-accumulation (tf @= jnt.motion_tf)
        # only works for a single base-rooted chain and mis-places branches.
        comp = self.mech.structure.compiled
        origins, axes = [], []
        for jidx in jids:
            parent_tf = self.mech.runtime_lnks[comp.plidx_of_jidx[jidx]].tf
            jtf = parent_tf @ comp.jtf0_by_idx[jidx]
            origins.append(jtf[:3, 3].copy())
            axes.append((jtf[:3, :3] @ comp.jax_by_idx[jidx]).copy())
        return origins, axes

    def _build_objects(self):
        comp = self.mech.structure.compiled
        jids = self._joint_ids()
        origins, axes = self._joint_frames(jids) if jids else ([], [])

        objs = []
        fr_h = self.link_radius * 3.0
        base_pos = self.mech.pos
        base_rotmat = self.mech.rotmat
        base_top = base_pos + base_rotmat[:, 2] * fr_h
        fr_rmodel = wsrmp.gen_frustrum_rmodel(
            height=fr_h,
            bottom_length=fr_h,
            top_length=fr_h * (2.0 / 3.0),
            rotmat=base_rotmat,
            pos=base_pos,
            rgb=wuc.ExtendedColor.SALMON_PINK,
            alpha=self.alpha)
        fr_obj = wsso.SceneObject(collision_type=None, is_floating=False)
        fr_obj.add_visual(fr_rmodel, auto_make_collision=False)
        objs.append(fr_obj)

        # Link rods follow the kinematic tree: connect each joint to the joint
        # that drives its parent link; a joint whose parent is the root link
        # connects (dashed) to the base marker.
        drives = {comp.clidx_of_jidx[jidx]: pos
                  for pos, jidx in enumerate(jids)}
        for pos, jidx in enumerate(jids):
            plidx = comp.plidx_of_jidx[jidx]
            if plidx in drives:
                spos = origins[drives[plidx]]
                if np.linalg.norm(origins[pos] - spos) < 1e-8:
                    continue
                objs.append(wssop.cylinder(
                    spos=spos, epos=origins[pos], radius=self.link_radius,
                    rgb=self.link_rgb, alpha=self.alpha))
            else:
                objs.append(wssop.dashed_cylinder(
                    spos=base_top, epos=origins[pos], radius=self.link_radius,
                    rgb=wuc.ExtendedColor.SALMON_PINK, alpha=self.alpha))

        for origin, axis in zip(origins, axes):
            axis_u = wum.unit_vec(axis, return_length=False)
            # Stator is on negative axis side, rotor on positive axis side.
            objs.append(wssop.cylinder(
                spos=origin - axis_u * self.axis_length, epos=origin,
                radius=self.axis_radius, rgb=self.stator_rgb, alpha=self.alpha))
            objs.append(wssop.cylinder(
                spos=origin, epos=origin + axis_u * self.axis_length,
                radius=self.axis_radius, rgb=self.rotor_rgb, alpha=self.alpha))
        return objs

    def add_to_scene(self, scene):
        if not self._objects:
            self._objects = self._build_objects()
        for obj in self._objects:
            scene.add(obj)

    def remove_from_scene(self, scene):
        for obj in self._objects:
            scene.remove(obj)
        self._objects = []

    def refresh(self, scene):
        self.remove_from_scene(scene)
        self.add_to_scene(scene)
