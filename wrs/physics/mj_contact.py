import mujoco
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.scene.scene_object_primitive as wssop


def debug_contacts(mjenv):
    print("ncon =", mjenv.data.ncon)
    for i in range(mjenv.data.ncon):
        c = mjenv.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = mjenv.model.geom_bodyid[g1]
        b2 = mjenv.model.geom_bodyid[g2]
        body1 = mujoco.mj_id2name(mjenv.model, mujoco.mjtObj.mjOBJ_BODY, b1)
        body2 = mujoco.mj_id2name(mjenv.model, mujoco.mjtObj.mjOBJ_BODY, b2)
        print(i, body1, "<->", body2, "dist=", c.dist)


class MJContactViz:
    def __init__(self, scene, max_contacts=64, radius=0.01):
        self.scene = scene
        self.max_contacts = max_contacts
        # Same ownership rule as MJContactForceViz: the markers belong to the
        # SCENE, so a rebuilt viz reuses them instead of stranding the old batch.
        spheres = getattr(scene, '_contact_spheres', None)
        if spheres is None:
            spheres = self._init_spheres(radius)
            scene._contact_spheres = spheres
        self._spheres = spheres
        self.max_contacts = len(spheres)
        self.clear()

    def _init_spheres(self, radius):
        spheres = []
        for _ in range(self.max_contacts):
            s = wssop.sphere(radius=radius,
                             rgb=(1, 0, 0),
                             alpha=wuc.ALPHA.SEMI,
                             collision_type=None)
            s.add_to_scene(self.scene)
            spheres.append(s)
        return spheres

    def clear(self):
        for s in self._spheres:
            s.alpha = wuc.ALPHA.TRANSPARENT

    def update_from_data(self, mj_data):
        nc = min(mj_data.ncon, self.max_contacts)
        self.clear()
        for i in range(nc):
            cp = mj_data.contact[i]
            pos = cp.pos
            s = self._spheres[i]
            s.pos = pos
            s.alpha = wuc.ALPHA.NEAR_SOLID


class MJContactForceViz:
    def __init__(self, scene,
                 max_contacts=64,
                 base_length=wuc.ForceArrowSize.BASE_LENGTH,
                 gain=wuc.ForceArrowSize.GAIN):
        self.scene = scene
        self.max_contacts = max_contacts
        self.base_length = base_length
        self.gain = gain
        # The arrows belong to the SCENE, not to this viz: an MJEnv (and with it
        # a fresh viz) is rebuilt on every collider refresh, so allocating a new
        # batch here would strand the previous one in the scene forever.
        arrows = getattr(scene, '_force_arrows', None)
        if arrows is None:
            arrows = self._init_arrows()
            scene._force_arrows = arrows
        self._arrows = arrows
        # The pool is shared and sized by whoever built it first, so a later viz
        # asking for MORE contacts than it holds must scale back rather than run
        # off the end of it.
        self.max_contacts = len(arrows)
        self.clear()

    def _init_arrows(self):
        arrows = []
        for _ in range(self.max_contacts):
            a = wssop.arrow(
                spos=np.zeros(3),
                epos=np.array([0.0, 0.0, self.base_length]),
                shaft_radius=wuc.ForceArrowSize.SHAFT_RADIUS,
                head_radius=wuc.ForceArrowSize.HEAD_RADIUS,
                head_length=wuc.ForceArrowSize.HEAD_LENGTH,
                n_segs=8, rgb=(1, 0, 0),
                alpha=wuc.ALPHA.SEMI,
                collision_type=None)
            a.add_to_scene(self.scene)
            arrows.append(a)
        return arrows

    def clear(self):
        for a in self._arrows:
            a.alpha = wuc.ALPHA.TRANSPARENT

    def update_from_data(self, mj_model, mj_data):
        self.clear()
        nc = min(mj_data.ncon, self.max_contacts)
        force = np.zeros(6)
        for i in range(nc):
            cp = mj_data.contact[i]
            spos = cp.pos
            mujoco.mj_contactForce(mj_model, mj_data, i, force)
            fn_vec = -cp.frame.reshape(3, 3) @ force[:3]
            fn = np.linalg.norm(fn_vec)
            if fn < 1e-10:
                continue
            direction = fn_vec / fn
            rotmat = wum.rotmat_between_vecs(wuc.StandardAxis.Z, direction)
            strength = np.tanh(fn * self.gain)
            r = strength
            g = strength * (1 - strength)
            b = 1 - strength
            rgba = (r, g, b, wuc.ALPHA.SOLID)
            a = self._arrows[i]
            a.set_pos_rotmat(pos=spos, rotmat=rotmat)
            a.rgba = rgba
