import wrs.utils.math as wum
import wrs.utils.decorator as wud
import wrs.utils.constant as wuc
import wrs.geom.loader as wgl
import wrs.scene.render_model as wsrm
import wrs.scene.collision_shape as wsc


class SceneObject:
    """A posed, renderable body.

    Its pose is a SINGLE world transform, ``tf`` -- there is no local/world
    pair, because a SceneObject is never expressed relative to another
    SceneObject. Scene objects form a FLAT set: ``add_to_scene`` is membership,
    and making one thing follow another is a MOUNT (``mech.mount`` /
    ``ee.hold``), which writes this object's ``tf`` outright.

    ``loc_tf`` elsewhere in the library always means "offset relative to my
    host" and belongs to the host-relative record, never to a body: a
    ``RenderModel``'s / collision shape's offset within its SceneObject, and a
    ``Mounting``'s / ``TCP``'s offset on a robot link."""

    @classmethod
    def from_file(cls, path,
                  loc_rotmat=None, loc_pos=None,  # render model offset
                  collision_type=None, is_floating=False,
                  rgb=None, alpha=1.0):
        """only allows changing local pose of the visual model"""
        instance = cls(collision_type=collision_type,
                       is_floating=is_floating)
        instance.file_path = path
        instance.add_visual(
            wsrm.RenderModel(geom=wgl.load_geometry(path),
                             rotmat=loc_rotmat, pos=loc_pos,
                             rgb=rgb, alpha=alpha),
            auto_make_collision=True)
        return instance

    def __init__(self, collision_type=None, is_floating=False, name=None):
        # world pose, plus the lazily rebuilt 4x4 cache of it
        self._rotmat = wum.ensure_rotmat(None)
        self._pos = wum.ensure_pos(None)
        self._tf = wum.tf_from_pos_rotmat(self._pos, self._rotmat)
        self._dirty = True
        self.name = name
        self.file_path = None
        self.visuals = []
        self.collisions = []
        self.toggle_render_collision = False
        self._inrtmat = None
        self._com = None
        self._mass = None
        self._is_floating = is_floating
        self._mounted_by = None     # set while a mech's mount owns our pose
        self._collision_type = collision_type
        # _collision_type = None: no auto collider generation
        self._seed_collision_role()   # two-way default from is_floating, then sticky

    # ---- pose ----------------------------------------------------------
    @wud.mark_dirty('_mark_dirty')
    def set_pos_rotmat(self, pos=None, rotmat=None):
        self._rotmat[:] = wum.ensure_rotmat(rotmat)
        self._pos[:] = wum.ensure_pos(pos)

    @property
    def pos(self):
        return self._pos.copy()

    @pos.setter
    @wud.mark_dirty('_mark_dirty')
    def pos(self, pos):
        self._pos[:] = wum.ensure_pos(pos)

    @property
    def rotmat(self):
        return self._rotmat.copy()

    @rotmat.setter
    @wud.mark_dirty('_mark_dirty')
    def rotmat(self, rotmat):
        self._rotmat[:] = wum.ensure_rotmat(rotmat)

    @property
    def quat(self):
        # TODO cache?
        return wum.quat_from_rotmat(self._rotmat)

    @property
    @wud.lazy_update('_dirty', '_rebuild_tf')
    def tf(self):
        """This body's pose in the world -- the only pose it has."""
        return self._tf.copy()

    @tf.setter
    @wud.mark_dirty('_mark_dirty')
    def tf(self, tf):
        tf = wum.ensure_tf(tf)
        self._rotmat[:] = tf[:3, :3]
        self._pos[:] = tf[:3, 3]

    def _rebuild_tf(self):
        if not self._dirty:
            return
        self._tf[:3, :3] = self._rotmat
        self._tf[:3, 3] = self._pos
        self._dirty = False

    def _mark_dirty(self):
        self._dirty = True

    # ---- scene membership ----------------------------------------------
    def add_to_scene(self, scene):
        """Put this object into ``scene`` so it renders. Scene membership ONLY --
        scene objects are a flat set, never a tree. Making one thing follow
        another is a MOUNT (``mech.mount`` / ``ee.hold``), which owns the
        child's pose outright."""
        from wrs.scene.scene import Scene
        if not isinstance(scene, Scene):
            raise TypeError(
                f"add_to_scene expects a Scene, got {type(scene)}. To make "
                f"this object follow a robot link use "
                f"mech.mount(obj, lnk, loc_tf) instead.")
        scene.add(self)

    def remove_from_scene(self, scene):
        """Remove this object from ``scene``. Undoing a mount is
        ``mech.unmount`` / ``ee.release``, a separate concern."""
        from wrs.scene.scene import Scene
        if not isinstance(scene, Scene):
            raise TypeError(
                f"remove_from_scene expects a Scene, got {type(scene)}. "
                f"Use mech.unmount(obj) to undo a mount.")
        scene.remove(self)

    def add_visual(self, model, auto_make_collision=True):
        self.visuals.append(model)
        if auto_make_collision:
            self._auto_make_collision_from_model(model)

    def add_collision(self, model):
        self.collisions.append(model)

    def clone(self, postfix="(clone)"):
        """DOES NOT clone the affiliated scene."""
        new = self.__class__(collision_type=self._collision_type,
                             is_floating=self.is_floating)
        new.name = self.name
        new.toggle_render_collision = self.toggle_render_collision
        new.file_path = self.file_path
        new.set_pos_rotmat(pos=self.pos,
                           rotmat=self.rotmat)
        new.set_inertia(self._inrtmat, self._com, self._mass)
        # clone all visuals
        for m in self.visuals:
            new.add_visual(m.clone(), auto_make_collision=False)
        # clone collisions if needed
        for c in self.collisions:
            new.add_collision(c.clone())
        return new

    def set_inertia(self, inrtmat=None, com=None, mass=None):
        if inrtmat is not None:
            self._inrtmat = inrtmat.copy()
        if com is not None:
            self._com = com.copy()
        if mass is not None:
            self._mass = mass

    @property
    def collision_group(self):
        return self._collision_group

    @collision_group.setter
    def collision_group(self, group):
        # The collision ROLE (ACTIVE / STATIC) is INDEPENDENT of is_floating: it is
        # seeded once at construction (floating -> ACTIVE, fixed -> STATIC), then
        # sticky. is_floating is physics only (free joint vs welded), so
        # grasping/mounting a body must NOT reclassify it -- a held object stays
        # ACTIVE and still collides with walls. A free body deliberately pinned as
        # a static obstacle declares STATIC explicitly, e.g.
        # ``obj.collision_group = CollisionGroup.STATIC``.
        self._collision_group = group

    @property
    def collision_affinity(self):
        return int(wuc.CollisionMatrix.DEFAULT[self.collision_group])

    @property
    def mounted_by(self):
        """The mech whose ``mount`` owns this object's pose, or None if free to
        be mounted. Orthogonal to ``is_floating``, which is physics only: a
        STATIC body is still perfectly mountable (a box picked off a table)."""
        return self._mounted_by

    @property
    def is_floating(self):
        return self._is_floating

    @is_floating.setter
    def is_floating(self, flag):
        # physics only (free joint vs welded). The collision role is decoupled --
        # see collision_group -- so this does NOT reclassify the body.
        self._is_floating = flag

    @property
    def rgb(self):
        if not self.visuals:
            return None
        return self.visuals[0].rgb

    @rgb.setter
    def rgb(self, value):
        for model in self.visuals:
            model.rgb = value

    @property
    def alpha(self):
        if not self.visuals:
            return None
        return self.visuals[0].alpha

    @alpha.setter
    def alpha(self, value):
        for model in self.visuals:
            model.alpha = value

    @property
    def rgba(self):
        if not self.visuals:
            return None
        m = self.visuals[0]
        return (*m.rgb, m.alpha)

    @rgba.setter
    def rgba(self, value):
        r, g, b, a = value
        for model in self.visuals:
            model.rgb = (r, g, b)
            model.alpha = a

    @property
    def inrtmat(self):
        if self._inrtmat is None:
            return None
        return self._inrtmat.copy()

    @property
    def com(self):
        if self._com is None:
            return None
        return self._com.copy()

    @property
    def mass(self):
        if self._mass is None:
            return None
        return self._mass

    def _auto_make_collision_from_model(self, m):
        if self._collision_type is None:
            print("Auto collision generation skipped for collision_type None.")
            return
        # if self.collisions: TODO: this check seems unnecessary?
        #     print("Auto collision generation skipped because collisions "
        #           "already exist.")
        #     return
        if self._collision_type == wuc.CollisionType.MESH:
            shape = wsc.MeshCollisionShape(file_path=self.file_path,
                                           geom=m.geom,
                                           rotmat=m.rotmat, pos=m.pos)
        elif self._collision_type == wuc.CollisionType.CVXHULL:
            shape = wsc.CvxHullCollisionShape.fit_from_geom(
                m.geom, m.rotmat, m.pos, file_path=self.file_path)
        elif self._collision_type == wuc.CollisionType.SPHERE:
            shape = wsc.SphereCollisionShape.fit_from_geom(
                m.geom, m.rotmat, m.pos)
        elif self._collision_type == wuc.CollisionType.CAPSULE:
            shape = wsc.CapsuleCollisionShape.fit_from_geom(
                m.geom, m.rotmat, m.pos)
        elif self._collision_type == wuc.CollisionType.CYLINDER:
            shape = wsc.CylinderCollisionShape.fit_from_geom(
                m.geom, m.rotmat, m.pos)
        elif self._collision_type == wuc.CollisionType.AABB:
            shape = wsc.AABBCollisionShape.fit_from_geom(
                m.geom, m.rotmat, m.pos)
        elif self._collision_type == wuc.CollisionType.OBB:
            shape = wsc.OBBCollisionShape.fit_from_geom(
                m.geom, m.rotmat, m.pos)
        elif self._collision_type == wuc.CollisionType.PLANE:
            shape = wsc.PlaneCollisionShape.fit_from_geom(
                m.geom, m.rotmat, m.pos)
        self.add_collision(shape)

    def _seed_collision_role(self):
        # Seed the collision role ONCE at construction from is_floating: a free body is
        # a manipulable ACTIVE object, a fixed body is STATIC scenery. This is a
        # smart two-way default -- it spares both objects and scenery an explicit
        # call. The role is then independent of is_floating (it does not re-run on
        # is_floating changes) and can be overridden via the collision_group setter.
        if self._is_floating:
            self._collision_group = wuc.CollisionGroup.ACTIVE
        else:
            self._collision_group = wuc.CollisionGroup.STATIC
