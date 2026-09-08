import numpy as np
import wrs.utils.math as wum
import wrs.utils.decorator as wud
import wrs.geom.geometry as wgg


class RenderModel:
    """
    rotmat and pos of model is for transforming local geometries
    it is intended to be immutable after creation.
    runtime pose updates must go through the owning SceneObject.
    """

    def __init__(self,
                 geom=None,
                 rotmat=None,
                 pos=None,
                 rgb=None,
                 alpha=1.0,
                 shader=None,
                 **kwargs):
        if isinstance(geom, tuple):
            self.geom = wgg.gen_geom_from_raw(*geom)
        else:
            self.geom = geom
        self.shader = shader
        self._rgb = wum.ensure_rgb(rgb)
        self._vrgbs = kwargs.get("vrgbs", None)
        self._alpha = alpha
        self._rotmat = wum.ensure_rotmat(rotmat)
        self._pos = wum.ensure_pos(pos)
        # cached
        self._loc_tf = wum.tf_from_pos_rotmat(self._pos, self._rotmat)
        self._dirty = True

    def clone(self):
        new = self.__class__(geom=self.geom,
                             rotmat=self._rotmat.copy(),
                             pos=self._pos.copy(),
                             rgb=self.rgb.copy(),
                             alpha=self.alpha,
                             shader=self.shader)
        return new

    @property
    def vrgbs(self):
        """Per-vertex colours, or None for anything but a point cloud."""
        return self._vrgbs

    @property
    def rgb(self):
        return self._rgb.copy()

    @rgb.setter
    def rgb(self, rgb):
        self._rgb = wum.ensure_rgb(rgb)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def quat(self):
        # TODO cache?
        return wum.quat_from_rotmat(self._rotmat)

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
    @wud.lazy_update("_dirty", "_rebuild_tf")
    def loc_tf(self):
        """Pose relative to the owning SceneObject."""
        return self._loc_tf.copy()

    @wud.mark_dirty('_mark_dirty')
    def set_pos_rotmat(self, pos, rotmat):
        self._rotmat[:] = wum.ensure_rotmat(rotmat)
        self._pos[:] = wum.ensure_pos(pos)

    def _rebuild_tf(self):
        if not self._dirty:
            return
        self._loc_tf[:] = np.eye(4, dtype=np.float32)
        self._loc_tf[:3, :3] = self._rotmat
        self._loc_tf[:3, 3] = self._pos
        self._dirty = False

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
