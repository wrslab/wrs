"""Scene camera.

Right-handed, looking down its own -z.  The projection maps depth to
[0, 1], the range wgpu clip space uses.
"""
import numpy as np
import wrs.utils.math as wum
import wrs.utils.decorator as wud


class Camera:
    """The scene camera. Self-contained: it carries its own pose (``tf``, world)
    rather than borrowing a scene-object base, because a camera is not a body in
    the scene -- it is never added to a Scene, mounted, or rendered."""

    def __init__(self,
                 pos=(2, 2, 2),
                 look_at=(0, 0, 0),
                 up=(0, 0, 1),
                 fov=45,
                 aspect=1.7778,
                 near=0.01,
                 far=1000.0):
        self._pos = np.asarray(pos, dtype=np.float32)
        self._look_at = np.asarray(look_at, dtype=np.float32)
        self._up = np.asarray(
            self._fix_up_vector(self._pos, self._look_at, up),
            dtype=np.float32)
        self._rotmat = wum.rotmat_from_look_at(
            pos=self._pos, look_at=self._look_at, up=self._up)
        self._tf = wum.tf_from_pos_rotmat(self._pos, self._rotmat)
        self._dirty = True
        self._fov = fov
        self._aspect = aspect  # default 16:9
        self._near = near
        self._far = far
        # cached
        self._proj_mat = None
        self._proj_dirty = True

    def set_to(self, pos, look_at, up=None):
        self.pos = np.asarray(pos, dtype=np.float32)
        self.look_at = np.asarray(look_at, dtype=np.float32)
        if up is not None:
            self.up = np.asarray(up, dtype=np.float32)

    def orbit(self, axis=(0, 0, 1), angle_rad=np.pi / 360):
        direction = self._pos - self._look_at
        rotmat = wum.rotmat_from_axangle(axis, angle_rad)
        direction_rotated = rotmat @ direction
        self._pos = self._look_at + direction_rotated
        self._up = (rotmat @ self._up)
        self._up /= np.linalg.norm(self._up)
        self._up = self._fix_up_vector(self._pos, self._look_at, self._up)
        self._dirty = True

    def mouse_orbit(self, dx, dy, sensitivity=0.002):
        right_axis = self.tf[:3, 0]
        up_axis = self.tf[:3, 1]
        self.orbit(axis=up_axis, angle_rad=-dx * sensitivity)
        self.orbit(axis=right_axis, angle_rad=dy * sensitivity)

    def mouse_pan(self, dx, dy, sensitivity=0.0003):
        right_axis = self.tf[:3, 0]
        up_axis = self.tf[:3, 1]
        self.pos = self.pos - right_axis * dx * sensitivity - up_axis * dy * sensitivity
        self.look_at = self.look_at - right_axis * dx * sensitivity - up_axis * dy * sensitivity

    def mouse_zoom(self, delta, sensitivity=0.05):
        direction = self.pos - self.look_at
        zoom_amount = delta * sensitivity
        self.pos = self.pos + direction * zoom_amount

    @property
    def pos(self):
        return self._pos.copy()

    @pos.setter
    @wud.mark_dirty('_mark_dirty')
    def pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float32)

    @property
    def rotmat(self):
        return self._rotmat.copy()

    @rotmat.setter
    def rotmat(self, rotmat):
        """Disable direct setting of rotmat on Camera -- it is derived from
        pos / look_at / up."""
        raise AttributeError("Cannot set rotmat directly on Camera. Use set_to() method instead.")

    @property
    @wud.lazy_update('_dirty', '_rebuild_tf')
    def tf(self):
        """The camera's pose in the world."""
        return self._tf.copy()

    def _mark_dirty(self):
        self._dirty = True

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    @wud.mark_dirty('_mark_dirty')
    def look_at(self, look_at):
        self._look_at = np.asarray(look_at, dtype=np.float32)

    @property
    def up(self):
        return self._up

    @up.setter
    @wud.mark_dirty('_mark_dirty')
    def up(self, up):
        self._up = np.asarray(up, dtype=np.float32)

    @property
    def fov(self):
        return self._fov

    @fov.setter
    @wud.mark_dirty('_proj_dirty')
    def fov(self, fov):
        self._fov = fov

    @property
    def near(self):
        return self._near

    @near.setter
    @wud.mark_dirty('_proj_dirty')
    def near(self, near):
        self._near = near

    @property
    def far(self):
        return self._far

    @far.setter
    @wud.mark_dirty('_proj_dirty')
    def far(self, far):
        self._far = far

    # getters for matrices, setting matrices should be done via other methods
    @property
    @wud.lazy_update('_dirty', '_rebuild_tf')
    def view_mat(self):
        return wum.tf_inverse(self._tf)

    @property
    @wud.lazy_update('_proj_dirty', '_rebuild_projmat')
    def proj_mat(self):
        return self._proj_mat

    def _mark_proj_dirty(self):
        self._proj_dirty = True

    def _fix_up_vector(self, pos, look_at, up):
        # TODO: elevate to utils.math
        fwd_length, fwd = wum.unit_vec(look_at - pos)
        up_length, up = wum.unit_vec(up)
        dot_val = np.dot(fwd, up)
        limit = 0.99 * (fwd_length * up_length)
        if dot_val > limit:
            if np.allclose(up, (0, 0, 1)):
                up = (1, 0, 0)
            else:
                up = (0, 0, 1)
        return up

    def _rebuild_tf(self):
        if not self._dirty:
            return
        self._up = np.asarray(self._fix_up_vector(self._pos, self._look_at, self._up),
                              dtype=np.float32)
        self._rotmat = wum.rotmat_from_look_at(pos=self._pos,
                                               look_at=self._look_at,
                                               up=self._up)
        self._tf[:3, :3] = self._rotmat
        self._tf[:3, 3] = self._pos
        self._dirty = False

    def _rebuild_projmat(self, width=None, height=None):
        if width is not None and height is not None:
            self._aspect = width / height
        self._proj_mat = perspective(
            self._fov, self._aspect, self._near, self._far)
        self._proj_dirty = False


def perspective(fov_deg, aspect, near, far):
    """Right-handed perspective looking down -z, depth mapped to [0, 1]."""
    f = 1.0 / np.tan(np.radians(fov_deg) * 0.5)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = far / (near - far)
    mat[2, 3] = (far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat
