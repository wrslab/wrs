from wrs.scene.scene_object import SceneObject
from wrs.robots.base.mech_base import MechBase


class Scene:

    def __init__(self):
        self.dirty = True  # shader group needs update
        self._sobjs = []
        self._lnks = []
        self._mecbas = []

    def __iter__(self):  # for rendering order
        # a FLAT set: everything renderable is registered here directly. A
        # mounted child is added by ``mount`` itself, so it is in _sobjs too.
        yield from self._sobjs
        yield from self._lnks

    def add(self, entity):
        if isinstance(entity, SceneObject):
            if entity not in self._sobjs:
                self._sobjs.append(entity)
        elif isinstance(entity, MechBase):
            if entity not in self._mecbas:
                self._mecbas.append(entity)
                for lnk in entity.runtime_lnks:
                    if lnk not in self._lnks:
                        self._lnks.append(lnk)
        else:
            raise TypeError(f"Unsupported type: {type(entity)}")
        self.dirty = True

    def remove(self, entity):
        if isinstance(entity, SceneObject):
            if entity in self._sobjs:
                self._sobjs.remove(entity)
        elif isinstance(entity, MechBase):
            for lnk in entity.runtime_lnks:
                if lnk in self._lnks:
                    self._lnks.remove(lnk)
            if entity in self._mecbas:
                self._mecbas.remove(entity)
        self.dirty = True

    @property
    def sobjs(self):
        return tuple(self._sobjs)

    @property
    def lnks(self):
        return tuple(self._lnks)

    @property
    def mecbas(self):
        return tuple(self._mecbas)
