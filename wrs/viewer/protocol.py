"""Scene -> wire.

The page never sees a SceneObject; it sees a flat list of drawables, each with
a stable id, plus a matrix per id per frame.  Everything that decides what that
looks like lives here, so the server is pure transport.

**Frame format.**  Geometry is bulk float data, and as JSON text it is roughly
eight times larger than it needs to be -- a robot cell clears 80 MB, which the
browser then has to parse.  So every scene message is one binary frame::

    [u32 header length][header JSON, space-padded to a multiple of 4][blob]

The header names each array and where it sits in the blob (``off`` in bytes,
``len`` in elements).  The padding is what lets the page point a Float32Array
straight at the buffer: typed-array views must start on a 4-byte boundary, and
every array here has 4-byte elements, so once the blob starts aligned all of
them do.  Small control messages (``status``, ``event``) stay JSON text; the
page tells them apart by frame type.

Two conventions the page depends on:

* **Matrices go out already transposed.**  ``(sobj.tf @ model.loc_tf).T``
  raveled row-major is byte-for-byte the column-major mat4x4 WGSL wants, so the
  browser feeds what it receives straight into a GPU buffer.
* **A model is a mesh unless ``geom.fs`` is None**, which marks a point cloud --
  the same test render.py used to split its pipelines.
"""
import itertools
import json
import struct
import weakref
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

DEFAULT_PORT = 8000

# Close code the hub uses to retire a publisher when a newer script takes
# over.  It has to be distinguishable from an ordinary drop: a plain close
# means "hub restarting, try again", while this one means "stop trying".
SUPERSEDED = 4000

# field -> numpy dtype.  The page mirrors this when it builds its views.
ARRAY_FIELDS = {
    'vertices': np.float32,
    'normals': np.float32,
    'faces': np.uint32,
    'points': np.float32,
    'colors': np.float32,
}


# ------------------------------------------------------------------- framing

def pack(header: Dict[str, Any], arrays: List[bytes]) -> bytes:
    """One binary frame from a header and the blobs its offsets refer to."""
    raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
    raw += b' ' * (-len(raw) % 4)      # keep the blob 4-byte aligned
    return b''.join([struct.pack('<I', len(raw)), raw] + arrays)


def unpack(frame: bytes) -> Tuple[Dict[str, Any], memoryview]:
    """Inverse of :func:`pack`: the header and a view on the blob."""
    (header_len,) = struct.unpack_from('<I', frame, 0)
    header = json.loads(bytes(frame[4:4 + header_len]).decode('utf-8'))
    return header, memoryview(frame)[4 + header_len:]


class _Blob:
    """Accumulates arrays and hands back the {off, len} the header records."""

    def __init__(self):
        self.parts: List[bytes] = []
        self.size = 0

    def add(self, array: np.ndarray, dtype) -> Dict[str, int]:
        raw = np.ascontiguousarray(array, dtype=dtype).tobytes()
        ref = {'off': self.size, 'len': len(raw) // np.dtype(dtype).itemsize}
        self.parts.append(raw)
        self.size += len(raw)
        return ref

    def add_raw(self, raw: bytes, dtype) -> Dict[str, int]:
        ref = {'off': self.size, 'len': len(raw) // np.dtype(dtype).itemsize}
        self.parts.append(raw)
        self.size += len(raw)
        return ref


# ------------------------------------------------------------- scene walking

# Scene objects get a serial number rather than being keyed by id(): CPython
# reuses the address of a collected object, so an object removed and another
# allocated between two ticks could land on the same id() and the delta the
# publisher computes would see no change at all.  A weak key means the entry
# disappears with the object, and the counter never repeats a number.
_sobj_serials = weakref.WeakKeyDictionary()
_serial_counter = itertools.count()


def _sobj_key(sobj) -> int:
    key = _sobj_serials.get(sobj)
    if key is None:
        key = next(_serial_counter)
        _sobj_serials[sobj] = key
    return key


def iter_scene_models(scene) -> Iterator[Tuple[str, Any, Any]]:
    """Walk the scene as (model_id, model, owner) -- no serialization.

    Cheap enough to run every tick, which is what lets the stream notice
    objects added or removed after run() rather than only at connect time.
    """
    for sobj in scene:
        for idx, model in enumerate(sobj.visuals):
            yield f"{_sobj_key(sobj)}:visual:{idx}", model, sobj
        # collision visualization (if enabled); these are always meshes
        if sobj.toggle_render_collision:
            for idx, col in enumerate(sobj.collisions):
                yield (f"{_sobj_key(sobj)}:collision:{idx}",
                       col.to_render_model(), sobj)


# ------------------------------------------------------------- serialization

# Geometry is shared: robot.clone() reuses its meshes, so a scene showing 15
# ghost poses holds 112 models over 7 distinct geoms.  Numbering the geoms lets
# each one travel once and every model that uses it just name it -- the same
# trick render.py played with its device-buffer cache.
_geom_serials = weakref.WeakKeyDictionary()
_geom_counter = itertools.count()


def _geom_key(geom) -> int:
    key = _geom_serials.get(geom)
    if key is None:
        key = next(_geom_counter)
        _geom_serials[geom] = key
    return key


def model_entry(model, model_id: str) -> Dict[str, Any]:
    """A drawable with no arrays of its own: which geometry, and what colour."""
    entry = {'id': model_id, 'geom': _geom_key(model.geom),
             'kind': 'pcd' if model.geom.fs is None else 'mesh'}
    if entry['kind'] == 'mesh':
        entry['rgba'] = [float(model.rgb[0]), float(model.rgb[1]),
                         float(model.rgb[2]), float(model.alpha)]
    return entry


def geometry_entry(model) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """That geometry's arrays, as (metadata, {field: raw bytes}).

    Kept apart from the framing so the hub can cache one and re-frame it for a
    viewer that connects later without ever decoding the floats.
    """
    geom = model.geom
    meta = {'id': _geom_key(geom),
            'kind': 'pcd' if geom.fs is None else 'mesh'}
    if geom.fs is None:
        vrgbs = model.vrgbs
        if vrgbs is None:
            raise ValueError('point cloud has no per-vertex colors')
        return meta, {
            'points': np.ascontiguousarray(geom.vs, np.float32).tobytes(),
            'colors': np.ascontiguousarray(vrgbs, np.float32).tobytes(),
        }
    # Ship the normals: vs_outline inflates the hull along them, so they have
    # to be geom.vns exactly -- anything the page recomputed for itself would
    # give the silhouette a different width.
    return meta, {
        'vertices': np.ascontiguousarray(geom.vs, np.float32).tobytes(),
        'faces': np.ascontiguousarray(geom.fs, np.uint32).tobytes(),
        'normals': np.ascontiguousarray(geom.vns, np.float32).tobytes(),
    }


def describe(pairs, known_geoms) -> Tuple[List[Dict[str, Any]],
                                          List[Tuple[Dict[str, Any],
                                                     Dict[str, bytes]]]]:
    """(model_id, model) pairs -> the entries a message carries.

    ``known_geoms`` is the set of geometry ids the far end already holds; it is
    updated here, so each geometry is serialized once per connection.
    """
    models, geometries = [], []
    for model_id, model in pairs:
        entry = model_entry(model, model_id)
        models.append(entry)
        if entry['geom'] not in known_geoms:
            known_geoms.add(entry['geom'])
            geometries.append(geometry_entry(model))
    return models, geometries


# ---------------------------------------------------------------- messages

def scene_message(msg_type, models, geometries,
                  camera=None, remove=None, replay=False) -> bytes:
    """``scene_init`` or ``scene_delta``: any geometry the far end is missing,
    the models that reference it, and (for a delta) the ids that went away.

    ``replay`` marks the hub catching a newly connected page up on a scene
    that was already running, as opposed to a script publishing a fresh one.
    The page needs the difference: a reload should keep the viewpoint you
    dragged to, while a script starting should get the camera it asked for.
    """
    blob = _Blob()
    geometry_entries = []
    for meta, fields in geometries:
        entry = dict(meta)
        for name, raw in fields.items():
            entry[name] = blob.add_raw(raw, ARRAY_FIELDS[name])
        geometry_entries.append(entry)
    header = {'type': msg_type, 'geometries': geometry_entries,
              'models': list(models)}
    if replay:
        header['replay'] = True
    if camera is not None:
        header['camera'] = camera
    if remove is not None:
        header['remove'] = remove
    return pack(header, blob.parts)


def split_geometries(header, blob) -> List[Tuple[Dict[str, Any],
                                                 Dict[str, bytes]]]:
    """Frame -> the (metadata, {field: bytes}) pairs geometry_entry makes."""
    out = []
    for entry in header.get('geometries', []):
        meta = {k: v for k, v in entry.items() if k not in ARRAY_FIELDS}
        fields = {}
        for name in ARRAY_FIELDS:
            ref = entry.get(name)
            if ref is None:
                continue
            size = ref['len'] * np.dtype(ARRAY_FIELDS[name]).itemsize
            fields[name] = bytes(blob[ref['off']:ref['off'] + size])
        out.append((meta, fields))
    return out


def transform_arrays(snapshot) -> Tuple[List[str], np.ndarray]:
    """A snapshot as (ids, (N, 16) float32) -- one column-major matrix a row.

    Split out from the message so the caller can diff against the previous
    frame: a scene is mostly furniture, and re-sending every matrix at 30 Hz
    costs more than the whole rest of the viewer put together.
    """
    ids = []
    mats = np.empty((len(snapshot), 16), dtype=np.float32)
    for i, (model_id, model, sobj) in enumerate(snapshot):
        ids.append(model_id)
        mats[i] = (sobj.tf @ model.loc_tf).T.reshape(16)   # as render.py did
    return ids, mats


MATRIX_BYTES = 64          # 16 float32


def transform_message(ids, matrices) -> bytes:
    """Poses for the ids given: ids in the header, matrices in one contiguous
    float32 run of 16 per id.  A partial set is fine -- the page applies what
    it is sent and leaves the rest alone."""
    raw = (matrices if isinstance(matrices, bytes)
           else np.ascontiguousarray(matrices, np.float32).reshape(-1).tobytes())
    blob = _Blob()
    ref = blob.add_raw(raw, np.float32)
    return pack({'type': 'scene_update', 'ids': list(ids), 'matrices': ref},
                blob.parts)


def split_transforms(header, blob):
    """A pose frame -> {id: raw 64 bytes}, for a cache that has to survive
    partial updates."""
    off = header['matrices']['off']
    return {model_id: bytes(blob[off + i * MATRIX_BYTES:
                                 off + (i + 1) * MATRIX_BYTES])
            for i, model_id in enumerate(header['ids'])}
