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

def serialize(model, model_id: str) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """One drawable as (metadata, {field: raw bytes}).

    Kept apart from the framing so the hub can cache a model and re-frame it
    for a viewer that connects later without ever decoding the floats.
    """
    geom = model.geom
    if geom.fs is None:
        vrgbs = model.vrgbs
        if vrgbs is None:
            raise ValueError(f"point cloud {model_id} has no per-vertex colors")
        meta = {'id': model_id, 'kind': 'pcd'}
        fields = {
            'points': np.ascontiguousarray(geom.vs, np.float32).tobytes(),
            'colors': np.ascontiguousarray(vrgbs, np.float32).tobytes(),
        }
        return meta, fields
    meta = {
        'id': model_id,
        'kind': 'mesh',
        'rgba': [float(model.rgb[0]), float(model.rgb[1]),
                 float(model.rgb[2]), float(model.alpha)],
    }
    # Ship the normals: vs_outline inflates the hull along them, so they have
    # to be geom.vns exactly -- anything the page recomputed for itself would
    # give the silhouette a different width.
    fields = {
        'vertices': np.ascontiguousarray(geom.vs, np.float32).tobytes(),
        'faces': np.ascontiguousarray(geom.fs, np.uint32).tobytes(),
        'normals': np.ascontiguousarray(geom.vns, np.float32).tobytes(),
    }
    return meta, fields


def collect_models(scene) -> List[Tuple[Dict[str, Any], Dict[str, bytes]]]:
    return [serialize(model, model_id)
            for model_id, model, _ in iter_scene_models(scene)]


# ---------------------------------------------------------------- messages

def scene_message(msg_type, models, camera=None, remove=None) -> bytes:
    """``scene_init`` or ``scene_delta`` -- models plus, for a delta, the ids
    that went away."""
    blob = _Blob()
    entries = []
    for meta, fields in models:
        entry = dict(meta)
        for name, raw in fields.items():
            entry[name] = blob.add_raw(raw, ARRAY_FIELDS[name])
        entries.append(entry)
    header = {'type': msg_type, 'models': entries}
    if camera is not None:
        header['camera'] = camera
    if remove is not None:
        header['remove'] = remove
    return pack(header, blob.parts)


def transform_message(scene) -> bytes:
    """Every model's matrix for this frame: ids in the header, matrices in one
    contiguous float32 run of 16 per id."""
    ids = []
    mats = []
    for model_id, model, sobj in iter_scene_models(scene):
        ids.append(model_id)
        mats.append((sobj.tf @ model.loc_tf).T)   # as render.py did
    blob = _Blob()
    stacked = (np.asarray(mats, dtype=np.float32).reshape(-1)
               if mats else np.empty(0, np.float32))
    ref = blob.add(stacked, np.float32)
    return pack({'type': 'scene_update', 'ids': ids, 'matrices': ref},
                blob.parts)


def split_models(header, blob) -> List[Tuple[Dict[str, Any], Dict[str, bytes]]]:
    """Frame -> the (metadata, {field: bytes}) pairs :func:`serialize` makes.

    Lets the hub keep a scene it never has to interpret: it slices the blob per
    model and can re-frame those slices for the next viewer verbatim.
    """
    out = []
    for entry in header.get('models', []):
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
