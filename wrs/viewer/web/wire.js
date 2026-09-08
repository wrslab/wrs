/**
 * Frame decoding, mirroring wrs/viewer/protocol.py.
 *
 *     [u32 header length][header JSON, space-padded to a multiple of 4][blob]
 *
 * The padding is the point: typed-array views must start on a 4-byte
 * boundary, so with an aligned blob every array in it can be read as a view
 * rather than copied out.  Geometry therefore goes from socket to GPU buffer
 * without ever becoming JavaScript numbers.
 *
 * Small control messages (status, event) stay JSON text and never reach here.
 */
const FIELD_TYPES = {
  vertices: Float32Array,
  normals: Float32Array,
  faces: Uint32Array,
  points: Float32Array,
  colors: Float32Array,
};

export function decode(buffer) {
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  const base = 4 + headerLength;
  const view = (ref, Type) => new Type(buffer, base + ref.off, ref.len);
  return { header, view };
}

/** One geometry entry plus its arrays, as Renderer.addGeometry wants it. */
export function readGeometry(entry, view) {
  const geometry = { id: entry.id, kind: entry.kind };
  Object.keys(FIELD_TYPES).forEach((name) => {
    if (entry[name]) geometry[name] = view(entry[name], FIELD_TYPES[name]);
  });
  return geometry;
}
