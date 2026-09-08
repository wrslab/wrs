/**
 * Just enough vector / matrix math for the viewer -- a port of the pieces of
 * wrs/utils/math.py and wrs/viewer/camera.py that the page needs, so nothing
 * here depends on a matrix library.
 *
 * Matrices are Float32Array(16) in COLUMN-MAJOR order, which is what WGSL's
 * mat4x4 reads and, conveniently, what the wire already carries: the server
 * sends (sobj.tf @ model.loc_tf).T raveled, and transposing a row-major matrix
 * yields exactly the column-major bytes -- so instance transforms go from the
 * socket into the GPU buffer untouched.
 */

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const len = (a) => Math.hypot(a[0], a[1], a[2]);

const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export function normalize(a) {
  const l = len(a);
  return l > 1e-12 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}

/** Rodrigues rotation -- the wum.rotmat_from_axangle(axis, angle) @ v of camera.orbit. */
export function rotateAboutAxis(v, axis, angle) {
  const k = normalize(axis);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const kv = dot(k, v);
  return add(add(scale(v, c), scale(cross(k, v), s)), scale(k, kv * (1 - c)));
}

/**
 * wum.rotmat_from_look_at: the camera basis, whose columns are right, up, -fwd.
 * Returned separately because the view matrix and the drag handlers both want
 * the axes rather than the assembled matrix.
 */
export function lookAtBasis(pos, lookAt, up) {
  const fwd = normalize(sub(lookAt, pos));
  const right = normalize(cross(fwd, up));
  return { right, up: cross(right, fwd), fwd };
}

/**
 * camera.py's view_mat = tf_inverse(tf), written column-major.
 *
 * tf is [R | pos] with R's columns right/up/-fwd, so the inverse is
 * [R^T | -R^T pos] -- R^T's ROWS are those same axes, which in column-major
 * storage means each axis lands in a strided slot rather than contiguously.
 */
export function writeViewMatrix(out, pos, basis) {
  const { right, up, fwd } = basis;
  out[0] = right[0]; out[4] = right[1]; out[8] = right[2];
  out[1] = up[0]; out[5] = up[1]; out[9] = up[2];
  out[2] = -fwd[0]; out[6] = -fwd[1]; out[10] = -fwd[2];
  out[3] = 0; out[7] = 0; out[11] = 0;
  out[12] = -dot(right, pos);
  out[13] = -dot(up, pos);
  out[14] = dot(fwd, pos);
  out[15] = 1;
}

/**
 * camera.perspective(): right-handed, looking down -z, depth mapped to [0, 1].
 * That depth range is why this cannot be a stock WebGL projection -- GL clips
 * z to [-1, 1], WebGPU (and wgpu) to [0, 1].
 */
export function writePerspective(out, fovDeg, aspect, near, far) {
  out.fill(0);
  const f = 1.0 / Math.tan((fovDeg * Math.PI) / 360.0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1.0;
  out[14] = (far * near) / (near - far);
}

/** sRGB colour -> the linear value an sRGB render target must be cleared to. */
export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
