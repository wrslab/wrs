/**
 * A port of wrs/viewer/camera.py, so a drag in the page moves the viewpoint
 * the same way the same drag moves it in the native window.
 *
 * Every method here follows camera.py's arithmetic literally, signs included
 * -- that is the whole contract.  Anything "tidied up" on this side shows up
 * as a drag that behaves differently in the page than in the native window,
 * which is exactly the class of bug this file exists to rule out.
 */
import * as math from './math.js';

export class Camera {
  constructor({ pos = [2, 2, 2], lookAt = [0, 0, 0], up = [0, 0, 1],
                fov = 45, near = 0.01, far = 1000.0 } = {}) {
    this.pos = [...pos];
    this.lookAt = [...lookAt];
    this.up = this._fixUp([...up]);
    this.fov = fov;
    this.near = near;
    this.far = far;
  }

  setTo(pos, lookAt) {
    this.pos = [...pos];
    this.lookAt = [...lookAt];
    this.up = this._fixUp(this.up);
  }

  get basis() {
    return math.lookAtBasis(this.pos, this.lookAt, this.up);
  }

  writeView(out) {
    math.writeViewMatrix(out, this.pos, this.basis);
  }

  writeProj(out, aspect) {
    math.writePerspective(out, this.fov, aspect, this.near, this.far);
  }

  // ------------------------------------------------------------- navigation

  orbit(axis, angleRad) {
    const dir = math.rotateAboutAxis(
      math.sub(this.pos, this.lookAt), axis, angleRad);
    this.pos = math.add(this.lookAt, dir);
    this.up = this._fixUp(math.normalize(
      math.rotateAboutAxis(this.up, axis, angleRad)));
  }

  /** camera.mouse_orbit -- dy is expected already flipped to grow upwards. */
  mouseOrbit(dx, dy, sensitivity = 0.002) {
    const { right, up } = this.basis;
    this.orbit(up, -dx * sensitivity);
    this.orbit(right, dy * sensitivity);
  }

  mousePan(dx, dy, sensitivity = 0.0003) {
    const { right, up } = this.basis;
    const delta = math.add(math.scale(right, -dx * sensitivity),
                           math.scale(up, -dy * sensitivity));
    this.pos = math.add(this.pos, delta);
    this.lookAt = math.add(this.lookAt, delta);
  }

  /** Positive delta pushes the eye away from the target. */
  mouseZoom(delta, sensitivity = 0.05) {
    const dir = math.sub(this.pos, this.lookAt);
    this.pos = math.add(this.pos, math.scale(dir, delta * sensitivity));
  }

  // ------------------------------------------------------------------ guard

  /**
   * camera._fix_up_vector: swap the up axis when the view direction is about
   * to line up with it and the basis would collapse.
   *
   * Ported literally, scaling included: camera.py compares a dot product of
   * UNIT vectors against 0.99 * |look_at - pos| * |up|, so past a metre of
   * viewing distance the limit exceeds 1 and the swap can never trigger.  That
   * is the native behaviour, and matching it is the point of this file.
   */
  _fixUp(up) {
    const toTarget = math.sub(this.lookAt, this.pos);
    const fwdLength = math.len(toTarget);
    const fwd = math.normalize(toTarget);
    const upLength = math.len(up);
    const unitUp = math.normalize(up);
    if (math.dot(fwd, unitUp) > 0.99 * fwdLength * upLength) {
      return (Math.abs(unitUp[0]) < 1e-6 && Math.abs(unitUp[1]) < 1e-6 &&
              Math.abs(unitUp[2] - 1) < 1e-6) ? [1, 0, 0] : [0, 0, 1];
    }
    return unitUp;
  }
}
