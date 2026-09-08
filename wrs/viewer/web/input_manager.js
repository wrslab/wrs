/**
 * Pointer bindings for Camera, mirroring wrs/viewer/input_manager.py.
 *
 * input_manager.py flips dy because its events count y downwards and the camera
 * wants a conventional right-handed drag; DOM pointer events count y downwards
 * too, so the same flip applies here and the drag arithmetic in camera.js is
 * shared verbatim with the native window.
 *
 * Buttons follow the native window (right orbits, middle pans) but left also
 * orbits: a page where the primary button does nothing reads as broken, and
 * the native viewer has no such expectation to honour.
 */
const ORBIT_BUTTONS = new Set([0, 2]);
const PAN_BUTTON = 1;

export class InputManager {
  constructor(camera, domElement, { onEvent = null } = {}) {
    this.camera = camera;
    this.domElement = domElement;
    this.onEvent = onEvent;
    this._buttons = new Set();
    this._lastX = 0;
    this._lastY = 0;
    this._bind();
  }

  _bind() {
    const el = this.domElement;
    el.addEventListener('pointerdown', (e) => this._onDown(e));
    el.addEventListener('pointermove', (e) => this._onMove(e));
    el.addEventListener('pointerup', (e) => this._onUp(e));
    el.addEventListener('pointercancel', (e) => this._onUp(e));
    el.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    // on window, not the canvas: a canvas only receives key events while
    // focused, and nothing here ever asks for focus
    window.addEventListener('keydown', (e) => this._onKey('on_key_press', e));
    window.addEventListener('keyup', (e) => this._onKey('on_key_release', e));
  }

  /**
   * Keys are not consumed here -- they go back to the script, which is what
   * @base.event on_key_press subscribes to.
   *
   * event.key is passed through untouched: it is a JS KeyboardEvent name,
   * which is exactly what viewer.key.symbol_from_name reads, so the symbol
   * table stays on the Python side alone.  Auto-repeat is dropped, so one
   * press is one event however long the key is held.
   */
  _onKey(name, e) {
    if (e.repeat || !this.onEvent) return;
    this.onEvent(name, e.key);
  }

  _onDown(e) {
    this._buttons.add(e.button);
    this._lastX = e.clientX;
    this._lastY = e.clientY;
    // keep receiving moves when the drag leaves the canvas, which the old
    // mouseleave-cancels-the-drag handling could not do
    this.domElement.setPointerCapture(e.pointerId);
  }

  _onMove(e) {
    if (this._buttons.size === 0) return;
    const dx = e.clientX - this._lastX;
    const dy = -(e.clientY - this._lastY);  // events count y downwards
    this._lastX = e.clientX;
    this._lastY = e.clientY;
    if (this._buttons.has(PAN_BUTTON)) {
      this.camera.mousePan(dx, dy);
    } else if ([...this._buttons].some((b) => ORBIT_BUTTONS.has(b))) {
      this.camera.mouseOrbit(dx, dy);
    }
  }

  _onUp(e) {
    this._buttons.delete(e.button);
    if (this._buttons.size === 0) {
      this.domElement.releasePointerCapture(e.pointerId);
    }
  }

  _onWheel(e) {
    e.preventDefault();
    // input_manager passes event dy / 100; a forward scroll is negative in
    // both worlds, and a negative delta pulls the eye towards the target.
    this.camera.mouseZoom(e.deltaY / 100.0);
  }
}
