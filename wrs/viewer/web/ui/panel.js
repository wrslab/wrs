/** Reusable panel shell. Its container owns positioning and its children own behavior. */
let nextPanelId = 0;
let nextPanelLayer = 10;
const ANCHORS = new Set(['inline', 'top-left', 'top-right', 'bottom-left', 'bottom-right']);

export class Panel {
  constructor({ title = '', description = '', container = null,
                anchor = 'inline', offset = 24, width = 296, height = null,
                fontSize = 13, closable = false, movable = false, visible = true } = {}) {
    this._events = new AbortController();
    this._children = new Set();
    this._destroyed = false;
    this._visible = true;
    this._contentVisible = true;
    this._position = null;
    this.container = null;
    this.element = document.createElement('aside');
    this.element.className = 'wrs-ui-panel';
    const bodyId = `wrs-panel-body-${++nextPanelId}`;
    this.element.innerHTML = `
      <header class="ui-header">
        <div class="ui-heading"><span class="ui-eyebrow">WRS / CONTROLS</span>
          <h2 class="ui-title"></h2></div>
        <div class="ui-header-actions"><button class="ui-collapse" type="button" aria-label="Collapse controls"
          aria-expanded="true" aria-controls="${bodyId}">
          <svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true">
            <path d="m5 12 5-5 5 5" fill="none" stroke="currentColor"
              stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
        <button class="ui-close" type="button" aria-label="Close panel" hidden>
          <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
            <path d="m6 6 8 8m0-8-8 8" fill="none" stroke="currentColor"
              stroke-width="1.5" stroke-linecap="round" />
          </svg>
        </button></div>
      </header>
      <div id="${bodyId}" class="ui-body">
        <p class="ui-description"></p>
        <div class="ui-controls"></div>
      </div>
      <footer class="ui-footer" hidden></footer>`;
    this.title = this.element.querySelector('.ui-title');
    this.description = this.element.querySelector('.ui-description');
    this.body = this.element.querySelector('.ui-body');
    this.controls = this.element.querySelector('.ui-controls');
    this.footer = this.element.querySelector('.ui-footer');
    this.header = this.element.querySelector('.ui-header');
    this.header.tabIndex = -1;
    this.closeButton = this.element.querySelector('.ui-close');
    this.closeButton.addEventListener('click', () => this.hide(), { signal: this._events.signal });
    const collapse = this.element.querySelector('.ui-collapse');
    collapse.addEventListener('click', () => {
      const collapsed = collapse.getAttribute('aria-expanded') === 'true';
      collapse.setAttribute('aria-expanded', String(!collapsed));
      collapse.setAttribute('aria-label', collapsed ? 'Expand controls' : 'Collapse controls');
      this.body.hidden = collapsed;
      this._collapsed = collapsed;
      this._applyLayout();
    }, { signal: this._events.signal });
    this.header.addEventListener('pointerdown', event => this._startDrag(event), { signal: this._events.signal });
    this.header.addEventListener('pointermove', event => {
      if (this._drag?.id !== event.pointerId) return;
      this._position = { x: this._drag.x + event.clientX - this._drag.clientX,
        y: this._drag.y + event.clientY - this._drag.clientY };
      this._placePosition();
    }, { signal: this._events.signal });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      this.header.addEventListener(type, event => {
        if (this._drag?.id === event.pointerId) this._endDrag();
      }, { signal: this._events.signal });
    }
    window.addEventListener('resize', () => this._placePosition(), { signal: this._events.signal });
    this.update({ title, description, fontSize, closable, movable, visible });
    this.setLayout({ anchor, offset, width, height });
    if (container) this.mount(container);
  }

  update({ title, description, fontSize, closable, movable, visible } = {}) {
    for (const [name, value] of Object.entries({ closable, movable, visible })) {
      if (value !== undefined && typeof value !== 'boolean') throw new TypeError(`${name} must be a boolean`);
    }
    if (fontSize !== undefined) {
      if (!Number.isFinite(fontSize) || fontSize <= 0) {
        throw new RangeError('fontSize must be positive');
      }
      this.element.style.setProperty('--ui-font-size', `${fontSize}px`);
    }
    if (title !== undefined) {
      this.title.textContent = title;
      this.element.setAttribute('aria-label', title || 'Controls');
    }
    if (description !== undefined) {
      this.description.textContent = description;
      this.description.hidden = !description;
    }
    if (closable !== undefined) this.closeButton.hidden = !closable;
    if (movable !== undefined) {
      this.movable = movable;
      if (!movable) this._endDrag();
    }
    if (visible !== undefined) this._visible = visible;
    this._applyVisibility();
    if (this.layout) this._applyLayout();
  }

  show() { this.update({ visible: true }); return this; }
  hide() { this.update({ visible: false }); return this; }

  _applyVisibility() {
    this.element.hidden = !this._visible || !this._contentVisible;
    if (this.element.hidden) {
      this._endDrag();
      if (this.element.contains(document.activeElement)) document.activeElement.blur();
    }
  }

  add(control) {
    if (this._destroyed) throw new Error('cannot add to a destroyed panel');
    control._panel?._children.delete(control);
    control._panel = this;
    this._children.add(control);
    this.controls.appendChild(control.element);
    return control;
  }

  mount(container) {
    if (this._destroyed) throw new Error('cannot mount a destroyed panel');
    if (!(container instanceof Element)) throw new TypeError('container must be a DOM element');
    this._endDrag();
    this._position = null;
    this.container = container;
    container.appendChild(this.element);
    this._applyLayout();
    return this;
  }

  setLayout(options = {}) {
    const layout = { ...this.layout, ...options };
    if (!ANCHORS.has(layout.anchor)) throw new RangeError('unknown panel anchor');
    if (!Number.isFinite(layout.offset) || layout.offset < 0
        || !Number.isFinite(layout.width) || layout.width <= 0) {
      throw new RangeError('offset must be nonnegative and width must be positive');
    }
    if (layout.height !== null && (!Number.isFinite(layout.height) || layout.height <= 0)) {
      throw new RangeError('height must be positive or null');
    }
    if (this.layout && ['anchor', 'offset', 'width', 'height'].some(key => layout[key] !== this.layout[key])) {
      this._endDrag();
      this._position = null;
    }
    this.layout = layout;
    this._applyLayout();
    return this;
  }

  _applyLayout() {
    const { anchor, offset, width, height } = this.layout;
    const style = this.element.style;
    const inline = anchor === 'inline';
    const viewport = this.container === document.body;
    style.position = inline ? 'relative' : (viewport ? 'fixed' : 'absolute');
    style.width = `${width}px`;
    style.height = height === null || this._collapsed ? 'auto' : `${height}px`;
    style.top = style.bottom = style.left = style.right = 'auto';
    style.maxWidth = inline ? '100%' : `calc(100% - ${offset * 2}px)`;
    style.maxHeight = inline ? 'none' : `calc(${viewport ? '100dvh' : '100%'} - ${offset * 2}px)`;
    if (!inline) {
      const [vertical, horizontal] = anchor.split('-');
      style[vertical] = `${offset}px`;
      style[horizontal] = `${offset}px`;
    }
    this.header.dataset.movable = String(this._canMove());
    this.header.title = this._canMove() ? 'Drag to move' : '';
    this._placePosition();
  }

  _canMove() {
    return Boolean(this.movable && this.container && this.layout?.anchor !== 'inline');
  }

  _bounds() {
    if (this.container === document.body) {
      return { left: 0, top: 0, x: 0, y: 0,
        width: document.documentElement.clientWidth, height: window.innerHeight };
    }
    const rect = this.container.getBoundingClientRect();
    return { left: rect.left + this.container.clientLeft - this.container.scrollLeft,
      top: rect.top + this.container.clientTop - this.container.scrollTop,
      x: this.container.scrollLeft, y: this.container.scrollTop,
      width: this.container.clientWidth, height: this.container.clientHeight };
  }

  _readPosition() {
    const rect = this.element.getBoundingClientRect(), bounds = this._bounds();
    return { x: rect.left - bounds.left, y: rect.top - bounds.top };
  }

  _placePosition() {
    if (!this._position || !this.container || this.element.hidden || !this.element.isConnected) return;
    const bounds = this._bounds(), rect = this.element.getBoundingClientRect();
    this._position.x = Math.max(bounds.x, Math.min(this._position.x,
      bounds.x + Math.max(0, bounds.width - rect.width)));
    this._position.y = Math.max(bounds.y, Math.min(this._position.y,
      bounds.y + Math.max(0, bounds.height - rect.height)));
    Object.assign(this.element.style, { left: `${this._position.x}px`, top: `${this._position.y}px`,
      right: 'auto', bottom: 'auto' });
  }

  _startDrag(event) {
    if (!this._canMove() || this._drag || event.button !== 0 || !event.isPrimary
        || event.target.closest('button, input, select, textarea, a')) return;
    event.preventDefault();
    this.element.style.zIndex = String(++nextPanelLayer);
    this.header.focus({ preventScroll: true });
    const pos = this._readPosition();
    this._drag = { id: event.pointerId, clientX: event.clientX, clientY: event.clientY, ...pos };
    this.header.setPointerCapture(event.pointerId);
    this.header.dataset.dragging = 'true';
  }

  _endDrag() {
    const drag = this._drag;
    this._drag = null;
    if (drag && this.header.hasPointerCapture(drag.id)) this.header.releasePointerCapture(drag.id);
    if (this.header) delete this.header.dataset.dragging;
  }

  destroy() {
    if (this._destroyed) return;
    this._endDrag();
    this._destroyed = true;
    this._events.abort();
    for (const child of this._children) child.destroy();
    this._children.clear();
    this.element.remove();
    this.container = null;
  }
}
