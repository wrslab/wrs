/** Native controls: DOM and callbacks only, independent of the viewer transport. */
let nextControlId = 0;

class Control {
  constructor(kind, props) {
    this.element = document.createElement('div');
    this.element.className = `ui-control ui-${kind}`;
    this.control = { id: '', label: '', enabled: true, ...props, kind };
    this.pending = false;
    this.editing = false;
    this._events = new AbortController();
  }

  _listen(element, event, callback) {
    element.addEventListener(event, callback, { signal: this._events.signal });
  }

  _bindBusyGuard(allowPending = () => false) {
    this._listen(this.input, 'pointerdown', (event) => {
      if (this.pending && !allowPending()) event.preventDefault();
    });
    this._listen(this.input, 'keydown', (event) => {
      if (this.pending && !allowPending() && event.key !== 'Tab') event.preventDefault();
    });
  }

  _updateInput(allowPending = false) {
    // aria-disabled preserves keyboard focus while a request is pending.
    this.input.disabled = !this.control.enabled;
    this.input.setAttribute('aria-disabled', String(this.input.disabled || (this.pending && !allowPending)));
    this.input.setAttribute('aria-busy', String(this.pending));
  }

  destroy() {
    this._panel?._children.delete(this);
    this._panel = null;
    this._events.abort();
    this.element.remove();
  }
}

export class Button extends Control {
  constructor({ onClick = () => {}, ...props } = {}) {
    super('button', props);
    this.input = document.createElement('button');
    this.input.type = 'button';
    this.input.className = 'ui-action';
    this.element.appendChild(this.input);
    this._listen(this.input, 'click', () => {
      if (this.control.enabled && !this.pending) onClick();
    });
    this._bindBusyGuard();
    this.update();
  }

  update(props = {}) {
    Object.assign(this.control, props);
    this._updateInput();
    this.input.textContent = this.pending ? `${this.control.label}…` : this.control.label;
  }
}

export class Slider extends Control {
  constructor({ onChange = () => {}, ...props } = {}) {
    super('slider', { min: 0, max: 1, step: 0.01, value: 0, unit: '',
      continuous: false, update_hz: 30, ...props });
    this._onChange = onChange;
    this._lastEmitAt = -Infinity;
    const top = document.createElement('div');
    top.className = 'ui-slider-top';
    this.label = document.createElement('label');
    this.input = document.createElement('input');
    this.input.type = 'range';
    this.input.id = `wrs-slider-${++nextControlId}`;
    this.label.htmlFor = this.input.id;
    this.output = document.createElement('output');
    this.output.htmlFor = this.input.id;
    top.append(this.label, this.output);
    const bounds = document.createElement('div');
    bounds.className = 'ui-bounds';
    this.min = document.createElement('span');
    this.max = document.createElement('span');
    bounds.append(this.min, this.max);
    this.element.append(top, this.input, bounds);
    this._listen(this.input, 'input', () => {
      if (!this.control.enabled) return;
      this.editing = true;
      this._display(this.input.valueAsNumber);
      if (this.control.continuous) this._queueValue(false);
    });
    this._listen(this.input, 'change', () => {
      this.editing = false;
      if (!this.control.enabled || (this.pending && !this.control.continuous)) return;
      this._queueValue(true);
      this.update();
    });
    this._listen(this.input, 'blur', () => {
      this.editing = false;
      this.update();
    });
    this._bindBusyGuard(() => this.control.continuous);
    this.update();
  }

  update(props = {}) {
    const next = { ...this.control, ...props };
    if (typeof next.continuous !== 'boolean') throw new TypeError('continuous must be a boolean');
    if (!Number.isFinite(next.update_hz) || next.update_hz <= 0) {
      throw new RangeError('update_hz must be positive');
    }
    const control = Object.assign(this.control, next);
    if (!control.enabled) this.cancelPending();
    this._updateInput(control.continuous);
    this.label.textContent = control.label;
    this.input.min = control.min;
    this.input.max = control.max;
    this.input.step = control.step;
    this.min.textContent = this._format(control.min);
    this.max.textContent = this._format(control.max);
    this._flushQueued();
    if (!this.editing && !this.pending && this._queuedValue === undefined) {
      this.input.value = control.value;
      this._display(this.input.valueAsNumber);
    }
  }

  _queueValue(final) {
    // Keep only the latest position while throttled or waiting for Python.
    this._queuedValue = this.input.valueAsNumber;
    this._flushImmediately = final;
    this._flushQueued();
  }

  _flushQueued() {
    clearTimeout(this._timer);
    if (this._queuedValue === undefined || this.pending) return;
    if (this._queuedValue === this.control.value) {
      this._queuedValue = undefined;
      return;
    }
    const delay = 1000 / this.control.update_hz - (performance.now() - this._lastEmitAt);
    if (!this._flushImmediately && delay > 0) {
      this._timer = setTimeout(() => this._flushQueued(), Math.min(delay, 2147483647));
      return;
    }
    const value = this._queuedValue;
    this._queuedValue = undefined;
    this._flushImmediately = false;
    this._lastEmitAt = performance.now();
    this.control.value = value;
    this._onChange(value);
  }

  cancelPending() {
    clearTimeout(this._timer);
    this._queuedValue = undefined;
    this._flushImmediately = false;
    this.editing = false;
  }

  destroy() {
    this.cancelPending();
    super.destroy();
  }

  _format(value) {
    return `${Number(value.toPrecision(8))}${this.control.unit ? ` ${this.control.unit}` : ''}`;
  }

  _display(value) {
    this.output.textContent = this._format(value);
    const percent = 100 * (value - this.control.min) / (this.control.max - this.control.min);
    this.input.style.setProperty('--fill', `${percent}%`);
    this.input.setAttribute('aria-valuetext', this.output.textContent);
  }
}

export class Select extends Control {
  constructor({ onChange = () => {}, ...props } = {}) {
    super('select', { options: [], value: '', ...props });
    this.label = document.createElement('label');
    this.input = document.createElement('select');
    this.input.id = `wrs-select-${++nextControlId}`;
    this.label.htmlFor = this.input.id;
    this.element.append(this.label, this.input);
    this._listen(this.input, 'change', () => {
      if (!this.control.enabled || this.pending) return;
      this.control.value = this.input.value;
      onChange(this.control.value);
    });
    this._bindBusyGuard();
    this.update();
  }

  update(props = {}) {
    Object.assign(this.control, props);
    this._updateInput();
    this.label.textContent = this.control.label;
    const signature = JSON.stringify(this.control.options);
    if (signature !== this.signature) {
      this.signature = signature;
      this.input.replaceChildren(...this.control.options.map(value => {
        const option = document.createElement('option');
        option.value = option.textContent = value;
        return option;
      }));
    }
    if (!this.pending) this.input.value = this.control.value;
  }
}

export class Checkbox extends Control {
  constructor({ onChange = () => {}, ...props } = {}) {
    super('checkbox', { value: false, ...props });
    this.input = document.createElement('input');
    this.input.type = 'checkbox';
    this.input.id = `wrs-checkbox-${++nextControlId}`;
    this.label = document.createElement('label');
    this.label.htmlFor = this.input.id;
    this.element.append(this.input, this.label);
    this._listen(this.input, 'click', (event) => {
      // Label clicks also activate the input while an earlier change is pending.
      if (!this.control.enabled || this.pending) event.preventDefault();
    });
    this._listen(this.input, 'change', () => {
      if (!this.control.enabled || this.pending) return;
      this.control.value = this.input.checked;
      onChange(this.control.value);
    });
    this._bindBusyGuard();
    this.update();
  }

  update(props = {}) {
    Object.assign(this.control, props);
    this._updateInput();
    this.label.textContent = this.control.label;
    if (!this.pending) this.input.checked = this.control.value;
  }
}

export class Text extends Control {
  constructor(props = {}) {
    super('label', { value: '', ...props });
    this.label = document.createElement('span');
    this.label.className = 'ui-label-name';
    this.output = document.createElement('span');
    this.output.className = 'ui-label-value';
    this.element.append(this.label, this.output);
    this.update();
  }

  update(props = {}) {
    Object.assign(this.control, props);
    this.label.textContent = this.control.label;
    this.output.textContent = this.control.value;
  }
}
