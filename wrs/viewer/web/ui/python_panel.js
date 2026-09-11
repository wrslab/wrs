/** Bind Python-owned UI state to native panel components. */
import { Button, Slider, Select, Checkbox, Text } from './controls.js';
import { Panel } from './panel.js';

export class UIPanel extends Panel {
  constructor(send, options = {}) {
    super({ title: 'Scene controls', anchor: 'top-right', ...options });
    this.send = send;
    this.online = false;
    this.state = null;
    this.rows = new Map();
    this.pending = new Map();
    this.signature = '';
    this.element.dataset.pythonUi = '';
    this._contentVisible = false;
    this._applyVisibility();
    this.error = document.createElement('p');
    this.error.className = 'ui-error';
    this.error.setAttribute('role', 'alert');
    this.error.hidden = true;
    this.body.appendChild(this.error);
    this.footer.hidden = false;
    this.footer.innerHTML = `
      <span class="ui-connection"><i aria-hidden="true"></i><span></span></span>
      <span class="ui-feedback" role="status" aria-live="polite"></span>`;
    this.feedback = this.element.querySelector('.ui-feedback');
    this.connection = this.element.querySelector('.ui-connection span');
    this.setConnected(false);
  }

  reset() {
    if (this._destroyed) return;
    this._clearPending();
    this.state = null;
    this.signature = '';
    for (const row of this.rows.values()) row.destroy();
    this.rows.clear();
    this.controls.replaceChildren();
    this._contentVisible = false;
    this._position = null;
    this.show();
    this._showError('');
    this.feedback.textContent = '';
  }

  setConnected(online) {
    if (this._destroyed) return;
    this.online = Boolean(online);
    this.element.dataset.online = String(this.online);
    this.connection.textContent = this.online ? 'Connected' : 'Offline';
    if (!online) {
      const interrupted = this.pending.size > 0;
      this._clearPending();
      for (const row of this.rows.values()) row.editing = false;
      this.feedback.textContent = 'Waiting for a script';
      if (interrupted) this._showError('Connection lost. The last action may not have completed.');
    } else if (!this.pending.size) {
      this.feedback.textContent = 'Ready';
    }
    this._updateRows();
  }

  apply(state) {
    if (this._destroyed) return;
    if (state.session !== this.state?.session) this.reset();
    if (this.state && state.revision < this.state.revision) return;
    const previous = this.state;
    this.state = state;
    this.element.dataset.panelId = state.id;
    this._contentVisible = state.controls.length > 0;
    const visibilityChanged = !previous || state.visibility_revision !== previous.visibility_revision
      || state.visible !== previous.visible;
    this.update({ title: state.title, description: state.description, fontSize: state.font_size,
      closable: state.closable ?? false, movable: state.movable ?? false,
      ...(visibilityChanged ? { visible: state.visible !== false } : {}) });
    if (state.layout) this.setLayout(state.layout);
    // Only structural changes replace nodes. Value updates preserve focus and drag.
    const signature = JSON.stringify(state.controls.map(c => [c.id, c.kind, c.group]));
    if (signature !== this.signature) {
      this.signature = signature;
      const previousRows = this.rows;
      this.rows = new Map();
      const fragment = document.createDocumentFragment();
      let lastGroup = null;
      for (const control of state.controls) {
        if (control.group && control.group !== lastGroup) {
          const heading = document.createElement('h3');
          heading.className = 'ui-group';
          heading.textContent = control.group;
          fragment.appendChild(heading);
        }
        lastGroup = control.group;
        const previous = previousRows.get(control.id);
        const row = previous?.control.kind === control.kind ? previous : this._createRow(control);
        this.rows.set(control.id, row);
        fragment.appendChild(row.element);
      }
      this.controls.replaceChildren(fragment);
      for (const [id, row] of previousRows) {
        if (this.rows.get(id) !== row) row.destroy();
      }
      for (const [event_id, { row, timer }] of this.pending) {
        if (this.rows.get(row.control.id) !== row) {
          clearTimeout(timer);
          this.pending.delete(event_id);
        }
      }
      if (!this.pending.size && this.feedback.textContent === 'Applying…') {
        this.feedback.textContent = 'Ready';
      }
    }
    this._updateRows();
  }

  receiveResult(result) {
    if (this._destroyed) return;
    if (result.session !== this.state?.session) return;
    const pending = this.pending.get(result.event_id);
    if (!pending) return;  // Another browser's reply, or an expired request.
    clearTimeout(pending.timer);
    this.pending.delete(result.event_id);
    pending.row.pending = false;
    if (result.state) this.apply(result.state);
    this._updateRows();
    this._showError(result.ok ? '' : (result.error || 'The action failed.'));
    this.feedback.textContent = this.pending.size ? 'Applying…' : (result.ok ? 'Applied' : 'Action failed');
  }

  _createRow(control) {
    if (control.kind === 'button') {
      return new Button({ ...control, onClick: () => this._commit(control.id, null) });
    }
    if (control.kind === 'slider') {
      return new Slider({ ...control, onChange: value => this._commit(control.id, value) });
    }
    if (control.kind === 'select') {
      return new Select({ ...control, onChange: value => this._commit(control.id, value) });
    }
    if (control.kind === 'checkbox') {
      return new Checkbox({ ...control, onChange: value => this._commit(control.id, value) });
    }
    return new Text(control);
  }

  _updateRows() {
    for (const control of this.state?.controls || []) {
      this.rows.get(control.id)?.update({
        ...control, enabled: this.online && control.enabled,
      });
    }
  }

  destroy() {
    this.reset();
    super.destroy();
  }

  _commit(id, value) {
    const row = this.rows.get(id);
    if (!this.online || !row || row.pending || !row.control.enabled) return;
    const event_id = crypto.randomUUID();
    const message = { type: 'ui_event', session: this.state.session,
      panel_id: this.state.id, event_id, id, value };
    if (!this.send(message)) {
      this.setConnected(false);
      return;
    }
    row.pending = true;
    const timer = setTimeout(() => {
      this.pending.delete(event_id);
      row.pending = false;
      this._updateRows();
      this._showError('No response yet. Check your script before trying again.');
      this.feedback.textContent = 'Response delayed';
    }, 10000);
    this.pending.set(event_id, { row, timer });
    this._showError('');
    this.feedback.textContent = 'Applying…';
    this._updateRows();
  }

  _clearPending() {
    for (const { row, timer } of this.pending.values()) {
      clearTimeout(timer);
      row.pending = false;
    }
    this.pending.clear();
  }

  _showError(message) {
    this.error.textContent = message;
    this.error.hidden = !message;
  }
}

/** Reconcile Python panels while preserving each panel's focus and pending actions. */
export class UIManager {
  constructor(send, options = {}) {
    this.send = send;
    this.options = options;
    this.panels = new Map();
    this.online = false;
    this.state = null;
  }

  apply(state) {
    if (state.session !== this.state?.session) this.reset();
    if (this.state && state.revision < this.state.revision) return;
    this.state = state;
    const states = state.panels;
    const ids = new Set(states.map(panel => panel.id));
    for (const [id, panel] of this.panels) {
      if (!ids.has(id)) {
        panel.destroy();
        this.panels.delete(id);
      }
    }
    for (const panelState of states) {
      const id = panelState.id;
      let panel = this.panels.get(id);
      if (!panel) {
        panel = new UIPanel(this.send, this.options);
        panel.setConnected(this.online);
        this.panels.set(id, panel);
      }
      panel.apply(panelState);
    }
  }

  receiveResult(result) {
    this.panels.get(result.panel_id)?.receiveResult(result);
  }

  setConnected(online) {
    this.online = Boolean(online);
    for (const panel of this.panels.values()) panel.setConnected(this.online);
  }

  reset() {
    for (const panel of this.panels.values()) panel.destroy();
    this.panels.clear();
    this.state = null;
  }
}
