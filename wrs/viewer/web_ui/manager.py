"""Named panels, a default panel and UI event routing."""
from .panel import UIPanel
from . import protocol


class UIManager(UIPanel):
    """World.ui: a default panel plus named panels with independent controls.

    ui.add_slider()/configure() calls address the default panel.
    Use add_panel() to create another panel and configure() on its returned object.
    """

    def __init__(self):
        super().__init__()
        self._panels = {'default': self}
        self._state_signature = None
        self._state_revision = 0

    def add_panel(self, panel_id, **options):
        """Create a named panel; options are UIPanel heading/layout parameters."""
        if not protocol.valid_id(panel_id):
            raise ValueError('panel_id must be a nonempty string of at most 128 characters')
        with self._lock:
            if panel_id in self._panels:
                raise ValueError(f'duplicate panel_id: {panel_id}')
            panel = UIPanel(**options)
            panel._panel_id = panel_id
            self._panels[panel_id] = panel
            return panel

    def remove_panel(self, panel_id):
        """Remove a named panel. The default panel is permanent."""
        if panel_id == 'default':
            raise ValueError('cannot remove the default panel')
        with self._lock:
            del self._panels[panel_id]

    def _snapshot_all(self, after_revision=None):
        with self._lock:
            states = [panel._snapshot() for panel in self._panels.values()]
            signature = tuple((state['session'], state['revision']) for state in states)
            if signature != self._state_signature:
                self._state_signature = signature
                self._state_revision += 1
            if after_revision == self._state_revision:
                return None
            return dict(type='ui_state', session=self._session,
                        revision=self._state_revision, panels=states)

    def _handle_event(self, payload):
        if not protocol.is_ui_event(payload):
            return None
        panel_id = payload['panel_id']
        with self._lock:
            panel = self._panels.get(panel_id)
        if panel is not None:
            return UIPanel._handle_event(panel, payload)
