"""UI message contract and pure validation, without sockets, DOM or callbacks.

ui_state: collection session/revision + panels; each panel carries its own
          id/session/revision, heading/layout, closable/movable/visible flags,
          visibility_revision and control definitions/values. Only an explicit
          Python visibility command increments visibility_revision, so value
          updates do not reopen panels that a browser locally closed.
ui_event: type, panel_id, session, event_id, id, value.
ui_result: matching panel/session/event/control IDs, ok, optional error, state.
ui_reset: clear the previous publisher's panels.

These JSON messages are separate from viewer.protocol's binary scene stream.
Panel existence, enabled state and callback execution belong to the manager
and panel. Only shape/value rules live here, shared by definitions and updates.
"""
import math
from numbers import Real


def valid_id(value):
    """Whether a panel/control/event/session ID is a nonempty bounded string."""
    return isinstance(value, str) and 0 < len(value) <= 128


def is_ui_event(payload):
    """Check the event envelope; the addressed panel validates its value."""
    return (isinstance(payload, dict) and payload.get('type') == 'ui_event'
            and all(valid_id(payload.get(key))
                    for key in ('panel_id', 'session', 'event_id', 'id')))


def finite_number(value):
    """Accept real finite numbers, excluding booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('expected a finite number')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError('expected a finite number') from exc
    if not math.isfinite(value):
        raise ValueError('expected a finite number')
    return value


def slider_value(control, value):
    """Validate bounds and snap to the native range input's step lattice."""
    value = finite_number(value)
    low, high, step = control['min'], control['max'], control['step']
    if not low <= value <= high:
        raise ValueError(f'value must be between {low:g} and {high:g}')
    ticks = math.floor((value - low) / step + 0.5)
    # Match the range input's step lattice, including a non-aligned max.
    ticks = min(ticks, math.floor((high - low) / step + 1e-10))
    return min(high, max(low, float(f'{low + ticks * step:.15g}')))


def select_value(control, value):
    """Accept only one of the dropdown's string options."""
    if not isinstance(value, str) or value not in control['options']:
        raise ValueError('value must be one of the select options')
    return value
