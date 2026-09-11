# Viewer controls

Define buttons, sliders, dropdowns, checkboxes and read-only text from Python. Native HTML
and CSS draw the panels above the WebGPU canvas, without a frontend framework,
external fonts or a build step.

## Example

Run from the repository root with WRS installed:

```sh
python -m examples.viewer_ui
```

The example opens three panels: six UR3 joint sliders at top left, a node
selector with X/Y/Z controls at top right, and Workspace status at bottom left.
The coordinate axes follow the selected node; toggle **Show coordinate axes**
to hide or show them. Drag a panel heading to move it;
close the joint/node panels with × and reopen them with **Show controls**.
Joint and position sliders update the scene while dragging.

## Python API

```python
from wrs import wssop, wvw
import wrs.viewer.web_ui as wvui

base = wvw.World(cam_pos=(0.8, 0.8, 0.5))
block = wssop.box()
block.add_to_scene(base.scene)
panel = base.ui.add_panel('position', title='Position',
                          anchor=wvui.Anchor.TOP_RIGHT, width=260, font_size=12,
                          closable=True, movable=True)


def move_x(value):
    block.pos = [value, 0, 0]
    panel.set_value('status', f'X = {value:.2f} m')


panel.add_slider('x', label='X position', unit='m',
                  min_value=-0.25, max_value=0.25, step=0.01,
                  value=0, on_change=move_x)
panel.add_label('status', label='Last action', value='Ready')
base.run()
```

`base.ui.add_*()` addresses the default panel. `add_panel(id, **options)` returns
an independent panel; `remove_panel(id)` removes a named panel. IDs are nonempty
strings of up to 128 characters, unique within their panel. Empty panels stay hidden.

| Method | Behavior |
| --- | --- |
| `add_button(id, label=..., on_click=...)` | Callback with no arguments |
| `add_slider(id, ..., on_change=...)` | Callback with a float when the value is committed |
| `add_select(id, options=[...], value=..., on_change=...)` | Callback with the selected string |
| `add_checkbox(id, value=False, on_change=...)` | Callback with a bool when toggled |
| `add_label(id, label=..., value=...)` | Read-only text |
| `set_value(id, value)` | Update a value without invoking its callback |
| `set_enabled(id, enabled)` | Enable or disable interaction |
| `remove(id)` | Remove a control |

Controls accept `group` to start a visual section. Declare controls in the same
group together. Slider units are display text; values are not converted.
Checkbox values must be Python booleans (`True` or `False`). For example:

```python
panel.add_checkbox('axes', label='Show coordinate axes', value=True,
                   on_change=lambda checked: print('Show axes:', checked))
panel.set_value('axes', False)  # Uncheck without invoking the callback.
```

## Slider updates

Sliders default to `continuous=False`: dragging previews the value in the browser,
and releasing commits it to Python. Set `continuous=True` to run the same
`on_change` callback during dragging too:

```python
panel.add_slider('live_x', label='X position', unit='m',
                  min_value=-0.25, max_value=0.25, step=0.01,
                  value=0, on_change=move_x, continuous=True, update_hz=30)
```

`update_hz` defaults to 30 and must be a positive finite number. It caps drag
updates per slider; it is not a guaranteed callback rate. Each slider waits for
its previous reply and keeps only the latest queued position. Release sends the
final value without waiting for the rate limit, or as soon as the previous reply
arrives. An already-sent, unchanged value is not sent again. Errors, timeouts,
disconnects and control removal discard queued input.

Browser-only `Slider` widgets accept the same `continuous` and `update_hz`
options with their `onChange` callback.

| Layer | Default behavior |
| --- | --- |
| Browser slider | Native input events update the local preview; continuous sends are capped by `update_hz` |
| Python event loop | `World(tick_hz=60)` checks queued events about every 16.7 ms |
| Scene and UI publishing | `World(hz=30)` checks for updates about every 33.3 ms and sends changed state; UI replies are also drained on this cycle |
| Browser rendering | Uses `requestAnimationFrame`, independently of the network rates |

The hub forwards messages as they arrive. Callback cost, network latency and
publishing time can reduce the effective rate. `World(hz=60, tick_hz=60)` raises
the publishing rate as well; a slider setting alone does not change it. These
settings control polling intervals, not real-time deadlines.

## Panel layout

Pass these options to `add_panel()` or change them with `panel.configure(...)`:

| Option | Default | Meaning |
| --- | --- | --- |
| `title`, `description` | `'Scene controls'`, `''` | Heading and optional description |
| `anchor` | `wvui.Anchor.TOP_RIGHT` | Also `TOP_LEFT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT` |
| `offset` | `24` | Distance from the anchored edges, in CSS pixels |
| `width`, `height` | `296`, `None` | CSS pixels; `None` gives automatic height |
| `font_size` | `13` | Base font size in CSS pixels; headings scale relative to it |
| `closable` | `False` | Show a close button |
| `movable` | `False` | Allow heading dragging within the viewport or container |
| `visible` | `True` | Show or hide the panel |

Text size is configurable; it does not shrink automatically with panel width.
Content scrolls when it exceeds the available height. The example uses 12px
text, 260px widths and 16px offsets. Mouse resizing is not implemented.

Closing and dragging affect only the current browser. Ordinary value updates
preserve those local choices. `panel.show()` and `panel.hide()` control visibility
in all viewers, retaining controls and callbacks. Repeated `show()` calls can
reopen a locally closed panel. Reloading restores the Python configuration.
Changing layout dimensions or anchor resets the dragged position; supplying the
same layout preserves it. Panels do not automatically avoid each other.

## Components and files

```text
wrs/viewer/
  web_ui/
    __init__.py       exports Anchor, UIPanel, UIManager
    constant.py       Anchor constants
    manager.py        default/named panels and event routing
    panel.py          control definitions, state and callbacks
    protocol.py       JSON message contract and value validation
  web/ui/
    index.js          browser exports
    controls.js       Button, Slider, Select, Checkbox, Text
    panel.js          Panel layout, collapse, close and drag
    python_panel.js   Python state/event binding
    styles.css        shared appearance
```

Import Python components from `wrs.viewer.web_ui`. Browser widgets own their DOM,
updates and cleanup; the small shared `Control` base handles listeners and
interaction state. They can also be used independently of Python:

```javascript
import { Panel, Button, Text } from './ui/index.js';

const panel = new Panel({ title: 'Local controls', container: document.body,
  anchor: 'bottom-left', width: 260, fontSize: 12, movable: true });
const status = panel.add(new Text({ label: 'Status', value: 'Ready' }));
panel.add(new Button({ label: 'Run',
  onClick: () => status.update({ value: 'Done' }) }));
```

Load `./ui/styles.css` once. `Panel` defaults to inline layout and can be mounted
later with `mount(element)`. Floating panels in a custom container require that
container to have `position: relative` and a defined size. `setLayout(...)`
changes placement, and `update(...)` changes heading, font or behavior options.
After resizing a custom container, call `setLayout()` to constrain its dragged
panel again. `show()` / `hide()` change visibility; `destroy()` removes a panel
and its owned controls. Individual widgets can use `element.hidden`.

## State and callbacks

Callbacks run on the `World.run()` thread. Sliders send on release, or also while
dragging when `continuous=True`. Keep callbacks short. Python validates values
and publishes the resulting state to all connected viewers.

`web_ui.protocol` defines `ui_state` (a collection of panels), `ui_event`,
`ui_result` and `ui_reset`. Events require a panel ID, session ID, event ID and
control ID. The hub relays messages and caches state for page reloads; the
manager routes events and the panel runs callbacks. These JSON messages are
separate from the binary scene protocol in `viewer.protocol`.

Session IDs reject stale events when a script or panel is replaced. Disconnected
controls are disabled, and actions are not replayed on reconnect. Callback errors
appear in the panel and restore that slider/select/checkbox value; other scene changes
made by the callback are not rolled back.
