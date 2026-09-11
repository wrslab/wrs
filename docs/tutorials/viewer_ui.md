# Viewer controls

Define buttons, sliders, dropdowns and read-only text from Python. Native HTML
and CSS draw the panels above the WebGPU canvas, without a frontend framework,
external fonts or a build step.

## Example

Run from the repository root with WRS installed:

```sh
python -m examples.viewer_ui
```

The example opens three panels: six UR3 joint sliders at top left, a node
selector with X/Y/Z controls at top right, and Workspace status at bottom left.
The coordinate axes follow the selected node. Drag a panel heading to move it;
close the joint/node panels with × and reopen them with **Show controls**.

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
| `add_label(id, label=..., value=...)` | Read-only text |
| `set_value(id, value)` | Update a value without invoking its callback |
| `set_enabled(id, enabled)` | Enable or disable interaction |
| `remove(id)` | Remove a control |

Controls accept `group` to start a visual section. Declare controls in the same
group together. Slider units are display text; values are not converted.

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
    controls.js       Button, Slider, Select, Text
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

Callbacks run on the `World.run()` thread. Dragging a slider previews locally;
releasing it commits the value. Keep callbacks short. Python validates values
and publishes the resulting state to all connected viewers.

`web_ui.protocol` defines `ui_state` (a collection of panels), `ui_event`,
`ui_result` and `ui_reset`. Events require a panel ID, session ID, event ID and
control ID. The hub relays messages and caches state for page reloads; the
manager routes events and the panel runs callbacks. These JSON messages are
separate from the binary scene protocol in `viewer.protocol`.

Session IDs reject stale events when a script or panel is replaced. Disconnected
controls are disabled, and actions are not replayed on reconnect. Callback errors
appear in the panel and restore that slider/select value; other scene changes
made by the callback are not rolled back.
