# The Workman Robot System (WRS)

A self-contained robotics library — kinematics, scene, mesh geometry, collision,
grasp planning, motion planning, and a viewer.

## Lineage

WRS is the third generation of this codebase:

| Generation | Renderer | Repository |
|---|---|---|
| 1st | Panda3D | [wanweiwei07/wrs](https://github.com/wanweiwei07/wrs) |
| 2nd | Pyglet / OpenGL | [wanweiwei07/one](https://github.com/wanweiwei07/one) |
| **3rd (this one)** | **WGPU** | — |

The rewrite is motivated by OpenGL's inherent limitations in how the CPU and the
GPU interact: state is global and implicit, buffer uploads and draw submission
are hard to separate, and validation happens per call rather than once. WGPU
replaces that with explicit pipelines, command buffers, and up-front validation,
so the renderer builds its state once and the per-frame work is just recording
draws.

## Architecture: one server, two clients

```
   Python client                      Web browser client
   (your script)                      (visualization)
        │                                     │
        │  publishes the scene                │  fetches the page,
        │  as it changes                      │  renders with WebGPU
        ▼                                     ▼
        └────────►  server (port 8000)  ◄─────┘
```

**The server** owns port 8000. It serves the viewer page and relays scenes; it
holds no simulation state of its own, only the latest scene it was told about.

**The Python client** is where you work. Building the scene is ordinary library
code; calling `run()` connects to the server and starts publishing.

```python
from wrs import wvw, wssop, wsso

base = wvw.World(cam_pos=(.3, .3, .3))
wssop.frame().add_to_scene(base.scene)
wsso.SceneObject.from_file("bunny.stl").add_to_scene(base.scene)
base.run()
```

**The web browser client** is the visualization. It connects to the local
service on port 8000, pulls the scene from the server, and draws it.

## It sets itself up

`run()` checks whether a server is already listening. If not, it starts one; if
so, it simply connects and begins sending. When no browser client is open, a web
page is opened for you, connected to the server, and showing the scene.

So the two-client mechanism underneath is **transparent to you**. In practice
there is one thing to know:

> Call `run()`, then look at **http://127.0.0.1:8000/** in your browser.

The rest is handled:

- **Several Python clients at once** — the newest one takes the page over, and
  the one it displaced stops rather than fighting for it.
- **Everything closed** — with no page and no script left, the server shuts
  itself down after a short grace period, so nothing is left running.
- **The page outlives your script** — rerun as often as you like; the page stays
  open and picks up whatever the next run publishes.

You can also start the server yourself, which is useful on a headless machine or
when you want it to stay put:

```
py -3.12 -m wrs.viewer.server
```

## Getting started

```
py -3.12 examples/test_bunny.py
```

More examples live in [`examples/`](examples/). For the library itself, see
[`docs/API_INDEX.md`](docs/API_INDEX.md) — a generated map of every public
function, so in-house utilities are easy to find before reaching for an external
one. The manipulation vocabulary — the atomic skills a task-decomposition
layer calls, with pre/post conditions and failure→repair guidance — is
[`docs/SKILLS.md`](docs/SKILLS.md) (machine form: `docs/skills.json`),
generated from the catalog in [`wrs/skills.py`](wrs/skills.py).

## Interactive viewer controls

Define buttons, sliders, dropdowns, and status text from Python with `base.ui`. Browser
actions run callbacks in the World loop, and Python updates both the scene and
the panels. Use `base.ui.add_panel()` for independently positioned panels with
configurable sizes. The UI uses native HTML/CSS with no frontend dependencies.

```
py -3.12 examples/viewer_ui.py
```

See [Viewer controls](docs/tutorials/viewer_ui.md) for the API, lifecycle, and
design notes.
