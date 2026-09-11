/**
 * The page host: owns the WebGPU renderer and the camera, and outlives the
 * scripts whose scenes it draws.
 *
 * Page, shaders and socket all come from the one hub process (wrs/viewer/
 * server.py), which a script starts for you.  The socket stays up across
 * reruns -- it is the hub we are talking to, not the script -- so a rerun
 * arrives as a fresh scene_init rather than as a reconnect.
 */
import { Camera } from './camera.js';
import { InputManager } from './input_manager.js';
import { UIManager } from './ui/index.js';
import { Renderer } from './renderer.js';
import { decode, readGeometry } from './wire.js';

const BACKGROUND = 0xf2f2f0;
const ORBIT_DEG_PER_SEC = 0.5;
const DEG = Math.PI / 180;
const SHADER_URLS = { mesh: './shaders/mesh.wgsl', pcd: './shaders/pcd.wgsl' };
const SOCKET_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/view`;

const canvas = document.getElementById('app');
const statusEl = document.createElement('div');
statusEl.style.cssText =
  'position:fixed;left:12px;bottom:12px;font:13px ui-monospace,monospace;' +
  'color:#555;background:rgba(255,255,255,.85);padding:6px 10px;' +
  'border-radius:6px;pointer-events:none;max-width:60ch;';
document.body.appendChild(statusEl);
function setStatus(text) {
  statusEl.textContent = text || '';
  statusEl.style.display = text ? 'block' : 'none';
}

const camera = new Camera({ pos: [2, 2, 2], lookAt: [0, 0, 0] });

// Set on every (re)connect, so keys reach whichever script is live now and
// are quietly dropped while none is.
let socket = null;
const ui = new UIManager((message) => {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(message));
  return true;
}, { container: document.body, anchor: 'top-right' });
new InputManager(camera, canvas, {
  onEvent: (name, key) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'event', name, key }));
    }
  },
});

// Only the first script of the session aims the camera.  After that the
// viewpoint is yours, and a restart must not snatch it back.
let autoOrbit = false;
let haveStoredView = false;

// The viewpoint survives a page reload.  The page is the host and outlives
// every script, so having F5 throw the view away is the one place that
// promise visibly breaks.
const VIEW_KEY = 'wrs.viewer.view';

function saveView() {
  try {
    window.localStorage.setItem(VIEW_KEY, JSON.stringify({
      pos: camera.pos, lookAt: camera.lookAt, up: camera.up, autoOrbit,
    }));
  } catch (err) { /* private window, or storage disabled */ }
}

function restoreView() {
  let stored = null;
  try {
    stored = JSON.parse(window.localStorage.getItem(VIEW_KEY) || 'null');
  } catch (err) { return; }
  if (!stored || !stored.pos) return;
  camera.pos = stored.pos;
  camera.lookAt = stored.lookAt;
  if (stored.up) camera.up = stored.up;
  autoOrbit = Boolean(stored.autoOrbit);
  haveStoredView = true;
}

/**
 * A script that publishes -- started, rerun, or taking over -- gets the
 * camera it asked for.  The hub's catch-up for a page that just (re)connected
 * is flagged as a replay and leaves the stored viewpoint alone.
 */
function applyCamera(cam, replayed) {
  if (!cam || (replayed && haveStoredView)) return;
  camera.setTo(cam.pos, cam.look_at);
  autoOrbit = Boolean(cam.auto_orbit);
  haveStoredView = true;
  saveView();
}

function connect(renderer) {
  const ws = new WebSocket(SOCKET_URL);
  socket = ws;
  ws.binaryType = 'arraybuffer';   // scene frames are binary; see wire.js
  ws.onopen = () => setStatus(null);
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      const payload = JSON.parse(event.data);
      if (payload.type === 'status') {
        ui.setConnected(payload.publisher);
        // the hub is still here; it is the script that comes and goes
        setStatus(payload.publisher ? null : 'no script publishing');
      } else if (payload.type === 'ui_state') {
        ui.apply(payload);
      } else if (payload.type === 'ui_result') {
        ui.receiveResult(payload);
      } else if (payload.type === 'ui_reset') {
        ui.reset();
      } else if (payload.type === 'caption') {
        // a script's set_caption takes the tab over; falling back to
        // the product name when it clears the caption
        document.title = payload.text || 'The Workman Robot System (WRS)';
      }
      return;
    }
    const { header, view } = decode(event.data);
    if (header.type === 'scene_init') {
      // Model ids are per-publisher serials, so a rerun sends a whole new
      // set; without clearing, every rerun piles another copy into the scene
      // and leaks its buffers.
      renderer.clear();
      applyCamera(header.camera, header.replay);
      // geometry first: a model is only a pose and a colour over one
      header.geometries.forEach((g) => renderer.addGeometry(readGeometry(g, view)));
      header.models.forEach((e) => renderer.add(e));
    } else if (header.type === 'scene_delta') {
      // objects added to or removed from the scene after run()
      (header.remove || []).forEach((id) => renderer.remove(id));
      header.geometries.forEach((g) => renderer.addGeometry(readGeometry(g, view)));
      header.models.forEach((e) => renderer.add(e));
    } else if (header.type === 'scene_update') {
      const matrices = view(header.matrices, Float32Array);
      header.ids.forEach((id, i) => renderer.setTransform(
        id, matrices.subarray(i * 16, i * 16 + 16)));
    }
  };
  ws.onerror = () => ws.close();
  ws.onclose = () => {
    ui.setConnected(false);
    setStatus('viewer hub not reachable -- run a script, or ' +
              'py -3.12 -m wrs.viewer.server');
    setTimeout(() => connect(renderer), 500);
  };
}

async function main() {
  setStatus('starting WebGPU ...');
  const shaders = Object.fromEntries(await Promise.all(
    Object.entries(SHADER_URLS).map(async ([name, url]) => {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`${url} -> ${response.status}; serve the page with `
                        + 'tools/serve_web.py, not a bare file:// open');
      }
      return [name, await response.text()];
    })));
  const renderer = await Renderer.create(canvas, shaders, BACKGROUND);

  // Before connecting: the first scene_init must already know whether a
  // stored viewpoint exists, or a replay would overwrite it.
  restoreView();
  window.addEventListener('beforeunload', saveView);

  setStatus('connecting ...');
  connect(renderer);

  // Handles for poking at a live page from the devtools console -- which is
  // the only way in, everything else here is module-scoped.
  window.wrs = { camera, renderer };

  // World(toggle_auto_cam_orbit=True) spins the view slowly about +Z, at the
  // 0.5 deg/s the native viewer used.  A drag still works; it just adds on top.
  let lastSave = 0;
  let last = performance.now();
  const frame = (now) => {
    const dt = (now - last) / 1000;
    last = now;
    if (autoOrbit) camera.orbit([0, 0, 1], ORBIT_DEG_PER_SEC * dt * DEG);
    if (now - lastSave > 1000) { lastSave = now; saveView(); }
    renderer.render(camera);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((err) => {
  setStatus(`cannot start: ${err.message}`);
  console.error(err);
});
