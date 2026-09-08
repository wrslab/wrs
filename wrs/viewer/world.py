"""The world a script builds its scene in, drawn by the browser page.

The page is the host: it owns the renderer and the camera and outlives the
script.  A script builds its scene, publishes it to the hub, and exits; the
page keeps the viewpoint you dragged to and picks up whatever the next run
publishes.  Nothing here touches the GPU -- no adapter, no device, no
pipelines -- so a rerun costs the import and the scene build, nothing more.

    base = wvw.World(cam_pos=(.3, .3, .3))
    ...
    base.run()

``run()`` starts the hub if nothing is listening yet, so there is no separate
server to remember.  To keep one running by hand instead::

    py -3.12 -m wrs.viewer.server
"""
import asyncio
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import websockets

import wrs.scene.scene as wss
import wrs.viewer.key as wvk
import wrs.viewer.protocol as wvp
from wrs.utils.scheduler import Scheduler

_HUB_BOOT_TIMEOUT = 10.0


class World:
    """Same surface as the old native World, minus the window."""

    def __init__(self,
                 cam_pos=(.1, .1, .1),
                 cam_lookat_pos=(0, 0, 0),
                 toggle_auto_cam_orbit=False,
                 host='127.0.0.1', port=wvp.DEFAULT_PORT, hz=30,
                 tick_hz=60,
                 auto_start_hub=True):
        self.scene = wss.Scene()
        self.cam_pos = tuple(float(v) for v in cam_pos)
        self.cam_lookat_pos = tuple(float(v) for v in cam_lookat_pos)
        self.toggle_auto_cam_orbit = toggle_auto_cam_orbit
        self.caption = None
        self._scheduler = Scheduler()
        self._handlers = {}
        self._tick_dt = 1.0 / float(tick_hz)
        self._closed = False
        self._host = host
        self._port = int(port)
        self._hz = float(hz)
        self._auto_start_hub = auto_start_hub
        self._publishing = False
        # Page events land here from the publisher thread and are dispatched
        # on the main one, in run() -- the thread a native on_key_press would
        # have arrived on.
        self._events = queue.SimpleQueue()
        self.pressed_keys = set()
        self._pending_presses = set()

    # ------------------------------------------------------------- public API

    def set_scene(self, scene):
        self.scene = scene

    def set_caption(self, caption):
        """Set the browser tab title; safe to call from a callback."""
        self.caption = caption

    def schedule_interval(self, function, interval=.01, *args, **kwargs):
        self._scheduler.schedule_interval(function, interval, *args, **kwargs)

    def schedule_once(self, function, delay=.01, *args, **kwargs):
        self._scheduler.schedule_once(function, delay, *args, **kwargs)

    def stop(self, function):
        self._scheduler.unschedule(function)

    def stop_after(self, function, delay):
        def _stop_cb(dt):
            self.stop(function)

        self._scheduler.schedule_once(_stop_cb, delay)

    def event(self, function):
        """``@base.event`` -- register a handler by its function name."""
        self._handlers[function.__name__] = function
        return function

    def dispatch(self, name, *args):
        handler = self._handlers.get(name)
        if handler is not None:
            handler(*args)

    def is_key_pressed(self, symbol):
        """True for as long as the key is held."""
        return symbol in self.pressed_keys

    def is_key_pressed_edge(self, symbol):
        """True once per press -- the caller consumes the edge.

        The page drops auto-repeat, so holding a key yields exactly one edge
        rather than a stream of them.
        """
        if symbol in self._pending_presses:
            self._pending_presses.discard(symbol)
            return True
        return False

    def close(self):
        self._closed = True

    def run(self):
        """Publish the scene and tick scheduled callbacks.

        Returns on Ctrl-C, on close(), or when another script takes the page
        over -- there is no viewer left to run for at that point.  The
        publisher thread samples ``self.scene`` on its own clock, so this loop
        only advances the script's own callbacks and drains page events.
        """
        if not self._publishing:
            self._ensure_hub()
            threading.Thread(target=self._publish_forever, daemon=True).start()
            self._publishing = True
        try:
            while not self._closed:
                self._drain_events()
                self._scheduler.tick()
                self.dispatch('on_draw')
                time.sleep(self._tick_dt)
        except KeyboardInterrupt:
            pass

    # ---------------------------------------------------------------- events

    def _post_event(self, name, args):
        """Queue an event from the publisher thread; run() drains it."""
        self._events.put((name, args))

    def _drain_events(self):
        while True:
            try:
                name, args = self._events.get_nowait()
            except queue.Empty:
                return
            if name == 'on_key_press':
                self.pressed_keys.add(args[0])
                self._pending_presses.add(args[0])
            elif name == 'on_key_release':
                self.pressed_keys.discard(args[0])
            self.dispatch(name, *args)

    # ------------------------------------------------------------------- hub

    def _hub_is_up(self):
        with socket.socket() as probe:
            probe.settimeout(0.25)
            return probe.connect_ex((self._host, self._port)) == 0

    def _ensure_hub(self):
        """Start the hub if the port is idle, and wait until it answers.

        Detached on purpose: the hub has to outlive this script, which is what
        lets the page survive a rerun.
        """
        if self._hub_is_up() or not self._auto_start_hub:
            return
        flags = {}
        if sys.platform == 'win32':
            flags['creationflags'] = (subprocess.DETACHED_PROCESS
                                      | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            flags['start_new_session'] = True
        argv = [sys.executable, '-m', 'wrs.viewer.server',
                '--host', self._host, '--port', str(self._port)]
        # Opening a window is the right default -- the hub only starts when
        # nothing was listening, so no page can be showing yet.  The escape
        # hatch is for headless boxes and CI.
        if not os.environ.get('WRS_VIEWER_NO_BROWSER'):
            argv.append('--open')
        subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **flags)
        deadline = time.time() + _HUB_BOOT_TIMEOUT
        while time.time() < deadline:
            if self._hub_is_up():
                print(f'viewer hub started on http://{self._host}:{self._port}/')
                return
            time.sleep(0.2)
        raise RuntimeError(
            f'viewer hub did not come up on {self._host}:{self._port}; '
            f'start it by hand with: {sys.executable} -m wrs.viewer.server')

    # ------------------------------------------------------------- publisher

    def _publish_forever(self):
        asyncio.run(self._publish_loop())

    async def _publish_loop(self):
        url = f'ws://{self._host}:{self._port}/publish'
        complained = None
        while not self._closed:
            try:
                async with websockets.connect(url, max_size=None) as ws:
                    complained = None
                    await self._publish(ws)
            except websockets.ConnectionClosed as exc:
                if exc.rcvd is not None and exc.rcvd.code == wvp.SUPERSEDED:
                    # run() is the viewer loop, and this script no longer has
                    # a viewer.  Ending it beats leaving a process burning CPU
                    # on a scene nobody can see and nobody remembers starting.
                    print('viewer: another script took over the page; '
                          'stopping this one', file=sys.stderr, flush=True)
                    self._closed = True
                    return
                complained = self._complain(url, exc, complained)
                await asyncio.sleep(0.5)
            except (OSError, websockets.WebSocketException) as exc:
                complained = self._complain(url, exc, complained)
                await asyncio.sleep(0.5)   # hub restarting, or not up yet

    @staticmethod
    def _complain(url, exc, last):
        """Report a publish failure once, and again only if it changes.

        Retrying silently would leave the page showing a stale scene with
        nothing anywhere to explain it."""
        reason = f'{type(exc).__name__}: {exc}'
        if reason != last:
            print(f'viewer: publish to {url} failed -- {reason}',
                  file=sys.stderr, flush=True)
        return reason

    async def _publish(self, ws):
        # Which geometries this connection already holds.  Reset per connect:
        # a reconnect means the far end starts empty again.
        sent_geoms = set()
        snapshot = list(wvp.iter_scene_models(self.scene))
        models, geometries = wvp.describe(
            [(model_id, model) for model_id, model, _ in snapshot], sent_geoms)
        await ws.send(wvp.scene_message(
            'scene_init', models, geometries,
            # The page keeps whatever camera the user last dragged to and only
            # takes these on its first connect, so a rerun does not yank the
            # viewpoint back to the default.
            camera={
                'pos': list(self.cam_pos),
                'look_at': list(self.cam_lookat_pos),
                'auto_orbit': bool(self.toggle_auto_cam_orbit),
            }))
        live = {entry['id'] for entry in models}
        # Scene out, events in -- concurrently, so a held key does not wait on
        # the next frame and a slow frame does not swallow a keystroke.
        await asyncio.gather(self._send_scene(ws, live, sent_geoms),
                             self._recv_events(ws))

    async def _send_scene(self, ws, live, sent_geoms):
        interval = 1.0 / self._hz
        sent_caption = None
        prev_ids, prev_mats = None, None
        while not self._closed:
            started = time.monotonic()
            # One walk per frame, shared by the delta and the poses.  The
            # scene belongs to the main thread and can change under us, so
            # sampling it twice would announce one set of objects and send
            # matrices for another -- the newcomers would then sit at identity
            # (piled on the origin) until a frame happened to line up.
            snapshot = list(wvp.iter_scene_models(self.scene))
            current = {model_id: model for model_id, model, _ in snapshot}
            added = [mid for mid in current if mid not in live]
            removed = [mid for mid in live if mid not in current]
            if added or removed:
                models, geometries = wvp.describe(
                    [(mid, current[mid]) for mid in added], sent_geoms)
                await ws.send(wvp.scene_message(
                    'scene_delta', models, geometries, remove=removed))
                live = set(current)
                prev_ids = None            # the id list moved; resend it all
            if self.caption != sent_caption:
                sent_caption = self.caption
                await ws.send(json.dumps(
                    {'type': 'caption', 'text': sent_caption}))

            # Send only what moved.  Most of a scene is furniture -- an IK
            # sweep leaves 74k poses untouched forever -- and re-sending every
            # matrix at 30 Hz costs more than the whole rest of the viewer.
            ids, mats = wvp.transform_arrays(snapshot)
            if prev_ids == ids and prev_mats is not None:
                moved = np.flatnonzero(np.any(mats != prev_mats, axis=1))
                if moved.size:
                    await ws.send(wvp.transform_message(
                        [ids[i] for i in moved], mats[moved]))
            else:
                await ws.send(wvp.transform_message(ids, mats))
            prev_ids, prev_mats = ids, mats

            # Pace by what the frame actually cost.  A scene big enough that
            # one pass takes longer than the interval would otherwise spin
            # flat out, starving the script's own callbacks of the GIL and
            # leaving no room to answer the hub's keepalive pings.
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(interval, elapsed))

    async def _recv_events(self, ws):
        """Page -> script.  Key names arrive as JS KeyboardEvent strings, which
        is exactly what key.symbol_from_name expects, so the browser ships the
        raw name and the mapping stays in one place."""
        async for message in ws:
            try:
                payload = json.loads(message)
            except ValueError:
                continue
            if payload.get('type') != 'event':
                continue
            name = payload.get('name')
            if name not in ('on_key_press', 'on_key_release'):
                continue
            symbol = wvk.symbol_from_name(payload.get('key', ''))
            if symbol is not None:
                # modifiers is 0, as the native input manager also left it
                self._post_event(name, (symbol, 0))
