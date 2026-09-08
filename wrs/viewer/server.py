"""The hub: one long-lived process, one port, serving the page and relaying
scenes to it.

    browser  --- ws /view ------>  hub  <------ ws /publish ---  your script
             <-- page, shaders --  (http)

Scripts are *clients* here.  That is the whole point: nothing a script does can
take the port, so two scripts running at once cannot collide, and killing one
never takes the page host down with it.  The hub keeps the last scene it was
sent, so a browser that opens later still sees something, and a script that
exits leaves its scene on screen until the next one publishes.

Run it explicitly::

    py -3.12 -m wrs.viewer.server

or just run a script -- ``World`` starts one if the port is idle.
"""
import argparse
import asyncio
import json
import mimetypes
import os
import webbrowser

import websockets
from websockets.http11 import Response
from websockets.datastructures import Headers

import wrs.viewer.protocol as wvp
from wrs.viewer.protocol import DEFAULT_PORT

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

# Seconds with no page and no script before the hub gives up.  Long enough to
# ride out a page reload, which drops the only viewer for about a second.
IDLE_TIMEOUT = 30.0


class Hub:
    """Fan-out from one publisher to any number of viewers."""

    def __init__(self, idle_timeout=IDLE_TIMEOUT):
        self.viewers = set()
        self.publisher = None
        self.idle_timeout = idle_timeout
        # Nothing has connected yet, so being empty is not yet idle: a hub
        # started by hand should wait for its first client, however long that
        # takes.  Only losing the last one starts the clock.
        self.seen_client = False
        self.stop = asyncio.Event()
        # The scene as last published, kept so a browser opening mid-run gets
        # the whole thing rather than only the deltas from here on.
        self.models = {}
        # geometry id -> (meta, {field: bytes}); models only name these, so a
        # scene of clones keeps one copy however many use it
        self.geoms = {}
        self.camera = None
        self.caption = None
        # id -> its latest matrix, accumulated: a page that opens (or
        # reloads) after the script exits would otherwise draw every model at
        # identity, all its links piled on the origin.  Accumulated rather
        # than kept as one frame because the publisher sends only what moved.
        self.transforms = {}

    # ------------------------------------------------------------- publisher

    async def watch_idle(self):
        """Shut down once the page and the script are both gone."""
        if not self.idle_timeout:
            return
        idle_for = 0.0
        while True:
            await asyncio.sleep(1.0)
            if self.viewers or self.publisher is not None or not self.seen_client:
                idle_for = 0.0
                continue
            idle_for += 1.0
            if idle_for >= self.idle_timeout:
                print(f'no page and no script for {self.idle_timeout:.0f}s; '
                      f'stopping', flush=True)
                self.stop.set()
                return

    async def on_publish(self, websocket):
        # A new script takes over; the old one is dropped rather than
        # interleaved, which is what a rerun means.  The close code matters:
        # the displaced script reconnects on its own, so without a signal that
        # it has been replaced the two would kick each other out forever.
        if self.publisher is not None:
            await self.publisher.close(
                wvp.SUPERSEDED, 'another script took over')
        self.publisher = websocket
        self.seen_client = True
        await self._tell_viewers_status(True)
        try:
            async for message in websocket:
                await self._relay(message)
        except websockets.ConnectionClosed:
            pass
        finally:
            if self.publisher is websocket:
                self.publisher = None
                await self._tell_viewers_status(False)

    async def _relay(self, message):
        """Keep enough to re-frame the scene for a later viewer, then pass the
        publisher's own frame through untouched.

        The cache holds each model as the raw bytes it arrived in -- the hub
        never looks at a vertex."""
        if isinstance(message, str):
            payload = json.loads(message)
            if payload.get("type") == "caption":
                self.caption = payload.get("text")
        else:
            header, blob = wvp.unpack(message)
            kind = header.get("type")
            if kind == "scene_init":
                self.models = {e["id"]: e for e in header["models"]}
                self.geoms = {meta["id"]: (meta, fields) for meta, fields
                              in wvp.split_geometries(header, blob)}
                self.camera = header.get("camera")
                self.transforms = {}        # belongs to the publisher that left
            elif kind == "scene_update":
                self.transforms.update(wvp.split_transforms(header, blob))
            elif kind == "scene_delta":
                for model_id in header.get("remove", []):
                    self.models.pop(model_id, None)
                for entry in header["models"]:
                    self.models[entry["id"]] = entry
                for meta, fields in wvp.split_geometries(header, blob):
                    self.geoms[meta["id"]] = (meta, fields)
        await self._broadcast(message)

    # --------------------------------------------------------------- viewers

    async def on_view(self, websocket):
        self.viewers.add(websocket)
        self.seen_client = True
        try:
            models = list(self.models.values())
            used = {entry["geom"] for entry in models}
            await websocket.send(wvp.scene_message(
                "scene_init", models,
                [self.geoms[g] for g in used if g in self.geoms],
                camera=self.camera))
            if self.transforms:
                ids = list(self.transforms)
                await websocket.send(wvp.transform_message(
                    ids, b"".join(self.transforms[i] for i in ids)))
            await websocket.send(json.dumps({
                "type": "status", "publisher": self.publisher is not None}))
            if self.caption is not None:
                await websocket.send(json.dumps(
                    {"type": "caption", "text": self.caption}))
            async for message in websocket:
                # events travel the other way: page -> script
                if self.publisher is not None:
                    await self.publisher.send(message)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.viewers.discard(websocket)

    async def _broadcast(self, message):
        for viewer in list(self.viewers):
            try:
                await viewer.send(message)
            except websockets.ConnectionClosed:
                self.viewers.discard(viewer)

    async def _tell_viewers_status(self, publishing):
        await self._broadcast(json.dumps(
            {"type": "status", "publisher": publishing}))


# ----------------------------------------------------------------- http side

def _static_response(path):
    """Serve web/ for anything that is not a websocket upgrade."""
    if path in ('/', ''):
        path = '/index.html'
    path = path.split('?')[0]
    # keep the request inside web/ no matter what the URL claims
    full = os.path.normpath(os.path.join(WEB_DIR, path.lstrip('/')))
    if not full.startswith(WEB_DIR) or not os.path.isfile(full):
        return Response(404, 'Not Found',
                        Headers({'Content-Length': '0'}), b'')
    with open(full, 'rb') as handle:
        body = handle.read()
    content_type = mimetypes.guess_type(full)[0] or 'application/octet-stream'
    if full.endswith('.wgsl'):
        content_type = 'text/plain; charset=utf-8'
    return Response(200, 'OK', Headers({
        'Content-Type': content_type,
        'Content-Length': str(len(body)),
        # the page is edited while it is being served, so never cache it
        'Cache-Control': 'no-store',
    }), body)


async def serve(host='127.0.0.1', port=DEFAULT_PORT,
                idle_timeout=IDLE_TIMEOUT):
    hub = Hub(idle_timeout)

    async def router(websocket):
        if websocket.request.path.rstrip('/') == '/publish':
            await hub.on_publish(websocket)
        else:
            await hub.on_view(websocket)

    def process_request(connection, request):
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            return None          # let websockets handle the handshake
        return _static_response(request.path)

    # No max_size: a scene_init carries every vertex of every mesh as JSON
    # text, and a modest robot cell clears 80 MB.  A cap here does not protect
    # anything -- the hub is bound to localhost and the sender is the user's
    # own script -- it only rejects the frame and leaves the page showing
    # whatever it had before, with nothing logged anywhere.
    async with websockets.serve(router, host, port, max_size=None,
                                process_request=process_request):
        watcher = asyncio.create_task(hub.watch_idle())
        await hub.stop.wait()
        watcher.cancel()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--open', action='store_true',
                    help='open the page in a browser once the hub is up')
    ap.add_argument('--idle-timeout', type=float, default=IDLE_TIMEOUT,
                    help='seconds with no page and no script before stopping; '
                         '0 to stay up forever')
    args = ap.parse_args()
    url = f'http://{args.host}:{args.port}/'
    idle = (f'stops {args.idle_timeout:.0f}s after the last page and script go'
            if args.idle_timeout else 'stays up until Ctrl-C')
    print(f'wrs viewer hub on {url}\n  serving {WEB_DIR}\n  {idle}',
          flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        asyncio.run(serve(args.host, args.port, args.idle_timeout))
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
