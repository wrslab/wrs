"""The window, the scene it shows, and the loop that drives them.

A World owns a RenderCanvas rather than being one, so a script holds a plain
object: ``base.scene`` to populate, ``base.camera`` to aim, and run /
schedule_interval / schedule_once / schedule_interval_after / stop /
stop_after / set_caption / event to drive it.  Timed callbacks are served by
the small scheduler below, ticked once per frame.
"""
import time
import wgpu
from rendercanvas.auto import RenderCanvas, loop

import wrs.scene.scene as wss
import wrs.viewer.context as wvc
import wrs.viewer.camera as wvcam
import wrs.viewer.input_manager as wvim
import wrs.viewer.render as wvr

SAMPLE_COUNT = 4


class _Scheduler:
    """Timed callbacks, called as ``fn(dt, *args, **kwargs)``.

    A linear scan per frame: measured at 0.14 us for the handful of callbacks
    a script actually schedules, against a 16.7 ms frame.  A sorted heap only
    pays off past ~10 callbacks, which no example comes near.
    """

    def __init__(self):
        self._items = []  # [fn, interval, next_t, last_t, args, kwargs, repeat]

    def schedule_interval(self, fn, interval, *args, **kwargs):
        now = time.perf_counter()
        self._items.append(
            [fn, interval, now + interval, now, args, kwargs, True])

    def schedule_once(self, fn, delay, *args, **kwargs):
        now = time.perf_counter()
        self._items.append([fn, delay, now + delay, now, args, kwargs, False])

    def unschedule(self, fn):
        self._items = [it for it in self._items if it[0] is not fn]

    def tick(self):
        now = time.perf_counter()
        due = [it for it in self._items if now >= it[2]]
        for item in due:
            fn, interval, _, last_t, args, kwargs, repeat = item
            # the time that actually elapsed, so a callback integrating over
            # dt keeps its rate when frames are slow
            fn(now - last_t, *args, **kwargs)
            item[3] = now
            if repeat:
                item[2] = now + interval
            else:
                # by identity: two items can compare equal on ==
                self._items = [it for it in self._items if it is not item]


class World:

    def __init__(self,
                 cam_pos=(.1, .1, .1),
                 cam_lookat_pos=(0, 0, 0),
                 win_size=None,
                 toggle_auto_cam_orbit=False):
        win_w, win_h = win_size if win_size else _default_win_size()
        self.canvas = RenderCanvas(size=(win_w, win_h), title='WRS World',
                                   update_mode='continuous', max_fps=60)
        self.device = wvc.get_device()
        self.context = self.canvas.get_context('wgpu')
        self.color_format = self.context.get_preferred_format(
            wvc.get_adapter())
        self.context.configure(device=self.device, format=self.color_format)
        self.camera = wvcam.Camera(pos=cam_pos, look_at=cam_lookat_pos,
                                    aspect=win_w / win_h)
        self.render = wvr.Render(self.camera, self.color_format, SAMPLE_COUNT)
        self.scene = wss.Scene()
        self.input_manager = wvim.InputManager(self, self.canvas)
        self._scheduler = _Scheduler()
        self._handlers = {}
        self._msaa_tex = None
        self._depth_tex = None
        self._target_size = (0, 0)
        self._closed = False
        if toggle_auto_cam_orbit:
            self.schedule_interval(self.auto_cam_orbit, interval=1 / 30.0)
        self.canvas.add_event_handler(self._on_resize, 'resize')
        self.canvas.add_event_handler(self._on_close, 'close')
        self.canvas.request_draw(self._draw_frame)

    # ------------------------------------------------------------- public API

    def set_scene(self, scene):
        self.scene = scene

    def set_caption(self, caption):
        self.canvas.set_title(caption)

    def auto_cam_orbit(self, dt, deg_per_sec=.5):
        self.camera.orbit(angle_rad=deg_per_sec * dt * (3.14159265 / 180.0))

    def schedule_interval(self, function, interval=.01, *args, **kwargs):
        self._scheduler.schedule_interval(function, interval, *args, **kwargs)

    def schedule_once(self, function, delay=.01, *args, **kwargs):
        self._scheduler.schedule_once(function, delay, *args, **kwargs)

    def schedule_interval_after(self, function, delay, interval=.01,
                                *args, **kwargs):
        def _start_cb(dt):
            self._scheduler.schedule_interval(
                function, interval, *args, **kwargs)

        self._scheduler.schedule_once(_start_cb, delay)

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

    def close(self):
        self.canvas.close()

    def run(self):
        loop.run()

    # ----------------------------------------------------------------- frames

    def _on_close(self, event):
        self._closed = True

    def _on_resize(self, event):
        width = max(1, int(event['width']))
        height = max(1, int(event['height']))
        self.camera._rebuild_projmat(width, height)

    def _ensure_targets(self, width, height):
        if self._target_size == (width, height):
            return
        self._target_size = (width, height)
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT
        self._msaa_tex = self.device.create_texture(
            size=(width, height, 1), format=self.color_format,
            sample_count=SAMPLE_COUNT, usage=usage)
        self._depth_tex = self.device.create_texture(
            size=(width, height, 1), format=wvr.DEPTH_FORMAT,
            sample_count=SAMPLE_COUNT, usage=usage)

    def _draw_frame(self):
        # A draw can still be requested after close(), once the swapchain is
        # already gone.
        if self._closed or self.canvas.get_closed():
            return
        self._scheduler.tick()
        try:
            target = self.context.get_current_texture()
        except RuntimeError:
            # The window went away between the draw request and here, so the
            # swapchain is already unconfigured.
            self._closed = True
            return
        width, height = target.size[0], target.size[1]
        self._ensure_targets(width, height)
        plan = self.render.prepare(self.scene, width, height)
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[{
                'view': self._msaa_tex.create_view(),
                'resolve_target': target.create_view(),
                'clear_value': (1.0, 1.0, 1.0, 1.0),
                'load_op': wgpu.LoadOp.clear,
                'store_op': wgpu.StoreOp.store}],
            depth_stencil_attachment={
                'view': self._depth_tex.create_view(),
                'depth_clear_value': 1.0,
                'depth_load_op': wgpu.LoadOp.clear,
                'depth_store_op': wgpu.StoreOp.store})
        self.render.draw(render_pass, plan)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        self.dispatch('on_draw')


def _default_win_size():
    return 1280, 960
