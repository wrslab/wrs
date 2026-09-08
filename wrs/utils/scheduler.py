"""Timed callbacks for a viewer loop, ticked once per frame.

Shared by the wgpu ``World`` and the headless web host so a script's
``schedule_interval`` behaves the same whichever one is driving it.
"""
import time


class Scheduler:
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
