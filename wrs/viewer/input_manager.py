"""Keyboard and pointer state, on rendercanvas events.

rendercanvas reports pointer y growing downwards and gives no drag delta,
so y is flipped and the previous position tracked by hand -- the camera
wants a conventional right-handed drag.
"""
import wrs.viewer.key as wvk
import wrs.viewer.mouse as wvm


class InputManager:

    def __init__(self, world, canvas):
        self._world = world
        self._canvas = canvas
        self.pressed_keys = set()
        self._pending_key_presses = set()
        self.pressed_buttons = set()
        self.last_mouse_x = None
        self.last_mouse_y = None
        canvas.add_event_handler(self._on_key_down, 'key_down')
        canvas.add_event_handler(self._on_key_up, 'key_up')
        canvas.add_event_handler(self._on_pointer_down, 'pointer_down')
        canvas.add_event_handler(self._on_pointer_up, 'pointer_up')
        canvas.add_event_handler(self._on_pointer_move, 'pointer_move')
        canvas.add_event_handler(self._on_wheel, 'wheel')

    # ------------------------------------------------------------- public API

    def is_key_pressed(self, symbol):
        return symbol in self.pressed_keys

    def is_key_pressed_edge(self, symbol):
        if symbol in self._pending_key_presses:
            self._pending_key_presses.discard(symbol)
            return True
        return False

    def is_button_pressed(self, button):
        return button in self.pressed_buttons

    # ---------------------------------------------------------------- events

    def _on_key_down(self, event):
        symbol = wvk.symbol_from_name(event['key'])
        if symbol is None:
            return
        self.pressed_keys.add(symbol)
        self._pending_key_presses.add(symbol)
        self._world.dispatch('on_key_press', symbol, 0)

    def _on_key_up(self, event):
        symbol = wvk.symbol_from_name(event['key'])
        if symbol is None:
            return
        self.pressed_keys.discard(symbol)
        self._world.dispatch('on_key_release', symbol, 0)

    def _on_pointer_down(self, event):
        button = wvm.BUTTON_MAP.get(event['button'])
        if button is not None:
            self.pressed_buttons.add(button)
        self.last_mouse_x, self.last_mouse_y = event['x'], event['y']

    def _on_pointer_up(self, event):
        button = wvm.BUTTON_MAP.get(event['button'])
        if button is not None:
            self.pressed_buttons.discard(button)
        self.last_mouse_x = self.last_mouse_y = None

    def _on_pointer_move(self, event):
        x, y = event['x'], event['y']
        if self.last_mouse_x is None:
            self.last_mouse_x, self.last_mouse_y = x, y
            return
        dx = x - self.last_mouse_x
        dy = -(y - self.last_mouse_y)  # events count y downwards
        self.last_mouse_x, self.last_mouse_y = x, y
        camera = self._world.camera
        if wvm.RIGHT in self.pressed_buttons:
            camera.mouse_orbit(dx, dy)
        elif wvm.MIDDLE in self.pressed_buttons:
            camera.mouse_pan(dx, dy)

    def _on_wheel(self, event):
        # scrolling forward gives dy == -100; forward should zoom in
        self._world.camera.mouse_zoom(event['dy'] / 100.0)
