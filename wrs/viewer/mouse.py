"""Mouse button constants.

rendercanvas numbers its buttons 1/2/3 = left/right/middle; ``BUTTON_MAP``
translates those onto the names below.
"""
LEFT = 1
MIDDLE = 2
RIGHT = 4
MOUSE4 = 8
MOUSE5 = 16

BUTTON_MAP = {1: LEFT, 2: RIGHT, 3: MIDDLE, 4: MOUSE4, 5: MOUSE5}
