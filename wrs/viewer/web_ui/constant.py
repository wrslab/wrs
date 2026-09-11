"""Web UI constants, following wrs.utils.constant's class/member convention."""


class Anchor:
    """Panel placement; values remain strings in the browser protocol."""

    TOP_LEFT = 'top-left'
    TOP_RIGHT = 'top-right'
    BOTTOM_LEFT = 'bottom-left'
    BOTTOM_RIGHT = 'bottom-right'

    ALL = (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT)
