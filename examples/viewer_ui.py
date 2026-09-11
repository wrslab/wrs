"""Three browser panels: UR3 joints, scene nodes and shared status."""
import numpy as np

from wrs import wssop, wvw
import wrs.viewer.web_ui as wvui
import wrs.robots.manipulators.universal_robots.ur3.ur3 as wrmuu3


base = wvw.World(cam_pos=(1.3, 1.5, 1.1), cam_lookat_pos=(0, 0, 0.2))
base.set_caption('WRS / Interactive workspace')
wssop.box(pos=(0, 0, -0.02), xyz_lengths=(1.2, 1.0, 0.02),
          rgb=(0.84, 0.87, 0.83)).add_to_scene(base.scene)

robot = wrmuu3.UR3(pos=(-0.15, 0, 0))
robot.add_to_scene(base.scene)
block = wssop.box(pos=(0, 0, 0.06), xyz_lengths=(0.12, 0.12, 0.12),
                 rgb=(0.22, 0.52, 0.43))
block.add_to_scene(base.scene)
ball = wssop.sphere(pos=(0.22, 0.18, 0.06), radius=0.06,
                   rgb=(0.80, 0.51, 0.29))
ball.add_to_scene(base.scene)

# Keep Python objects in a dictionary; the dropdown only sends their names.
nodes = {'Block': block, 'Ball': ball, 'UR3': robot}
home_positions = {name: node.pos.copy() for name, node in nodes.items()}
selected_node = 'Block'
selection_frame = wssop.frame(pos=block.pos, length_scale=2.2)
selection_frame.add_to_scene(base.scene)

arm_panel = base.ui.add_panel('arm', title='UR3 joints', anchor=wvui.Anchor.TOP_LEFT,
                              width=260, offset=16, font_size=12, closable=True, movable=True)
node_panel = base.ui.add_panel('nodes', title='Scene nodes', anchor=wvui.Anchor.TOP_RIGHT,
                               width=260, offset=16, font_size=12, closable=True, movable=True,
                               description='Choose a node. The coordinate axes follow it.')
status_panel = base.ui.add_panel('status', title='Workspace', anchor=wvui.Anchor.BOTTOM_LEFT,
                                 width=260, offset=16, font_size=12, movable=True)
status_panel.add_label('status', label='Last action', value='Ready to explore')
status_panel.add_label('joints', label='Joint angles · degrees', value='')


def show_controls():
    arm_panel.show()
    node_panel.show()


status_panel.add_button('show', label='Show controls', on_click=show_controls)


def update_position():
    pos = nodes[selected_node].pos
    selection_frame.pos = pos
    for axis, value in zip('xyz', pos):
        node_panel.set_value(axis, float(value))
    node_panel.set_value('position', ' / '.join(f'{v:.2f}' for v in pos) + ' m')


def select_node(name):
    global selected_node
    selected_node = name
    update_position()
    status_panel.set_value('status', f'Selected {name}')


def move_node(axis, value):
    pos = nodes[selected_node].pos.copy()
    pos[axis] = value
    nodes[selected_node].pos = pos
    update_position()
    status_panel.set_value('status', f'Moved {selected_node}')


def reset_position():
    nodes[selected_node].pos = home_positions[selected_node]
    update_position()
    status_panel.set_value('status', 'Home position restored')


node_panel.add_select('node', label='Selected node', options=list(nodes),
                       value=selected_node, on_change=select_node)
for axis, value in enumerate(block.pos):
    node_panel.add_slider(
        'xyz'[axis], label=f'{"XYZ"[axis]} position', unit='m',
        min_value=0 if axis == 2 else -0.5, max_value=0.6 if axis == 2 else 0.5,
        step=0.01, value=float(value),
        on_change=lambda value, axis=axis: move_node(axis, value))
node_panel.add_button('reset', label='Reset position', on_click=reset_position)
node_panel.add_label('position', label='World position · X / Y / Z')

home_qs = robot.qs.copy()


def update_joints():
    angles = np.rad2deg(robot.qs)
    for joint, angle in enumerate(angles):
        arm_panel.set_value(f'joint_{joint + 1}', round(float(angle)))
    status_panel.set_value('joints', ' / '.join(f'{v:.0f}' for v in angles))


def move_joint(joint, angle):
    qs = robot.qs.copy()
    qs[joint] = np.deg2rad(angle)
    robot.fk(qs=qs)
    update_joints()
    status_panel.set_value('status', f'Joint {joint + 1} → {angle:.0f}°')


def reset_joints():
    robot.fk(qs=home_qs)
    update_joints()
    status_panel.set_value('status', 'Home pose restored')


for joint, angle in enumerate(np.rad2deg(home_qs)):
    arm_panel.add_slider(
        f'joint_{joint + 1}', label=f'Joint {joint + 1}', unit='°',
        min_value=-180, max_value=180, step=1, value=round(float(angle)),
        on_change=lambda value, joint=joint: move_joint(joint, value))
arm_panel.add_button('reset', label='Home pose', on_click=reset_joints)
update_position()
update_joints()

if __name__ == '__main__':
    base.run()
