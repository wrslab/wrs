import cProfile
import numpy as np
from wrs import wvw, wssop

profiler = cProfile.Profile()
profiler.enable()

base = wvw.World(cam_pos=np.array([1.5, 0, 1]),
                toggle_auto_cam_orbit=True)
for i in range(2000):
    pos = np.random.rand(3)
    cyl = wssop.frame(pos=pos)
    cyl.add_to_scene(base.scene)
base.run()

profiler.disable()
profiler.dump_stats("result.prof")
print("Profile saved to result.prof")