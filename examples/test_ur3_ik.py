import time
import builtins
import numpy as np

from wrs import wum, wvw, wssop
import wrs.robots.manipulators.universal_robots.ur3.ur3 as wrmuu3

base = wvw.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                 toggle_auto_cam_orbit=True)
oframe = wssop.frame().add_to_scene(base.scene)
robot = wrmuu3.UR3(rotmat=wum.rotmat_from_euler(0, 0, -wum.pi / 2))
robot.add_to_scene(base.scene)
builtins.robot = robot  # for debug access
builtins.base = base

tgt_rotmat = wum.rotmat_from_euler(wum.pi, 0, 0)
results = []
xs = np.linspace(-0.5, 0.5, 15)
ys = np.linspace(-0.5, 0.5, 15)
zs = np.linspace(-0.5, 0.5, 15)
for x in xs:
    for y in ys:
        for z in zs:
            tgt_pos = (x, y, z)
            wssop.frame(pos=tgt_pos, rotmat=tgt_rotmat).add_to_scene(base.scene)
            tic = time.perf_counter_ns()
            qs_list = robot.ik(tgt_pos, tgt_rotmat)
            toc = time.perf_counter_ns()
            success = bool(qs_list)
            time_ns = toc - tic
            results.append(
                {"pos": tgt_pos, "success": success, "time_ns": time_ns,
                 "n_solutions": 0 if not success else len(qs_list)})
            if success:
                for qs in qs_list:
                    tmp_robot = robot.clone()
                    tmp_robot.fk(qs=qs)
                    tmp_robot.add_to_scene(base.scene)

succ = [r for r in results if r["success"]]
fail = [r for r in results if not r["success"]]
print("total:", len(results))
print("success:", len(succ))
print("fail:", len(fail))
if succ:
    times_ms = [r["time_ns"] / 1e6 for r in succ]
    avg_time = np.mean(times_ms)
    best_time = np.min(times_ms)
    worst_time = np.max(times_ms)
    best_case = succ[np.argmin(times_ms)]
    worst_case = succ[np.argmax(times_ms)]
    print("avg time (ms):", avg_time)
    print("best time (ms):", best_time)
    print("worst time (ms):", worst_time)
    print("avg solutions:", np.mean([r["n_solutions"] for r in succ]))
    print("\n--- BEST CASE ---")
    print("  pos:", best_case["pos"])
    print("  time (ms):", best_time)
    print("  n_solutions:", best_case["n_solutions"])
    print("\n--- WORST CASE ---")
    print("  pos:", worst_case["pos"])
    print("  time (ms):", worst_time)
    print("  n_solutions:", worst_case["n_solutions"])
base.run()
