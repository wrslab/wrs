import os
import builtins
import numpy as np

from wrs import wum, wuc, wvw, wssop, khi_rs007l, or_2fg7


if __name__ == "__main__":
    cand_file = "rs007l_grasp_candidates.npz"
    if not os.path.exists(cand_file):
        raise FileNotFoundError(
            "Candidate file not found. Run test_rs007l_grasp_motion.py first "
            "to generate rs007l_grasp_candidates.npz."
        )

    data = np.load(cand_file)
    pre_pos = data["pre_pos"]
    pre_rot = data["pre_rot"]
    jaw_width = data["jaw_width"]

    base = wvw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, 0.75))
    builtins.base = base
    scene = base.scene
    wssop.frame().add_to_scene(scene)

    robot = khi_rs007l.RS007L()
    robot.add_to_scene(scene)
    builtins.robot = robot

    gripper = or_2fg7.OR2FG7()
    gripper.add_to_scene(scene)
    robot.mount(gripper, robot.runtime_lnks[-1], update=True)

    wssop.frame(pos =robot.tcp('flange').tf[:3, 3],
                rotmat=robot.tcp('flange').tf[:3, :3],
                color_mat=wuc.CoordColor.MYC).add_to_scene(scene)

    n_total = len(pre_pos)
    n_ik = 0
    n_ik_valid = 0

    for i in range(n_total):
        tgt_pos = pre_pos[i]+np.array([0.5, 0.5, 0.3])
        tgt_rot = pre_rot[i]
        jw = float(jaw_width[i])

        _s = robot.ik(tgt_pos, tgt_rot, tcp=gripper.tcp('grasp_center'), max_solutions=1)
        qs = _s[0] if _s else None
        if qs is None:
            wssop.frame(
                pos=tgt_pos,
                rotmat=tgt_rot,
                color_mat=wuc.CoordColor.DYO,
                alpha=0.25,
            ).add_to_scene(scene)
            continue
        n_ik += 1

        wssop.frame(
            pos=tgt_pos,
            rotmat=tgt_rot,
            color_mat=wuc.CoordColor.MYC,
            alpha=0.35,
        ).add_to_scene(scene)

        # Optional self-check in scene only: visualize successful IK.
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.alpha = 0.08
        tmp_robot.add_to_scene(scene)

        # quick jaw-state consistency gate
        if gripper.jaw_range[0] <= jw <= gripper.jaw_range[1]:
            n_ik_valid += 1

    print(f"Total candidates: {n_total}")
    print(f"IK solved: {n_ik}")
    print(f"IK solved + jaw in range: {n_ik_valid}")

    base.run()
