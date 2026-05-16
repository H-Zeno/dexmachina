"""Render the baked Orca kinematic-retargeting trajectory to an mp4.

Loads:
- ``assets/retargeter_results/orca_hand/s01/box_use_01_vector.npy`` — baked
  per-frame hand_qpos for LEFT and RIGHT hands (produced by
  ``parallel_retarget.py --save_retargeter_only``).
- ``assets/arctic/processed/s01/box_use_01.npy`` — the ARCTIC clip with
  per-frame box pose (``params['obj_trans']``, ``params['obj_quat']``).
- Both Orca URDFs from ``assets/orca_hand/``.

Sets both hands to their retargeted qpos and the box to its demo pose
each frame, renders offscreen, writes mp4 next to the .npy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

import genesis as gs

REPO = Path("/home/zeno-hamers/dexmachina/dexmachina")
RETARGET_NPY = REPO / "assets" / "retargeter_results" / "orca_hand" / "s01" / "box_use_01_vector.npy"
DEMO_NPY = REPO / "assets" / "arctic" / "processed" / "s01" / "box_use_01.npy"
URDF_LEFT = REPO / "assets" / "orca_hand" / "orca_left_6dof.urdf"
URDF_RIGHT = REPO / "assets" / "orca_hand" / "orca_right_6dof.urdf"
_DATA_ROOT = Path(os.environ.get("R2S2T_DATA_ROOT", "/home/zeno-hamers/r2s2t_data"))
OUT_MP4 = _DATA_ROOT / "videos" / "dexmachina_orca_retargeting" / "box_use_01_retargeted_mirrored_left.mp4"


def main() -> None:
    os.chdir(REPO)
    retar = np.load(RETARGET_NPY, allow_pickle=True).item()
    demo = np.load(DEMO_NPY, allow_pickle=True).item()
    params = demo["params"]
    obj_trans = params["obj_trans"]
    obj_quat = params["obj_quat"]
    n_frames = retar["left"]["hand_qpos"].shape[0]
    print(f"Replaying {n_frames} frames")

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60, substeps=1, gravity=(0, 0, 0)),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            enable_joint_limit=False,
        ),
        show_viewer=False,
        use_visualizer=True,
        show_FPS=False,
        vis_options=gs.options.VisOptions(
            plane_reflection=True,
            ambient_light=(0.5, 0.5, 0.5),
            lights=[
                {"type": "directional", "dir": (0.3, 0.3, -1), "color": (1, 1, 1), "intensity": 3.0},
            ],
        ),
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    hands = {}
    for side, urdf in [("left", URDF_LEFT), ("right", URDF_RIGHT)]:
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf),
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
                collision=False,
            )
        )
        hands[side] = entity

    # NB: fixed=False is required for set_pos / set_quat to take effect;
    # Genesis silently ignores those calls on fixed=True entities.
    obj_urdf = REPO / "assets" / "arctic" / "box" / "decomp" / "box_decomp.urdf"
    obj = scene.add_entity(
        gs.morphs.URDF(
            file=str(obj_urdf),
            fixed=False,
            merge_fixed_links=False,
            recompute_inertia=True,
            collision=False,
        )
    )

    # Top-down 3/4 view. Camera elevated above the action and slightly forward
    # so both hands appear flanking the box. The hands grip the box at world
    # +x ~ +0.21 (left hand) and world -x ~ -0.24 (right hand); placing the
    # camera high up means neither hand is hidden behind the box from this POV.
    cam_pos = (0.0, 0.4, 2.0)
    cam_lookat = (0.0, -0.10, 1.10)
    camera = scene.add_camera(
        pos=cam_pos,
        lookat=cam_lookat,
        res=(1280, 720),
        fov=45,
        spp=1,
    )

    scene.build(n_envs=1, env_spacing=(2.0, 2.0))

    # Refresh camera after build (Genesis quirk noted in feedback memory)
    camera.set_pose(pos=cam_pos, lookat=cam_lookat)

    dof_idxs = {}
    for side, entity in hands.items():
        actuated = [j for j in entity.joints if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
        idxs = [j.dof_idx_local for j in actuated]
        names = [j.name for j in actuated]
        baked_names = list(retar[side]["actuated_dof_names"])
        assert names == baked_names, f"DOF order mismatch on {side}:\n  genesis: {names}\n  baked: {baked_names}"
        dof_idxs[side] = idxs

    palm_link = {side: entity.get_link(f"{side}_palm") for side, entity in hands.items()}

    frames: list[np.ndarray] = []
    for t in range(n_frames):
        for side, entity in hands.items():
            qpos = torch.tensor(retar[side]["hand_qpos"][t], dtype=torch.float32).unsqueeze(0)
            entity.set_dofs_position(
                position=qpos,
                dofs_idx_local=dof_idxs[side],
                zero_velocity=True,
            )

        pos = torch.tensor(obj_trans[t], dtype=torch.float32).unsqueeze(0)
        quat = torch.tensor(obj_quat[t], dtype=torch.float32).unsqueeze(0)
        obj.set_pos(pos=pos)
        obj.set_quat(quat=quat)

        scene.step()
        frame, *_ = camera.render()
        frames.append(frame)
        if t % 100 == 0:
            l_pos = palm_link["left"].get_pos().cpu().numpy().squeeze()
            r_pos = palm_link["right"].get_pos().cpu().numpy().squeeze()
            o_pos = obj.get_pos().cpu().numpy().squeeze()
            print(f"  t={t:4d}  left_palm={l_pos.round(3)}  right_palm={r_pos.round(3)}  obj={o_pos.round(3)}")

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(OUT_MP4), frames, fps=30, quality=8)
    print(f"\nSaved {len(frames)} frames to {OUT_MP4}")


if __name__ == "__main__":
    main()
