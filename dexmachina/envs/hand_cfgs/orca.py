"""DexMachina hand-cfg for the ORCA (Soft Robotics Lab) v1b tendon-driven
hand. URDFs and meshes are copied from real2sim2target's
`sim_assets/robots/orcahand_v1b_hand/urdf/` and follow the dexmachina
hand-asset processing recipe (6-DoF floating wrist already attached).

PD gains and force ranges mirror `real2sim2target/configs/hands/orca_v1b.py`
so a run here is the cleanest possible apples-to-apples comparison with the
in-repo Isaac Lab port. Finger joint regex `joint_(abd|pip|iip|dip)_*`
matches the orca_v1b joint-naming convention (no L_/R_ prefix on finger
joints; only on the forearm chain).
"""

from os.path import join

from dexmachina.asset_utils import get_urdf_path


orca_asset_dir = "orca_hand/"
left_rel_urdf = join(orca_asset_dir, "orca_left_6dof.urdf")
right_rel_urdf = join(orca_asset_dir, "orca_right_6dof.urdf")


_ORCA_ACTUATORS = {
    "finger": dict(
        joint_exprs=[r"joint_(abd|pip|iip|dip)_(thumb|index|middle|ring|pinky)"],
        kp=50.0,
        kv=1.0,
        force_range=4.0,
    ),
    "wrist_rot": dict(
        joint_exprs=[r"[LR]_forearm_(roll|pitch|yaw)_link_joint"],
        kp=400.0,
        kv=60.0,
        force_range=10.0,
    ),
    "wrist_trans": dict(
        joint_exprs=[r"[LR]_forearm_t[xyz]_link_joint"],
        kp=1000.0,
        kv=60.0,
        force_range=80.0,
    ),
}


ORCA_LEFT_CFG = {
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "left_palm",
    "kpt_link_names": [
        "left_thumb_fingertip",
        "left_index_fingertip",
        "left_middle_fingertip",
        "left_ring_fingertip",
        "left_pinky_fingertip",
    ],
    "actuators": _ORCA_ACTUATORS,
    # collision_groups left empty: --group_collisions is incompatible with
    # post-merge Genesis Pydantic strict-field enforcement (see notes in
    # base_env.py). The bimanual default path runs without it.
    "collision_groups": {},
    "collision_palm_name": "left_palm",
}

ORCA_RIGHT_CFG = {
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "right_palm",
    "kpt_link_names": [
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ],
    "actuators": _ORCA_ACTUATORS,
    "collision_groups": {},
    "collision_palm_name": "right_palm",
}

ORCA_CFGs = dict(
    left=ORCA_LEFT_CFG,
    right=ORCA_RIGHT_CFG,
)
