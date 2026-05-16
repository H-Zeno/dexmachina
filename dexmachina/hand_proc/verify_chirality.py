"""Verify LEFT/RIGHT URDF chirality consistency for a dexmachina hand.

Catches three classes of bug that have hit this codebase:

1. **Floating-base prismatic axes not world-aligned.**
   ``dexmachina/retargeting/retarget_utils.py:119-124`` (``retarget_one_hand``)
   short-circuits the IK solver for joint names containing ``tx``/``_x``,
   ``ty``/``_y``, ``tz``/``_z`` by doing ``val += wrist_pos[0/1/2]``. This
   assumes axes point along world ``+x``/``+y``/``+z`` on BOTH hands. If a
   LEFT URDF has e.g. ``tx`` axis ``(-1, 0, 0)``, the LEFT hand gets placed at
   the mirror of the intended world x and colocates with the RIGHT hand.
   (This bit Orca; see commit history of orca_left_6dof.urdf.)

2. **Side-prefix drift in joint names** (``L_forearm_*`` in a RIGHT URDF or
   vice versa). The retargeter's substring match on ``tx``/``ty``/``tz`` still
   works, but every other consumer that keys off the L_/R_ prefix (configs,
   logging, action labelling) silently mis-attributes the affected hand.
   ``dex3_hand/dex3_right_6dof.urdf`` currently has this.

3. **Revolute axis mismatch with the inferred mirror plane.** Different hands
   in this repo use different mirror planes — Orca is YZ-mirrored (negate x in
   origins), Dex3/XHand are XZ-mirrored (negate y in origins). The revolute
   axes must follow the pseudovector rule for whichever plane is used:
   * YZ-plane: ``(a, b, c) -> (a, -b, -c)``  (x kept, y/z negated)
   * XZ-plane: ``(a, b, c) -> (-a, b, -c)``  (y kept, x/z negated)
   If the origins are mirrored one way but the axes another, fingers curl
   the wrong way under retargeting.

Usage:
    python verify_chirality.py <left.urdf> <right.urdf>
    python verify_chirality.py --hand orca   # auto-resolves under assets/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"

# Heuristic URDF paths per hand — extend as new hands ship.
HAND_PAIRS: dict[str, tuple[str, str]] = {
    "orca":    ("orca_hand/orca_left_6dof.urdf",       "orca_hand/orca_right_6dof.urdf"),
    "dex3":    ("dex3_hand/dex3_left_6dof.urdf",       "dex3_hand/dex3_right_6dof.urdf"),
    "xhand":   ("xhand/xhand_left_6dof.urdf",          "xhand/xhand_right_6dof.urdf"),
    "ability": ("ability_hand/ability_hand_left_6dof.urdf",
                "ability_hand/ability_hand_right_6dof.urdf"),
    "inspire": ("inspire_hand/left_xyz_copy.urdf",     "inspire_hand/right_xyz_copy.urdf"),
}

_EPS = 1e-6


@dataclass
class Finding:
    severity: str  # "OK", "WARN", "ERROR"
    message: str


def _axis(joint: etree._Element) -> tuple[float, float, float] | None:
    ax = joint.find("axis")
    if ax is None:
        return None
    parts = ax.get("xyz", "").split()
    if len(parts) != 3:
        return None
    return tuple(float(x) for x in parts)  # type: ignore[return-value]


def _origin_xyz(joint: etree._Element) -> tuple[float, float, float] | None:
    org = joint.find("origin")
    if org is None:
        return None
    parts = org.get("xyz", "").split()
    if len(parts) != 3:
        return None
    return tuple(float(x) for x in parts)  # type: ignore[return-value]


def _name_to_side_agnostic(name: str) -> str:
    """Strip L_/R_ and left_/right_ prefixes so LEFT and RIGHT joints align."""
    for lpref, rpref in (("L_", "R_"), ("left_", "right_")):
        if name.startswith(lpref):
            return name[len(lpref):]
        if name.startswith(rpref):
            return name[len(rpref):]
    return name


def _joints_by_key(tree: etree._ElementTree) -> dict[str, etree._Element]:
    out: dict[str, etree._Element] = {}
    for j in tree.getroot().findall(".//joint"):
        if j.get("type") not in ("revolute", "prismatic"):
            continue
        key = _name_to_side_agnostic(j.get("name", ""))
        out[key] = j
    return out


def _infer_mirror_plane(L: dict[str, etree._Element], R: dict[str, etree._Element]) -> str | None:
    """Pick the plane whose component is negated on more finger joint origins.

    Returns 'YZ' (negate x), 'XZ' (negate y), or None if undetectable.
    Floating-base joints are excluded because their origins are typically zero.
    """
    x_negs = y_negs = total = 0
    for key in sorted(L.keys() & R.keys()):
        if "forearm" in key or "tx" in key or "ty" in key or "tz" in key:
            continue
        lo = _origin_xyz(L[key])
        ro = _origin_xyz(R[key])
        if lo is None or ro is None:
            continue
        if abs(lo[0]) > _EPS or abs(ro[0]) > _EPS:
            total += 1
            if abs(lo[0] + ro[0]) < _EPS:
                x_negs += 1
        if abs(lo[1]) > _EPS or abs(ro[1]) > _EPS:
            total += 1
            if abs(lo[1] + ro[1]) < _EPS:
                y_negs += 1
    if total == 0:
        return None
    if x_negs > y_negs:
        return "YZ"
    if y_negs > x_negs:
        return "XZ"
    return None


def _pseudovector_mirror(axis: tuple[float, float, float], plane: str) -> tuple[float, float, float]:
    a, b, c = axis
    if plane == "YZ":
        return (a, -b, -c)
    if plane == "XZ":
        return (-a, b, -c)
    return axis


def _close(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return all(abs(x - y) < _EPS for x, y in zip(a, b))


def verify(left_path: Path, right_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    L = _joints_by_key(etree.parse(str(left_path)))
    R = _joints_by_key(etree.parse(str(right_path)))

    # Check 1: side-prefix sanity. Joint name starting with L_/left_ should
    # live in the LEFT URDF, R_/right_ in the RIGHT.
    for label, tree_path, expect_pref, wrong_pref in (
        ("LEFT", left_path, ("L_", "left_"), ("R_", "right_")),
        ("RIGHT", right_path, ("R_", "right_"), ("L_", "left_")),
    ):
        for j in etree.parse(str(tree_path)).getroot().findall(".//joint"):
            if j.get("type") not in ("revolute", "prismatic"):
                continue
            name = j.get("name", "")
            if any(name.startswith(p) for p in wrong_pref):
                findings.append(Finding(
                    "ERROR",
                    f"{label} URDF contains joint with opposite-side prefix: {name!r}",
                ))

    # Check 2: matching joint coverage between sides.
    only_left = sorted(L.keys() - R.keys())
    only_right = sorted(R.keys() - L.keys())
    if only_left:
        findings.append(Finding("WARN", f"joints only on LEFT: {only_left}"))
    if only_right:
        findings.append(Finding("WARN", f"joints only on RIGHT: {only_right}"))

    # Check 3: prismatic floating-base axes must be world-aligned on both sides.
    # The retarget_one_hand short-circuit assumes +x/+y/+z. Any deviation is fatal.
    expected_prismatic = {
        ("tx",): (1.0, 0.0, 0.0),
        ("ty",): (0.0, 1.0, 0.0),
        ("tz",): (0.0, 0.0, 1.0),
    }
    for tag, expected in expected_prismatic.items():
        token = tag[0]
        for label, joints in (("LEFT", L), ("RIGHT", R)):
            matches = [k for k in joints if token in k]
            for key in matches:
                j = joints[key]
                if j.get("type") != "prismatic":
                    continue
                ax = _axis(j)
                if ax is None:
                    continue
                if not _close(ax, expected):
                    findings.append(Finding(
                        "ERROR",
                        f"{label} prismatic {key!r} axis={ax} != world-aligned {expected}; "
                        "violates retarget_utils.retarget_one_hand assumption",
                    ))

    # Check 4: infer mirror plane and verify revolute axes follow the
    # pseudovector rule for it.
    plane = _infer_mirror_plane(L, R)
    if plane is None:
        findings.append(Finding("WARN", "Could not infer mirror plane (origins look identical)."))
    else:
        findings.append(Finding("OK", f"Inferred mirror plane: {plane} (origins negate {'x' if plane == 'YZ' else 'y'})"))
        for key in sorted(L.keys() & R.keys()):
            jl, jr = L[key], R[key]
            if jl.get("type") != "revolute" or jr.get("type") != "revolute":
                continue
            la, ra = _axis(jl), _axis(jr)
            if la is None or ra is None:
                continue
            expected_left = _pseudovector_mirror(ra, plane)
            if not _close(la, expected_left):
                findings.append(Finding(
                    "WARN",
                    f"revolute {key!r}: LEFT axis={la} but pseudovector mirror of RIGHT "
                    f"axis={ra} across {plane} = {tuple(round(x, 4) for x in expected_left)}",
                ))

    if not any(f.severity in ("ERROR", "WARN") for f in findings):
        findings.append(Finding("OK", "all chirality checks passed"))
    return findings


def _resolve_pair(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.hand:
        if args.hand not in HAND_PAIRS:
            sys.exit(f"unknown hand {args.hand!r}; known: {sorted(HAND_PAIRS)}")
        lrel, rrel = HAND_PAIRS[args.hand]
        return ASSETS_ROOT / lrel, ASSETS_ROOT / rrel
    if not (args.left and args.right):
        sys.exit("provide --hand <name> or both <left.urdf> <right.urdf>")
    return Path(args.left), Path(args.right)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand", help=f"shortcut: one of {sorted(HAND_PAIRS)}")
    parser.add_argument("left", nargs="?", help="path to LEFT URDF")
    parser.add_argument("right", nargs="?", help="path to RIGHT URDF")
    args = parser.parse_args()
    left_path, right_path = _resolve_pair(args)

    if not left_path.exists() or not right_path.exists():
        sys.exit(f"URDF not found: {left_path if not left_path.exists() else right_path}")

    print(f"LEFT : {left_path}")
    print(f"RIGHT: {right_path}")
    findings = verify(left_path, right_path)
    exit_code = 0
    for f in findings:
        print(f"  [{f.severity:5s}] {f.message}")
        if f.severity == "ERROR":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
