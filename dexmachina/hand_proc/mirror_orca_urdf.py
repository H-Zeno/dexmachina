"""Generate a properly-mirrored LEFT Orca 6-DoF URDF from the working RIGHT URDF.

The existing `orca_left_6dof.urdf` in dexmachina (and in real2sim2target's
sim_assets) is a copy of the right URDF with `right_` -> `left_` rename only,
so the left hand carries right-handed geometry. This script:

1. Reads the right URDF.
2. Reflects every <origin xyz=".." rpy=".."/> across the YZ plane (flip x,
   pitch, yaw) so each joint and inertial frame moves to its mirror position.
3. Renames identifiers (R_ -> L_, right_ -> left_) on link / joint / parent /
   child references.
4. Rewrites mesh refs to point at the already-mirrored visual + collision STLs
   shipped in real2sim2target's GeoRT bundle (orca_v1b_left/meshes/{visual,
   collision}). These are byte-different from the right STLs (verified).
5. Writes the result as a new URDF.

Run:
    python /tmp/mirror_orca_urdf.py <right.urdf> <left_meshes_src_dir> <left.urdf.out>

Idempotent. Pure ElementTree text manipulation, no Pinocchio / etc.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


def _flip_origin(xml: str) -> str:
    """Replace every `<origin xyz="x y z" rpy="r p y"/>` with x->-x, p->-p, y->-y."""

    origin_re = re.compile(
        r'(<origin\s+xyz=")(-?\d+(?:\.\d+(?:[eE]-?\d+)?)?)(\s+)'  # xyz x
        r'(-?\d+(?:\.\d+(?:[eE]-?\d+)?)?)(\s+)'                  # xyz y
        r'(-?\d+(?:\.\d+(?:[eE]-?\d+)?)?)("\s+rpy=")'            # xyz z
        r'(-?\d+(?:\.\d+(?:[eE]-?\d+)?)?)(\s+)'                  # rpy r
        r'(-?\d+(?:\.\d+(?:[eE]-?\d+)?)?)(\s+)'                  # rpy p
        r'(-?\d+(?:\.\d+(?:[eE]-?\d+)?)?)("\s*/?>)'              # rpy y
    )

    def _flip_num(s: str) -> str:
        v = float(s)
        v = -v if v != 0.0 else 0.0
        return f"{v:.18g}"

    def _sub(m: re.Match) -> str:
        return (
            f"{m.group(1)}{_flip_num(m.group(2))}{m.group(3)}"
            f"{m.group(4)}{m.group(5)}"
            f"{m.group(6)}{m.group(7)}"
            f"{m.group(8)}{m.group(9)}"
            f"{_flip_num(m.group(10))}{m.group(11)}"
            f"{_flip_num(m.group(12))}{m.group(13)}"
        )

    return origin_re.sub(_sub, xml)


def _rename_sides(xml: str) -> str:
    # Order matters: rename longest distinguishing token first to avoid partials.
    xml = re.sub(r'\bR_forearm_', 'L_forearm_', xml)
    xml = re.sub(r'\bright_', 'left_', xml)
    return xml


def _rewrite_axes(xml: str) -> str:
    """Joint axes for the YZ-plane mirror — *revolute joints only*.

    The geometric YZ-plane reflection rules are:
    - **Revolute** axes are pseudovectors: ``(a, b, c) -> (a, -b, -c)``
      (x preserved, y and z negated).
    - **Prismatic** axes are polar vectors: ``(a, b, c) -> (-a, b, c)``
      (only x flips).

    We deliberately leave **prismatic axes UNCHANGED**. Reason: upstream's
    ``retarget_utils.retarget_one_hand`` (lines 119-124) substitutes the
    raw MANO wrist position into the prismatic qpos values directly via
    ``val += wrist_pos[0/1/2]``, assuming `tx/ty/tz` axes point along
    world `+x/+y/+z`. Flipping the LEFT prismatic ``tx`` axis to
    ``(-1, 0, 0)`` would invert this short-circuit and place the LEFT
    palm at world ``-x`` (colocated with the RIGHT hand). The "mirror"
    character of LEFT vs RIGHT is therefore carried entirely by mirrored
    joint origins + flipped revolute axes; the prismatic floating-base
    DOFs stay in world frame on both sides.

    Parsed via lxml because regex can't reliably scope an ``<axis>`` tag
    to its surrounding ``<joint type="...">`` element.
    """

    from lxml import etree as _et

    parser = _et.XMLParser(remove_comments=True)
    root = _et.fromstring(xml.encode("utf-8"), parser=parser)

    def _fmt(v: float) -> str:
        return "0" if v == 0.0 else f"{v:.18g}"

    for joint in root.findall(".//joint"):
        if joint.get("type", "") != "revolute":
            continue
        axis = joint.find("axis")
        if axis is None:
            continue
        comps = [float(x) for x in axis.get("xyz", "").split()]
        if len(comps) != 3:
            continue
        a, b, c = comps
        a_new = a
        b_new = -b if b != 0.0 else 0.0
        c_new = -c if c != 0.0 else 0.0
        axis.set("xyz", f"{_fmt(a_new)} {_fmt(b_new)} {_fmt(c_new)}")

    return _et.tostring(root, pretty_print=False).decode("utf-8")


# NOTE: joint limits are intentionally NOT rewritten.
#
# Reasoning: under the YZ-plane axis flip ``(a, b, c) -> (a, -b, -c)``, the
# parent frame and the joint axis BOTH transform consistently. Empirically
# (and by working out the right-hand rule on a representative joint), the net
# effect is that ``+qpos`` on the mirrored hand produces the world-frame
# *mirror* of the rotation that ``+qpos`` produces on the original hand.
# Since the original ``+qpos`` = flexion, the mirror is also flexion (just
# on the mirrored side of the body). So the convention "positive qpos =
# flexion" is preserved on both hands, and the original limit range applies
# unchanged. Swapping/negating limits asymmetrically clips flexion on the
# mirrored hand.


def _rewrite_mesh_refs(xml: str, mesh_subdir_rel: str) -> str:
    """Point mesh filenames at `<mesh_subdir_rel>/<basename>.stl` so the LEFT
    URDF uses the already-mirrored STLs we will copy in. We assume the right
    URDF references flat filenames (e.g. ``filename="converted_palm_mesh.stl"``)
    — keep the basename, just prepend the subdir.
    """

    mesh_re = re.compile(r'(<mesh\s+filename=")([^"]+)(")')

    def _sub(m: re.Match) -> str:
        full = m.group(2)
        basename = Path(full).name
        return f'{m.group(1)}{mesh_subdir_rel}/{basename}{m.group(3)}'

    return mesh_re.sub(_sub, xml)


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: mirror_orca_urdf.py <right.urdf> <geort_left_meshes_root> <out.urdf>")
        sys.exit(2)

    right_urdf = Path(sys.argv[1])
    geort_left_root = Path(sys.argv[2])
    out_urdf = Path(sys.argv[3])

    xml = right_urdf.read_text()

    xml = _flip_origin(xml)
    xml = _rewrite_axes(xml)
    xml = _rename_sides(xml)
    xml = _rewrite_mesh_refs(xml, mesh_subdir_rel="left_meshes")

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(xml)

    # Copy mirrored meshes from the GeoRT bundle into <out_urdf.parent>/left_meshes/
    dest_meshes = out_urdf.parent / "left_meshes"
    dest_meshes.mkdir(parents=True, exist_ok=True)
    for sub in ("visual", "collision"):
        src = geort_left_root / "meshes" / sub
        if not src.exists():
            print(f"warning: missing {src}", file=sys.stderr)
            continue
        for stl in src.glob("*.stl"):
            shutil.copy2(stl, dest_meshes / stl.name)

    print(f"wrote {out_urdf}")
    print(f"copied {sum(1 for _ in dest_meshes.glob('*.stl'))} mirrored mesh files to {dest_meshes}")


if __name__ == "__main__":
    main()
