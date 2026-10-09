"""Generate a replay URDF from repository meshes and official FR3 v2 parameters."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def prepare_robot(source: Path, output: Path) -> tuple[Path, Path]:
    config = Path(__file__).parent / "replay_assets/fr3v2"
    inertials = yaml.safe_load((config / "inertials.yaml").read_text())
    hand = yaml.safe_load((config / "hand_inertials.yaml").read_text())
    limits = yaml.safe_load((config / "joint_limits.yaml").read_text())
    kinematics = yaml.safe_load((config / "kinematics.yaml").read_text())
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    for link in root.findall("link"):
        name = link.get("name")
        spec = inertials.get(name.removeprefix("fr3_")) or hand.get(name.removeprefix("panda_"))
        if spec is None:
            continue
        old = link.find("inertial")
        if old is not None:
            link.remove(old)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {k: str(v) for k, v in spec["origin"].items()})
        ET.SubElement(inertial, "mass", value=str(spec["mass"]))
        ET.SubElement(inertial, "inertia", {"i" + k: str(v) for k, v in spec["inertia"].items()})
        i = spec["inertia"]
        moments = np.linalg.eigvalsh([[i["xx"], i["xy"], i["xz"]],
                                    [i["xy"], i["yy"], i["yz"]],
                                    [i["xz"], i["yz"], i["zz"]]])
        if moments.min() <= 0 or moments[-1] > moments[:2].sum():
            raise ValueError(f"Invalid official inertia for {name}: {moments}")
    for joint in root.findall("joint"):
        # Both prismatic joints receive the same target in the collector.
        # Avoid the importer's rotational-axis mimic representation for fingers.
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)
        key = joint.get("name").removeprefix("fr3_")
        if key in limits:
            joint.find("limit").attrib.update({k: str(v) for k, v in limits[key]["limit"].items()})
        if key in kinematics:
            kin = kinematics[key]["kinematic"]
            joint.find("origin").attrib.update(
                xyz=" ".join(str(kin[k]) for k in ("x", "y", "z")),
                rpy=" ".join(str(kin[k]) for k in ("roll", "pitch", "yaw")),
            )
    output.mkdir(parents=True, exist_ok=True)
    urdf = output / "fr3v2_panda_gripper.urdf"
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    description = output / "lula_robot.yaml"
    description.write_text(yaml.safe_dump({
        "api_version": 1.0,
        "cspace": [f"fr3_joint{i}" for i in range(1, 8)],
        "default_q": [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8],
        "cspace_to_urdf_rules": [
            {"name": f"panda_finger_joint{i}", "rule": "fixed", "value": 0.03}
            for i in (1, 2)
        ],
    }))
    return urdf, description
