#!/usr/bin/env python3
"""Replay ``Trajectory_successful(2)`` in Isaac Sim 6 and stage DemoGrasp data.

The script intentionally keeps the Isaac Sim process separate from LeRobot.  Isaac
Sim writes lossless per-frame ``npz`` files and a manifest; ``convert_replay_to_v3.py``
then encodes exactly those frames with the installed LeRobot package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "Trajectory_successful(2)"
DEFAULT_USD = ROOT / "FR3v2(1).usd"
if not DEFAULT_USD.is_file():
    DEFAULT_USD = ROOT / "FR3v2.usd"
DEFAULT_URDF = ROOT / "assets/fr3_gripper/fr3_panda_gripper.urdf"
# PhysX friction is dimensionless.  Values around 0.5-1.5 are already high
# friction; 70 makes the gripper behave like a numerically sticky constraint.
CONTACT_STATIC_FRICTION = 1.2
CONTACT_DYNAMIC_FRICTION = 1.0
CONTACT_RESTITUTION = 0.0


class H264FrameRoundTrip:
    """Encode/decode one RGB frame with the same H.264 pixel format as v3."""

    def __init__(self, width: int = 256, height: int = 256, fps: int = 10) -> None:
        import av

        self.av = av
        self.encoder = av.CodecContext.create("libx264", "w")
        self.encoder.width = width
        self.encoder.height = height
        self.encoder.pix_fmt = "yuv420p"
        self.encoder.time_base = Fraction(1, fps)
        self.encoder.framerate = Fraction(fps, 1)
        self.encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "18"}
        self.decoder = av.CodecContext.create("h264", "r")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        frame = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        decoded = []
        for packet in self.encoder.encode(frame):
            decoded.extend(self.decoder.decode(packet))
        if len(decoded) != 1:
            raise RuntimeError(f"H.264 round-trip produced {len(decoded)} frames for one input")
        return decoded[0].to_ndarray(format="rgb24")


def xyz(value: dict) -> np.ndarray:
    return np.asarray([value[axis] for axis in ("x", "y", "z")], dtype=np.float32)


def quat(value: dict) -> np.ndarray:
    # Dataset quaternions are xyzw. Isaac Sim APIs require wxyz.
    if "w" not in value:
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler("xyz", [value[a] for a in "xyz"]).as_quat().astype(np.float32)
    return np.asarray([value.get(axis, 0.0) for axis in ("x", "y", "z", "w")], dtype=np.float32)


def source_files(raw: Path) -> list[Path]:
    files = sorted(raw.glob("Trajectory_*_filtered_*.json"), key=lambda p: int(p.name.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"No filtered JSON trajectories found in {raw}")
    return files


def resample(payload: dict, trajectory: dict, steps: int) -> np.ndarray:
    points = np.asarray([[p[a] for a in ("x", "y", "z")] for p in trajectory["points"]], dtype=np.float32)
    rotations = np.asarray([[r.get(a, 0.0) for a in ("x", "y", "z")] for r in trajectory["rotations"]], dtype=np.float32)
    grips = np.asarray(trajectory["gripperStates"], dtype=bool)
    if not (len(points) == len(rotations) == len(grips)) or len(points) < 2:
        raise ValueError(f"trajectory {trajectory.get('trajectoryId')} has inconsistent samples")
    transitions = np.flatnonzero(grips[1:] != grips[:-1]) + 1
    if len(transitions) != 2 or not grips[transitions[0]] or grips[transitions[1]]:
        raise ValueError(f"trajectory {trajectory.get('trajectoryId')} must close once and open once")
    close_i, open_i = map(int, transitions)
    if steps != 40:
        raise ValueError("DemoGrasp collection requires 40 frames per episode")
    samples = np.concatenate([
        np.linspace(0, close_i, 10), np.full(2, close_i),
        np.linspace(close_i + 1, open_i - 1, 20), np.full(2, open_i),
        np.linspace(min(open_i + 1, len(points) - 1), len(points) - 1, 6),
    ])
    if len(samples) != steps:
        raise ValueError(f"resampling produced {len(samples)} frames")
    pos = np.column_stack([np.interp(samples, np.arange(len(points)), points[:, i]) for i in range(3)])
    q = np.empty((steps, 4), dtype=np.float32)
    # Source rotations are XYZ Euler angles. Convert without scipy dependency.
    for j, index in enumerate(np.rint(samples).astype(int)):
        rx, ry, rz = rotations[index] * 0.5
        cx, cy, cz, sx, sy, sz = np.cos(rx), np.cos(ry), np.cos(rz), np.sin(rx), np.sin(ry), np.sin(rz)
        q[j] = (sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz)
    g_open = float(payload["gripperOpenAngle"])
    g_closed = float(payload["gripperClosedAngle"])
    g = np.asarray([g_open] * 11 + [g_closed] * 21 + [g_open] * 8, dtype=np.float32)[:, None]
    return np.concatenate((pos, q, g), axis=1).astype(np.float32)


def load_episodes(raw: Path, limit: int | None, scene_count: int = 10, per_scene: int = 10,
                  scene_ids: list[int] | None = None, candidate_multiplier: int = 1,
                  trajectory_pairs: set[tuple[int, int]] | None = None) -> list[dict]:
    episodes = []
    files = source_files(raw)
    if scene_ids:
        files = [f for f in files if int(f.name.split("_")[1]) in scene_ids]
    for scene_file in files[:scene_count]:
        payload = json.loads(scene_file.read_text(encoding="utf-8"))
        candidates = [t for t in payload["trajectories"] if t.get("execution_success")]
        if trajectory_pairs is not None:
            scene_id = int(scene_file.name.split("_")[1])
            candidates = [t for t in candidates if (scene_id, int(t["trajectoryId"])) in trajectory_pairs]
        else:
            candidates = candidates[:per_scene * candidate_multiplier]
        for trajectory in candidates:
            action = resample(payload, trajectory, 40)
            episodes.append({
                "source_file": scene_file.name,
                "trajectory_id": int(trajectory["trajectoryId"]),
                "action": action,
                "cube_a": xyz(payload["defaultCubeAStartPosition"]),
                "cube_b": xyz(payload["defaultCubeBStartPosition"]),
                "cube_a_rotation": quat(payload["defaultCubeAStartRotation"]),
                "cube_b_rotation": quat(payload["defaultCubeBStartRotation"]),
                "boxes": {f"box{i}": {"position": xyz(payload[f"box{i}Position"]),
                                        "rotation": quat(payload[f"box{i}Rotation"])} for i in (1, 2, 3)},
            })
            if limit is not None and len(episodes) >= limit:
                return episodes
    return episodes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--usd", type=Path, default=DEFAULT_USD)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--output", type=Path, default=ROOT / "data/isaacsim_replay")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--scene-count", type=int, default=10)
    p.add_argument("--per-scene", type=int, default=10)
    p.add_argument("--scene-ids", type=int, nargs="+")
    p.add_argument("--trajectories", nargs="+", metavar="SCENE:TRAJECTORY",
                   help="replay these exact source scene and trajectory IDs")
    p.add_argument("--review-video-dir", type=Path,
                   help="record an additional full-scene Isaac Sim replay video per candidate")
    p.add_argument("--candidate-multiplier", type=int, default=5, choices=range(1, 6))
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--resume", action="store_true", help="retain completed attempts and collect missing scene quotas")
    p.add_argument("--num-gpus", type=int, default=1, choices=range(1, 7))
    p.add_argument("--dry-run", action="store_true", help="validate every source JSON without starting Isaac Sim")
    p.add_argument("--physics-steps-per-frame", type=int, default=12)
    p.add_argument("--static-friction", type=float, default=CONTACT_STATIC_FRICTION)
    p.add_argument("--dynamic-friction", type=float, default=CONTACT_DYNAMIC_FRICTION)
    p.add_argument("--restitution", type=float, default=CONTACT_RESTITUTION)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--eval-manifest", type=Path, help="v3 meta/scenes.jsonl for closed-loop evaluation")
    p.add_argument("--eval-index", type=int, help="evaluate exactly one dataset episode")
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=5555)
    return p


def dry_run(args: argparse.Namespace, episodes: list[dict]) -> None:
    heights = {round(float(x["cube_a"][2]), 6) for x in episodes}
    print(json.dumps({"candidate_episodes": len(episodes), "requested_successful_episodes": args.scene_count * args.per_scene,
                      "frames_per_episode": 40,
                      "cube_a_size_m": 0.02, "cube_b_size_m": 0.025,
                      "source_cube_center_height_m": sorted(heights),
                      "usd": str(args.usd), "urdf": str(args.urdf)}, indent=2))


def run_isaac(args: argparse.Namespace, episodes: list[dict]) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(args.num_gpus)))
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "multi_gpu": args.num_gpus > 1,
                         "max_gpu_count": args.num_gpus,
                         "enable_viewport": False, "width": 256, "height": 256})
    try:
        _replay_with_simulation(args, episodes)
    except Exception:
        import traceback
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        # Isaac Sim 6 can still own an asynchronous viewport task in a headless
        # process. Give Kit one frame to drain it before closing.
        try:
            import omni.kit.app
            omni.kit.app.get_app().update()
        except Exception:
            pass
        app.close()


def _replay_with_simulation(args: argparse.Namespace, episodes: list[dict]) -> None:
    """Isaac Sim implementation. Imports stay here so --dry-run works anywhere."""
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim, SingleXFormPrim
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.utils.types import ArticulationAction
    from scipy.spatial.transform import Rotation
    from replay_robot_config import prepare_robot
    from isaacsim.sensors.camera import Camera
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    import socket
    from lerobot_policy.socket_protocol import command_message, read_command, receive_arrays, send_arrays

    def write_review_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required for --review-video-dir")
        height, width, channels = frames[0].shape
        if channels != 3 or any(frame.shape != (height, width, 3) for frame in frames):
            raise ValueError("review-camera frames have inconsistent RGB shapes")
        process = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
             "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
             "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
        finally:
            process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        status = process.wait()
        if status:
            raise RuntimeError(f"ffmpeg failed for {path}: {error}")

    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.review_video_dir is not None:
        args.review_video_dir = args.review_video_dir.resolve()
        args.review_video_dir.mkdir(parents=True, exist_ok=True)
    previous = None
    manifest_path = out / "replay_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest_path} exists; use --resume or a new output directory")
        previous = json.loads(manifest_path.read_text())
        if previous.get("fps") != 120 // args.physics_steps_per_frame:
            raise ValueError("Cannot resume with a different sampling rate")
    configured_scene_input = args.usd.name == "configured_scene.usda" and args.usd.exists()
    omni.usd.get_context().open_stage(str(args.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"failed to open USD {args.usd}")
    # The supplied scene carries a stale controller graph and an unrelated
    # placeholder robot prim. Disable both so the imported URDF is the only
    # active FR3 articulation in the physics scene.
    for stale_path in ("/Graphs", "/fr3v2", "/SimpleRoom"):
        stale = stage.GetPrimAtPath(stale_path)
        if stale.IsValid():
            stale.SetActive(False)
    # The exported configured scene contains the visual/physics robot layer as
    # an audit artifact. Remove that runtime articulation before importing the
    # exact same URDF/mesh again; retaining serialized joint transforms causes
    # PhysX to restore stale link poses on reopen. The scene geometry, cameras,
    # and contact material remain authored by the configured scene.
    if configured_scene_input:
        stage.RemovePrim("/World/ImportedFR3")
        for prim in list(stage.GetPseudoRoot().GetChildren()):
            if prim.GetName().startswith("Flattened_Prototype_"):
                stage.RemovePrim(str(prim.GetPath()))
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
    # Preserve the authored wall geometries as separate fixed colliders.
    def set_pose(path, position, orientation):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"Missing scene prim: {path}")
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, position)))
        q = np.asarray(orientation, dtype=float)
        if np.linalg.norm(q) < 1e-8:
            q = np.array([0, 0, 0, 1.0])
        q /= np.linalg.norm(q)
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3]))))
    # Apply explicit mass values to the existing dynamic cube schemas.
    for dynamic_path in ("/World/cubeA", "/World/cubeB"):
        dynamic_prim = stage.GetPrimAtPath(dynamic_path)
        if dynamic_prim.IsValid():
            UsdPhysics.RigidBodyAPI.Apply(dynamic_prim)
            UsdPhysics.CollisionAPI.Apply(dynamic_prim)
            mass = UsdPhysics.MassAPI.Apply(dynamic_prim)
            mass.CreateMassAttr(0.05 if dynamic_path.endswith("cubeA") else 0.08)
    for fixed_path in ("/World/box", "/World/box1", "/World/box2"):
        fixed_prim = stage.GetPrimAtPath(fixed_path)
        if fixed_prim.IsValid():
            for wall in fixed_prim.GetChildren():
                if wall.IsA(UsdGeom.Cube):
                    UsdPhysics.CollisionAPI.Apply(wall).CreateCollisionEnabledAttr(True)
    for name, size in (("cubeA", 0.02), ("cubeB", 0.025)):
        stage.GetPrimAtPath(f"/World/{name}").GetAttribute("xformOp:scale").Set(Gf.Vec3d(size))
    if not stage.GetPrimAtPath("/World/defaultGroundPlane").IsValid():
        world.scene.add_default_ground_plane(z_position=0.0)
    # Import the repository URDF with the FR3 v2 joint names and mimic finger.
    # A configured scene already contains the repaired robot reference. Reuse
    # that exact reference during evaluation so collection and replay share
    # the same mesh layer instead of silently importing a second articulation.
    imported_dir = out / "imported_urdf"; imported_dir.mkdir(exist_ok=True)
    replay_urdf, lula_description = prepare_robot(args.urdf, out / "robot_config")
    robot_usd = URDFImporter(URDFImporterConfig(urdf_path=str(replay_urdf), usd_path=str(imported_dir),
                                                collision_from_visuals=False, merge_mesh=False,
                                                allow_self_collision=False)).import_urdf()
    # Edit the generated robot layer before referencing it into the scene.
    # Once composed as a reference, prototype attributes such as purpose are
    # not writable from the parent stage.
    # The importer stores meshes in payloads/geometries.usd, so repair every
    # generated USD layer rather than only the top-level reference layer.
    generated_layers = sorted(imported_dir.rglob("*.usd")) + sorted(imported_dir.rglob("*.usda"))
    for layer_path in generated_layers:
        robot_stage = Usd.Stage.Open(str(layer_path))
        if robot_stage is None:
            continue
        changed = False
        for prim in robot_stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                purpose = prim.GetAttribute("purpose")
                if purpose.IsValid():
                    purpose.Set(UsdGeom.Tokens.default_)
                else:
                    UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
                changed = True
        if changed:
            robot_stage.GetRootLayer().Save()
        del robot_stage
    # Use the imported URDF articulation for joint control.  The supplied USD
    # remains the scene source for the authored table, cubes and box geometry.
    stage.DefinePrim("/World/ImportedFR3", "Xform").GetReferences().AddReference(robot_usd)
    stage.Load()
    import omni.kit.app
    for _ in range(8):
        omni.kit.app.get_app().update()
    # The Isaac Sim URDF importer emits the visual mesh instances under
    # ``*_1`` link prims, but marks their meshes as purpose=guide.  Guide
    # geometry is omitted by camera rendering, leaving only sparse collision
    # geometry visible.  Restore normal render purpose for visual instances;
    # collision instances remain untouched for physics.
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if path.startswith("/Flattened_Prototype_"):
            # Prototype opinions override instance opinions in the imported
            # USD, so repair the source mesh purpose at this level too.
            purpose = prim.GetAttribute("purpose")
            if purpose.IsValid():
                purpose.Set(UsdGeom.Tokens.default_)
            else:
                UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
            continue
        if not path.startswith("/World/ImportedFR3/"):
            continue
        visual_instance = any(part.endswith("_1") for part in path.split("/")[:-1])
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
        if not visual_instance:
            # URDF importer creates a sibling collision instance (linkN,
            # hand, finger) and a visual instance (linkN_1, hand_1,
            # finger_1). Hide only the former from rendering; collision
            # shapes remain active in PhysX because visibility is visual-only.
            for ancestor in prim.GetPath().GetPrefixes():
                name = ancestor.name
                if name in {"hand", "finger"} or (name.startswith("link") and name[4:].isdigit()):
                    UsdGeom.Imageable(stage.GetPrimAtPath(ancestor)).CreateVisibilityAttr().Set(
                        UsdGeom.Tokens.invisible
                    )
                    break
    stage.Load()
    for _ in range(4):
        omni.kit.app.get_app().update()
    imported_root = "/World/ImportedFR3/Geometry/fr3_link0"
    if not stage.GetPrimAtPath(imported_root).IsValid():
        candidates = [str(p.GetPath()) for p in stage.Traverse()
                      if p.GetTypeName() == "Xform" and "ImportedFR3" in str(p.GetPath())
                      and "link0" in str(p.GetPath())]
        raise RuntimeError(f"Imported FR3 root was not composed at {imported_root}; candidates={candidates}")
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/ImportedFR3")).SetTranslate(Gf.Vec3d(-0.615, 0.0, 0.0))
    root_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/World/ImportedFR3/Physics/root_joint"))
    root_joint.GetBody0Rel().ClearTargets(True)
    root_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    # URDF dynamics damping does not supply a position-drive stiffness.
    # Set explicit PD gains; otherwise position targets exert no restoring force.
    for number in range(1, 8):
        joint = stage.GetPrimAtPath(f"/World/ImportedFR3/Physics/fr3_joint{number}")
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(1200.0)
        drive.CreateDampingAttr(80.0)
    for number in (1, 2):
        joint = stage.GetPrimAtPath(f"/World/ImportedFR3/Physics/panda_finger_joint{number}")
        drive = UsdPhysics.DriveAPI.Apply(joint, "linear")
        drive.CreateStiffnessAttr(1500.0)
        drive.CreateDampingAttr(50.0)
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(imported_root))
    articulation_api.CreateSolverPositionIterationCountAttr(16)
    articulation_api.CreateSolverVelocityIterationCountAttr(4)
    # URDF USD default prim is /fr3; its articulation root is the base link.
    # The top-level reference Xform is only a namespace/container.
    # The configured scene already authors the base translation. Passing a
    # second position to SingleArticulation would reset that authored pose on
    # reopen, shifting all EEF coordinates by one robot-base offset.
    robot = SingleArticulation(imported_root, name="fr3_v2", position=np.array([-0.615, 0, 0]))
    world.scene.add(robot)
    cubes = [world.scene.add(SingleRigidPrim(f"/World/{name}", name=name)) for name in ("cubeA", "cubeB")]
    contact_material = PhysicsMaterial(
        "/World/ContactFriction",
        static_friction=args.static_friction,
        dynamic_friction=args.dynamic_friction,
        restitution=args.restitution,
    )
    for name in ("cubeA", "cubeB"):
        SingleGeometryPrim(f"/World/{name}").apply_physics_material(contact_material)
    gripper_material_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/ImportedFR3/") and prim.GetName() in {"panda_leftfinger", "panda_rightfinger"}:
            UsdShade.MaterialBindingAPI(prim).Bind(contact_material.material)
            gripper_material_paths.append(path)
    if not gripper_material_paths:
        raise RuntimeError("No FR3V2 gripper collision meshes found for friction material")
    ee_path = next(str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith("/World/ImportedFR3/") and p.GetName() == "ee_link")
    eef = SingleXFormPrim(ee_path, name="eef")
    cameras = [Camera(f"/World/Camera{i}", name=f"camera{i}", resolution=(256, 256), frequency=30) for i in (1, 2)]
    for camera, eye in zip(cameras, ([0.7, -0.8, 0.7], [0.6, 0.8, 0.65])):
        world.scene.add(camera)
        eye = np.asarray(eye, dtype=float)
        backward = eye - np.array([-0.15, 0, 0.06])
        backward /= np.linalg.norm(backward)
        right = np.cross([0, 0, 1], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        qxyzw = Rotation.from_matrix(np.column_stack((right, up, backward))).as_quat()
        camera.set_world_pose(eye, qxyzw[[3, 0, 1, 2]], camera_axes="usd")
        # Camera.set_focal_length uses centimetres; 1.8 writes the authored
        # USD focalLength=18.  Set the aperture explicitly as well because a
        # newly-created Camera otherwise keeps a different default aperture
        # from the cameras used to record the training videos.
        camera.set_focal_length(1.8)
        camera.prim.GetAttribute("verticalAperture").Set(20.955)
        camera.set_clipping_range(0.01, 100.0)
    review_camera = None
    if args.review_video_dir is not None:
        review_camera = Camera("/World/ReviewCamera", name="review_camera", resolution=(960, 540), frequency=30)
        world.scene.add(review_camera)
        eye = np.array([1.15, -2.05, 1.72], dtype=float)
        target = np.array([-0.25, -0.05, 0.18], dtype=float)
        backward = eye - target
        backward /= np.linalg.norm(backward)
        right = np.cross([0, 0, 1], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        qxyzw = Rotation.from_matrix(np.column_stack((right, up, backward))).as_quat()
        review_camera.set_world_pose(eye, qxyzw[[3, 0, 1, 2]], camera_axes="usd")
        review_camera.set_focal_length(1.8)
        review_camera.prim.GetAttribute("verticalAperture").Set(11.787188)
        review_camera.set_clipping_range(0.01, 100.0)
    configured_scene_path = args.usd.resolve() if configured_scene_input else (out / "configured_scene.usda").resolve()
    if not configured_scene_input:
        stage.Export(str(configured_scene_path))
    world.reset()
    [camera.initialize() for camera in cameras]
    # Camera.initialize() restores its default 1-inch-style aperture and may
    # also adjust it to the render target aspect ratio.  Reapply the authored
    # training-camera intrinsics after initialization so runtime rendering
    # matches the recorded USD cameras.
    for camera in cameras:
        camera.set_focal_length(1.8)
        camera.prim.GetAttribute("verticalAperture").Set(20.955)
        camera.prim.GetAttribute("horizontalAperture").Set(20.955)
        camera.set_clipping_range(0.01, 100.0)
    if review_camera is not None:
        review_camera.initialize()
        review_camera.set_focal_length(1.8)
        review_camera.prim.GetAttribute("verticalAperture").Set(11.787188)
        review_camera.prim.GetAttribute("horizontalAperture").Set(20.955)
        review_camera.set_clipping_range(0.01, 100.0)
    robot.set_joint_positions(np.array([0, -0.6, 0, -2.2, 0, 1.6, 0.8, 0.03, 0.03], dtype=np.float32))
    robot.apply_action(ArticulationAction(joint_positions=robot.get_joint_positions()))
    for _ in range(12):
        world.step(render=True)
    controller = robot.get_articulation_controller()
    dof_names = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7", "panda_finger_joint1"]
    indices = [robot.get_dof_index(name) for name in dof_names]
    print(f"FR3 DOFs: {list(zip(dof_names, indices))}; robot_dof_count={len(robot.get_joint_positions())}", flush=True)
    print(f"drive_properties={robot.dof_properties}", flush=True)
    ik = LulaKinematicsSolver(str(lula_description), str(replay_urdf))
    ik.set_robot_base_pose(np.array([-0.615, 0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    ik_frame = "ee_link"
    records = previous["episodes"] if previous else []
    counts = Counter(r["source_file"] for r in records if r["success"])
    attempted = {(r["source_file"], r["trajectory_id"]) for r in records}
    next_index = max((r["episode_index"] for r in records), default=-1) + 1
    scene_files = sorted({item["source_file"] for item in episodes})
    manifest = {"usd": str(args.usd), "configured_scene": str(configured_scene_path),
                "urdf": str(args.urdf), "replay_urdf": str(replay_urdf),
                "fr3_v2_dof_names": dof_names, "episodes": records,
                "fps": 120 // args.physics_steps_per_frame, "frames_per_episode": 40,
                "scene_files": scene_files, "per_scene": args.per_scene,
                "frame_convention": "observation before action; xyz in FR3 base frame; quaternion xyzw; finger position in metres",
                "retargeting": "source XY preserved; Z adapted to resized cubes, carry clearance, held descent, open and retreat",
                "official_fr3v2_commit": "7aeeddc449edf8d62b594f9e36a81da53e7796f9"}
    manifest["raw_root"] = str(args.raw_root.resolve())
    manifest["source_sha256"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.usd, args.urdf, replay_urdf] + [args.raw_root / name for name in scene_files]
    }
    manifest["configured_scene_sha256"] = hashlib.sha256(configured_scene_path.read_bytes()).hexdigest()
    manifest.update(cube_a_size_m=0.02, cube_b_size_m=0.025, cube_a_mass_kg=0.05, cube_b_mass_kg=0.08,
                    physics_dt_s=1 / 120, physics_steps_per_frame=args.physics_steps_per_frame,
                    camera_resolution=[256, 256], camera_clipping_m=[0.01, 100.0],
                    success_criteria={"max_xy_error_m": 0.009, "stack_center_z_difference_m": 0.0225,
                                      "max_stack_z_error_m": 0.004, "max_settled_drift_m": 0.001,
                                      "min_lift_m": 0.015, "max_instantaneous_speed_mps": 0.05,
                                      "settle_steps": 120, "stable_window_steps": 60})
    manifest["contact_friction"] = {
        "static_friction": args.static_friction,
        "dynamic_friction": args.dynamic_friction,
        "restitution": args.restitution,
        "bound_to": "cubeA,cubeB,FR3V2 panda_leftfinger/panda_rightfinger collision meshes",
    }
    policy_socket = None
    policy_roundtrips = None
    if args.eval_index is not None:
        policy_socket = socket.create_connection((args.policy_host, args.policy_port), timeout=120)
        policy_socket.settimeout(120)
        send_arrays(policy_socket, command_message("ping"))
        if read_command(receive_arrays(policy_socket)) != "pong":
            raise RuntimeError("Policy server did not answer ping")
        policy_roundtrips = [H264FrameRoundTrip(), H264FrameRoundTrip()]
    for candidate_index, item in enumerate(episodes):
        if counts[item["source_file"]] >= args.per_scene or (item["source_file"], item["trajectory_id"]) in attempted:
            continue
        episode_index = next_index
        next_index += 1
        a, b = item["cube_a"].copy(), item["cube_b"].copy()
        # Restore all three recorded coordinates; let contact with the authored
        # support geometry determine the resting height of the resized cubes.
        for cube, position, orientation in zip(cubes, (a, b), (item["cube_a_rotation"], item["cube_b_rotation"])):
            cube.set_world_pose(position=position, orientation=orientation[[3, 0, 1, 2]])
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        for i in (1, 2, 3):
            box = item["boxes"][f"box{i}"]
            usd_box_names = {1: "box", 2: "box1", 3: "box2"}
            path = f"/World/{usd_box_names[i]}"
            set_pose(path, box["position"], box["rotation"])
        world.step(render=True)
        # The first observation must be rendered after the initial pose has
        # settled.  Without a rendered step here, Isaac returns the previous
        # camera buffer for frame 0 while the telemetry already reports the
        # new joint state, creating a state/image mismatch at episode start.
        for step in range(120):
            world.step(render=(step == 119))
        resting_a = cubes[0].get_world_pose()[0]
        resting_b = cubes[1].get_world_pose()[0]
        states, actions = [], []
        # Keep joint telemetry for replay audits.  These fields are not policy
        # inputs; DemoGrasp trains on eefpose+handdof for this robot variant.
        joint_positions, joint_targets = [], []
        frames = {"observation.camera_1.rgb": [], "observation.camera_2.rgb": []}
        review_frames = []
        retargeted = item["action"].copy()
        # Preserve recorded offsets while retargeting the source cube heights
        # to the explicitly requested 20 mm and 25 mm geometry.
        grasp_delta = float(resting_a[2]) - float(item["cube_a"][2])
        place_delta = float(resting_b[2]) + 0.0125 + 0.01 + 0.005 - float(retargeted[32, 2])
        retargeted[:12, 2] += grasp_delta
        retargeted[12:32, 2] += np.linspace(grasp_delta, place_delta, 20)
        retargeted[32:, 2] += place_delta
        # The smaller target leaves less clearance for the unchanged fingers.
        # Carry above its top, descend while still holding, then release.
        retargeted[14:32, 2] = np.maximum(retargeted[14:32, 2], float(resting_b[2]) + 0.0575)
        retargeted[32, 7] = retargeted[31, 7]
        retargeted[34:, 2] += np.linspace(0.0, 0.08, 6)
        first = retargeted[0]
        start_q, start_ok = ik.compute_inverse_kinematics(ik_frame, first[:3] + [-0.615, 0, 0], first[[6, 3, 4, 5]], position_tolerance=0.001)
        if not start_ok:
            raise RuntimeError("Initial source pose is unreachable")
        robot.set_joint_positions(np.r_[start_q, first[7], first[7]])
        robot.set_joint_velocities(np.zeros(9))
        robot.apply_action(ArticulationAction(joint_positions=np.r_[start_q, first[7], first[7]]))
        # Synchronize the camera buffer with the initial robot pose before
        # recording observation frame 0.
        for step in range(120):
            world.step(render=(step == 119))
        cube_history = []
        ik_failures = 0
        grasp_xy_errors = []
        if policy_socket is not None:
            send_arrays(policy_socket, command_message("reset"))
            if read_command(receive_arrays(policy_socket)) != "ok":
                raise RuntimeError("Policy server did not reset")
        for frame_index, expert_target in enumerate(retargeted):
            # Save the command actually applied after geometry retargeting.
            current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            arm_current = current[:7]
            # Isaac's camera annotator is asynchronous with the physics step;
            # flush the render queue before reading the observation buffer.
            world.render()
            rgb = []
            for camera in cameras:
                image = camera.get_rgb()
                for _ in range(4):
                    if image is not None:
                        break
                    world.render()
                    image = camera.get_rgb()
                if image is None:
                    raise RuntimeError(f"camera {camera.name} did not produce RGB after warmup")
                image = np.asarray(image, dtype=np.uint8)
                if image.ndim != 3 or image.shape[0] != 256 or image.shape[1] != 256:
                    raise RuntimeError(f"camera {camera.name} returned invalid RGB shape {image.shape}")
                rgb.append(image[:, :, :3].copy())
            policy_rgb = rgb
            if policy_roundtrips is not None:
                policy_rgb = [roundtrip(image) for roundtrip, image in zip(policy_roundtrips, rgb)]
            eef_position, eef_orientation = eef.get_world_pose()
            state_now = np.r_[eef_position - np.array([-0.615, 0, 0]),
                              eef_orientation[[1, 2, 3, 0]], current[7]].astype(np.float32)
            if policy_socket is None:
                target = expert_target
            else:
                send_arrays(policy_socket, {
                    "command": np.asarray("predict"),
                    "observation.state": state_now[None],
                    "observation.camera_1.rgb": policy_rgb[0][None],
                    "observation.camera_2.rgb": policy_rgb[1][None],
                })
                prediction = receive_arrays(policy_socket)
                if read_command(prediction) != "action" or prediction["action"].shape != (1, 8):
                    raise RuntimeError(f"Invalid policy action at frame {frame_index}")
                target = prediction["action"][0]
            target_position = target[:3].astype(np.float64).copy()
            # Source trajectories are expressed in the FR3 base frame.  The
            # imported FR3 base is placed at the USD pose (-0.615, 0, 0).
            target_position += np.array([-0.615, 0.0, 0.0], dtype=np.float64)
            if 10 <= frame_index <= 31:
                cube_a_position = np.asarray(cubes[0].get_world_pose()[0], dtype=np.float64)
                grasp_xy_errors.append(float(np.linalg.norm(target_position[:2] - cube_a_position[:2])))
            ik_q, ik_ok = ik.compute_inverse_kinematics(
                ik_frame, target_position, target[[6, 3, 4, 5]].astype(np.float64),
                warm_start=arm_current.astype(np.float64), position_tolerance=0.0002,
                orientation_tolerance=0.01,
            )
            ik_failures += int(not ik_ok)
            q = np.zeros(len(indices), dtype=np.float32)
            q[:7] = np.asarray(ik_q if ik_ok else arm_current, dtype=np.float32)
            q[7] = float(target[7])
            if not np.isfinite(q).all():
                raise RuntimeError(f"Nonfinite robot state in episode {episode_index}, frame {frame_index}")
            if args.diagnostics and frame_index in (0, 10, 20, 30, 39):
                actual = np.asarray(robot.get_joint_positions(), dtype=np.float64)
                robot_world = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath("/World/ImportedFR3")).ComputeLocalToWorldTransform(0).ExtractTranslation())
                eef_prim = stage.GetPrimAtPath(ee_path)
                eef_world = np.asarray(UsdGeom.Xformable(eef_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()) if eef_prim.IsValid() else None
                cube_a_now = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath("/World/cubeA")).ComputeLocalToWorldTransform(0).ExtractTranslation())
                cube_b_now = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath("/World/cubeB")).ComputeLocalToWorldTransform(0).ExtractTranslation())
                try:
                    eef_pos, eef_quat = ik.compute_forward_kinematics(ik_frame, actual[:7])
                    print(f"ep={episode_index} frame={frame_index} ik={ik_ok} robot_world={robot_world.tolist()} target={target_position.tolist()} q={q.tolist()} actual={actual.tolist()} eef={np.asarray(eef_pos).tolist()} usd_eef={None if eef_world is None else eef_world.tolist()} cubes={cube_a_now.tolist()},{cube_b_now.tolist()}", flush=True)
                except Exception as exc:
                    print(f"ep={episode_index} frame={frame_index} ik={ik_ok} robot_world={robot_world.tolist()} target={target_position.tolist()} q={q.tolist()} actual={actual.tolist()} usd_eef={None if eef_world is None else eef_world.tolist()} cubes={cube_a_now.tolist()},{cube_b_now.tolist()} eef_error={exc}", flush=True)
            if review_camera is not None:
                image = review_camera.get_rgb()
                for _ in range(4):
                    if image is not None:
                        break
                    world.render()
                    image = review_camera.get_rgb()
                if image is None:
                    raise RuntimeError("review camera did not produce RGB after warmup")
                review_frames.append(np.asarray(image, dtype=np.uint8)[:, :, :3].copy())
            states.append(state_now)
            actions.append(target.astype(np.float32)); frames["observation.camera_1.rgb"].append(rgb[0]); frames["observation.camera_2.rgb"].append(rgb[1])
            joint_positions.append(current[:7].copy())
            joint_targets.append(q[:7].copy())
            controller.apply_action(ArticulationAction(joint_positions=np.r_[q, q[7]], joint_indices=np.r_[indices, robot.get_dof_index("panda_finger_joint2")]))
            for physics_step in range(args.physics_steps_per_frame):
                world.step(render=(physics_step == args.physics_steps_per_frame - 1))
            cube_history.append([cubes[0].get_world_pose()[0].tolist(), cubes[1].get_world_pose()[0].tolist()])
        state = np.asarray(states, dtype=np.float32)
        action = np.asarray(actions, dtype=np.float32)
        if args.diagnostics:
            np.savez_compressed(out / f"diagnostic_{episode_index:06d}.npz", state=state, action=action,
                            camera_1_rgb=np.asarray(frames["observation.camera_1.rgb"]),
                            camera_2_rgb=np.asarray(frames["observation.camera_2.rgb"]))
        # Physical success: cube A is stably stacked on cube B after settling.
        stable = []
        settle_positions = []
        for step in range(120):
            world.step(render=review_camera is not None and step % 12 == 11)
            if review_camera is not None and step % 12 == 11:
                image = review_camera.get_rgb()
                if image is None:
                    raise RuntimeError("review camera did not produce RGB during settling")
                review_frames.append(np.asarray(image, dtype=np.uint8)[:, :, :3].copy())
            if step >= 60:
                pa, pb = [cube.get_world_pose()[0] for cube in cubes]
                settle_positions.append([pa.copy(), pb.copy()])
                stable.append(bool(np.linalg.norm(pa[:2] - pb[:2]) <= 0.009 and abs(float(pa[2] - pb[2]) - 0.0225) <= 0.004))
        pa = cubes[0].get_world_pose()[0]
        pb = cubes[1].get_world_pose()[0]
        xy = float(np.linalg.norm(pa[:2] - pb[:2])); zd = float(pa[2] - pb[2])
        speed_a, speed_b = [float(np.linalg.norm(cube.get_linear_velocity())) for cube in cubes]
        position_window = np.asarray(settle_positions)
        position_drift = np.max(np.linalg.norm(position_window - position_window[0], axis=-1), axis=0)
        lifted = max(frame[0][2] for frame in cube_history) > float(resting_a[2]) + 0.015
        success = all(stable) and lifted and np.all(position_drift < 0.001) and speed_a < 0.05 and speed_b < 0.05 and ik_failures == 0
        grasp_xy_error = float(min(grasp_xy_errors)) if grasp_xy_errors else float("inf")
        if review_camera is not None:
            scene_id = int(item["source_file"].split("_")[1])
            status = "SUCCESS" if success else "REJECTED"
            video_path = args.review_video_dir / f"scene_{scene_id:02d}_trajectory_{item['trajectory_id']:03d}_{status}.mp4"
            write_review_video(review_frames, video_path, 10)
            video_path.with_suffix(".json").write_text(json.dumps({
                "source_file": item["source_file"], "trajectory_id": item["trajectory_id"],
                "replay_episode_index": episode_index, "success": bool(success), "frames": len(review_frames),
                "fps": 10, "resolution": [960, 540], "camera": "Isaac Sim full-scene overview",
                "xy_error_m": xy, "z_delta_m": zd, "settled_position_drift_m": position_drift.tolist(),
                "grasp_xy_error_m": grasp_xy_error,
                "ik_failures": ik_failures,
            }, indent=2) + "\n", encoding="utf-8")
        if success or policy_socket is not None:
            np.savez_compressed(out / f"episode_{episode_index:06d}.npz", state=state, action=action,
                                joint_position=np.asarray(joint_positions, dtype=np.float32),
                                joint_target=np.asarray(joint_targets, dtype=np.float32),
                                camera_1_rgb=np.asarray(frames["observation.camera_1.rgb"]),
                                camera_2_rgb=np.asarray(frames["observation.camera_2.rgb"]))
            if success:
                counts[item["source_file"]] += 1
        records.append({"episode_index": episode_index, "success": bool(success), "source_file": item["source_file"],
                        "trajectory_id": item["trajectory_id"], "xy_error_m": xy, "z_delta_m": zd,
                        "grasp_xy_error_m": grasp_xy_error,
                        "cube_a_size_m": 0.02, "cube_b_size_m": 0.025,
                        "cube_a_initial_xyz": a.tolist(), "cube_b_initial_xyz": b.tolist(),
                        "cube_a_rotation_xyzw": item["cube_a_rotation"].tolist(), "cube_b_rotation_xyzw": item["cube_b_rotation"].tolist(),
                        "cube_a_resting_xyz": resting_a.tolist(), "cube_b_resting_xyz": resting_b.tolist(),
                        "cube_a_final_xyz": pa.tolist(), "cube_b_final_xyz": pb.tolist(),
                        "cube_a_speed_mps": speed_a, "cube_b_speed_mps": speed_b,
                        "settled_position_drift_m": position_drift.tolist(),
                        "lifted": bool(lifted), "stable_samples": sum(stable), "ik_failures": ik_failures,
                        "cube_history": cube_history,
                        "demo_grasp_modality": "eefpose+handdof; state/action shape (8,), two RGB cameras",
                        "joint_audit": {"names": dof_names[:7], "state_saved": True, "target_saved": True},
                        "boxes": {key: {"position": value["position"].tolist(),
                                        "rotation": value["rotation"].tolist()}
                                  for key, value in item["boxes"].items()}})
        print(f"candidate {candidate_index + 1}/{len(episodes)} success={success} xy={xy:.5f} dz={zd:.5f} counts={dict(counts)}", flush=True)
        manifest["success_counts"] = dict(counts)
        manifest["complete"] = all(counts[f] == args.per_scene for f in scene_files)
        pending_manifest = out / "replay_manifest.pending.json"
        pending_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pending_manifest.replace(out / "replay_manifest.json")
    if policy_socket is not None:
        policy_socket.close()


def main() -> None:
    args = build_parser().parse_args()
    if (args.eval_manifest is None) != (args.eval_index is None):
        raise ValueError("--eval-manifest and --eval-index must be provided together")
    if not args.usd.is_file() or not args.urdf.is_file():
        raise FileNotFoundError(f"missing USD/URDF: {args.usd}, {args.urdf}")
    if args.physics_steps_per_frame <= 0 or 120 % args.physics_steps_per_frame:
        raise ValueError("Physics steps per frame must be a positive divisor of 120")
    if args.static_friction < 0 or args.dynamic_friction < 0 or args.restitution < 0:
        raise ValueError("Contact material parameters must be non-negative")
    if args.static_friction < args.dynamic_friction:
        raise ValueError("Static friction must be greater than or equal to dynamic friction")
    trajectory_pairs = None
    if args.eval_manifest is not None:
        scene_rows = [json.loads(line) for line in args.eval_manifest.read_text().splitlines() if line.strip()]
        if not 0 <= args.eval_index < len(scene_rows):
            raise ValueError(f"eval index {args.eval_index} outside 0..{len(scene_rows)-1}")
        scene = scene_rows[args.eval_index]
        args.scene_ids = [int(scene["source_file"].split("_")[1])]
        args.scene_count = args.per_scene = 1
        trajectory_pairs = {(args.scene_ids[0], int(scene["trajectory_id"]))}
    if args.trajectories:
        try:
            trajectory_pairs = {tuple(map(int, value.split(":"))) for value in args.trajectories}
        except ValueError as exc:
            raise ValueError("--trajectories entries must be SCENE:TRAJECTORY pairs") from exc
        if any(len(pair) != 2 for pair in trajectory_pairs):
            raise ValueError("--trajectories entries must be SCENE:TRAJECTORY pairs")
    episodes = load_episodes(args.raw_root.resolve(), args.max_episodes, args.scene_count, args.per_scene,
                             args.scene_ids, args.candidate_multiplier, trajectory_pairs)
    if trajectory_pairs is not None:
        found = {(int(item["source_file"].split("_")[1]), item["trajectory_id"]) for item in episodes}
        missing = trajectory_pairs - found
        if missing:
            raise ValueError(f"Requested source trajectories are missing or unsuccessful: {sorted(missing)}")
    if not episodes:
        raise ValueError("No source trajectories selected")
    if args.dry_run:
        dry_run(args, episodes); return
    run_isaac(args, episodes)


if __name__ == "__main__":
    main()
