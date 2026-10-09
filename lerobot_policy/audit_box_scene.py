"""Inspect actual Gym box transforms, collision shapes, and full-scene render."""
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from isaacgym import gymapi
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lerobot_policy.isaacgym_v3_closed_loop import (
    SCENE_OFFSET, configure, create_env, scene_rows, set_scene,
)


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg):
    configure(cfg)
    row = scene_rows(Path(str(cfg.v3_manifest)), None)[int(cfg.get("v3_start", 0))]
    output = Path(str(cfg.v3_output))
    output.mkdir(parents=True, exist_ok=True)
    env = create_env(cfg)
    set_scene(env, row)
    props = gymapi.CameraProperties()
    props.width, props.height = 1280, 960
    camera = env.gym.create_camera_sensor(env.envs[0], props)
    env.gym.set_camera_location(camera, env.envs[0], gymapi.Vec3(1.5, -1.6, 1.7),
                                gymapi.Vec3(0.3, 0, 0.1))
    for _ in range(12):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.write_camera_image_to_file(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR,
                                       str(output / "full_scene.png"))
    report = []
    for index, actor_index in enumerate(env.fixed_box_indices[0]):
        handle = env.gym.find_actor_handle(env.envs[0], f"fixed_box_{index}")
        actual = env.root_state_tensor[actor_index, :3].detach().cpu().numpy().astype(float)
        expected = np.asarray(row['boxes'][f'box{index + 1}']['position']) + SCENE_OFFSET
        report.append(dict(expected=expected.tolist(), actual=actual.tolist(),
                           root=env.root_state_tensor[actor_index, :7].cpu().tolist(),
                           collision_shapes=env.gym.get_actor_rigid_shape_count(env.envs[0], handle),
                           error_m=float(np.linalg.norm(expected - actual))))
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    os._exit(0 if all(r['error_m'] < 1e-5 and r['collision_shapes'] == 5 for r in report) else 1)


if __name__ == '__main__':
    main()
