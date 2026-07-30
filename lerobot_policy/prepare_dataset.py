#!/usr/bin/env python3
"""Convert a subset of a DemoGrasp LeRobot v2.0 dataset to LeRobot v3.0."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

av.logging.set_level(av.logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="LeRobot v2.0 dataset root")
    parser.add_argument("--output", type=Path, required=True, help="New LeRobot v3.0 dataset root")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--vcodec", default="h264", help="FFmpeg encoder used for v3 videos")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def video_frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def source_episode_path(
    source: Path,
    template: str,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    return source / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.num_episodes <= 0 or args.start_episode < 0:
        raise ValueError("--num-episodes must be positive and --start-episode must be non-negative")

    info = load_json(source / "meta/info.json")
    if info.get("codebase_version") != "v2.0":
        raise ValueError(f"Expected a v2.0 source dataset, got {info.get('codebase_version')!r}")

    episode_records = {row["episode_index"]: row for row in load_jsonl(source / "meta/episodes.jsonl")}
    selected = list(range(args.start_episode, args.start_episode + args.num_episodes))
    missing = [index for index in selected if index not in episode_records]
    if missing:
        raise ValueError(f"Source dataset is missing requested episodes: {missing[:10]}")

    source_scenes_path = source / "meta/scenes.jsonl"
    source_scenes = None
    if source_scenes_path.is_file():
        source_scenes = {
            row["episode_index"]: row for row in load_jsonl(source_scenes_path)
        }
        missing_scenes = [index for index in selected if index not in source_scenes]
        if missing_scenes:
            raise ValueError(
                f"Source dataset is missing scene metadata for episodes: {missing_scenes[:10]}"
            )

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    source_features = info["features"]
    feature_names = ["observation.state", "action"] + [
        key for key, value in source_features.items() if value["dtype"] in {"image", "video"}
    ]
    features = {key: copy.deepcopy(source_features[key]) for key in feature_names}
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    camera_keys = [key for key in feature_names if features[key]["dtype"] in {"image", "video"}]
    fps = int(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=fps,
        robot_type=info.get("robot_type"),
        features=features,
        use_videos=True,
        vcodec=args.vcodec,
        streaming_encoding=True,
    )

    try:
        for new_episode_index, source_episode_index in enumerate(selected):
            parquet_path = source_episode_path(
                source, info["data_path"], source_episode_index, chunks_size
            )
            rows = pq.read_table(parquet_path, columns=["observation.state", "action"]).to_pylist()
            decoded = {}
            for camera_key in camera_keys:
                path = source_episode_path(
                    source,
                    info["video_path"],
                    source_episode_index,
                    chunks_size,
                    video_key=camera_key,
                )
                decoded[camera_key] = video_frames(path)

            expected_length = episode_records[source_episode_index]["length"]
            lengths = {"parquet": len(rows), **{key: len(value) for key, value in decoded.items()}}
            if any(length != expected_length for length in lengths.values()):
                raise ValueError(
                    f"Episode {source_episode_index} length mismatch: expected {expected_length}, got {lengths}"
                )

            task = episode_records[source_episode_index]["tasks"][0]
            for frame_index, row in enumerate(rows):
                frame = {
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "task": task,
                }
                frame.update({key: decoded[key][frame_index] for key in camera_keys})
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
            print(
                f"[{new_episode_index + 1:03d}/{len(selected):03d}] "
                f"converted source episode {source_episode_index}"
            )
    finally:
        dataset.finalize()

    if source_scenes is not None:
        converted_scenes = []
        for new_episode_index, source_episode_index in enumerate(selected):
            scene = copy.deepcopy(source_scenes[source_episode_index])
            scene["source_episode_index"] = source_episode_index
            scene["episode_index"] = new_episode_index
            converted_scenes.append(scene)
        write_jsonl(output / "meta/scenes.jsonl", converted_scenes)

        collection_config_path = source / "meta/collection_config.json"
        if collection_config_path.is_file():
            shutil.copy2(collection_config_path, output / "meta/collection_config.json")

    converted_info = load_json(output / "meta/info.json")
    if converted_info["total_episodes"] != len(selected):
        raise RuntimeError("Converted dataset episode count is inconsistent")
    print(
        f"Created {output}: {converted_info['total_episodes']} episodes, "
        f"{converted_info['total_frames']} frames, format {converted_info['codebase_version']}"
    )


if __name__ == "__main__":
    main()
