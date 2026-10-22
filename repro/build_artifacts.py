"""Maintainer: export exactly the data/assets used by the published reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-only', action='store_true', help='Append verified preprocessing caches to an existing export')
    args = parser.parse_args()
    source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=args.cache_only)
    evaluation = rows(source / 'data/evaluation_manifests/fr3v2_existing_holdouts_20261021/complete_regression_scenes.jsonl')
    train_base = source / 'data/training_manifests/fr3v2_cube40_cube60_20261021'
    training = rows(train_base / 'train_4hz.jsonl') + rows(train_base / 'validation_4hz.jsonl')
    training_uids={row['uid'] for row in rows(train_base/'train_4hz.jsonl')}
    sets = {name: set() for name in ['runtime', 'evaluation', 'training', 'cache']}
    for package in {row['package_root'] for row in evaluation + training}:
        for folder in ['source/assets', 'source/RobotDescriptions']:
            sets['runtime'].update(p.relative_to(source).as_posix()
                                   for p in (source/package/folder).rglob('*') if p.is_file())
        for name in ['source/scene.usda', 'meta/source_hashes.json',
                     'meta/collection_config.json', 'meta/scenes.jsonl']:
            sets['runtime'].add(f'{package}/{name}')
    for group, entries in [('evaluation', evaluation), ('training', training)]:
        for row in entries:
            for key in ['npz', 'diagnostics', 'scene_metadata']:
                if row.get(key):sets[group].add(row[key])
    # One recorded policy rollout enables deterministic offline action reproduction.
    sets['evaluation'].add('lerobot_policy/outputs/existing398_20261021/ensemble_full_remaining/index_005_seed_00/simulation/rollout.npz')
    sets['training'] -= sets['evaluation']
    for module, folder in [('delivery_policy_contract.py','delivery_pointcloud_v2'),
                           ('rgbd_object_features.py','delivery_rgbd_goal_memory_v1')]:
        code_hash=digest(source/'lerobot_policy'/module)[:12]
        for row in training:
            relative=f"data/derived/{folder}/{row['source_npz_sha256']}_{code_hash}.npz"
            if not (source/relative).is_file():
                if row['uid'] in training_uids:raise FileNotFoundError(relative)
                continue  # Unused validation episodes can be preprocessed on demand.
            sets['cache'].add(relative)
    manifest = dict(format_version=1,
        release_base='https://github.com/forriver2005/VR_training/releases/download/demograsp-best-20261022-v1',
        source_training_episodes=398, source_validation_episodes=100,
        evaluation_configurations=24, evaluation_cases=138,
        sets={})
    if args.cache_only:
        manifest=json.loads(args.manifest.read_text());sets={'cache':sets['cache']}
    for group, paths in sets.items():
        files = []
        for relative in sorted(paths):
            path = source / relative
            if path.is_symlink():raise ValueError('Symlinks are not supported: '+relative)
            files.append(dict(path=relative, size=path.stat().st_size, sha256=digest(path)))
        chunks = [[]]
        current_size = 0
        for item in files:
            if chunks[-1] and current_size + item['size'] > 900 * 1024**2:
                chunks.append([]);current_size = 0
            chunks[-1].append(item);current_size += item['size']
        assets = []
        for index, chunk in enumerate(chunks):
            name = f'demograsp-{group}-{index:03d}.tar.gz'
            target = args.output / name
            with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
                for item in chunk:
                    archive.add(source / item['path'], arcname=item['path'], recursive=False)
            assets.append(dict(name=name, size=target.stat().st_size, sha256=digest(target), files=chunk))
            print(json.dumps(dict(group=group, asset=name, bytes=target.stat().st_size, files=len(chunk))), flush=True)
        manifest['sets'][group] = dict(assets=assets, total_files=len(files),
                                      unpacked_bytes=sum(f['size'] for f in files))
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print('MANIFEST', args.manifest, flush=True)


if __name__ == '__main__':
    main()
