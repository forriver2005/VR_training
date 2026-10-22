"""Download and safely unpack the exact, SHA256-verified reproduction resources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):h.update(block)
    return h.hexdigest()


def destination(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or name.startswith('/'):
        raise ValueError('Unsafe archive path: '+name)
    return path


def valid(path, item):
    return path.is_file() and path.stat().st_size == item['size'] and digest(path) == item['sha256']


def download(url, target, item):
    if target.exists():
        if valid(target, item):return
        raise ValueError('Corrupt cached archive; remove and retry: '+str(target))
    partial = target.with_suffix(target.suffix+'.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'User-Agent': 'DemoGrasp-reproduction/1',
                                                 **({'Range': f'bytes={offset}-'} if offset else {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        with partial.open('ab' if append else 'wb') as stream:
            shutil.copyfileobj(response, stream, 8 << 20)
    if not valid(partial, item):raise ValueError('Downloaded archive checksum mismatch: '+str(partial))
    partial.replace(target)


def unpack(archive_path, files):
    expected = {item['path']: item for item in files}
    observed = set()
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in observed:
                raise ValueError('Unexpected archive member: '+member.name)
            item = expected[member.name];observed.add(member.name)
            if member.size != item['size']:raise ValueError('Member size mismatch: '+member.name)
            target = destination(member.name)
            if target.exists():
                if valid(target, item):continue
                raise FileExistsError('Refusing to overwrite different local data: '+str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.repro-part')
            with archive.extractfile(member) as source, temporary.open('wb') as stream:
                shutil.copyfileobj(source, stream, 8 << 20)
            if not valid(temporary, item):raise ValueError('Extracted file checksum mismatch: '+member.name)
            temporary.replace(target)
    if observed != set(expected):raise ValueError('Missing archive members')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', nargs='+', choices=['runtime', 'evaluation', 'training', 'cache'],
                        default=['runtime', 'evaluation'])
    parser.add_argument('--local-source', type=Path, help='Use already downloaded archives')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'repro/artifacts.json').read_text())
    cache = ROOT / '.repro_cache';cache.mkdir(exist_ok=True)
    for group in dict.fromkeys(args.sets):
        for asset in manifest['sets'][group]['assets']:
            if all(valid(destination(f['path']), f) for f in asset['files']):
                print('VERIFIED', asset['name'], flush=True);continue
            if args.verify_only:raise ValueError('Missing or changed files from '+asset['name'])
            archive = (args.local_source or cache) / asset['name']
            print('FETCH', asset['name'], f"({asset['size']/1024**2:.1f} MiB)", flush=True)
            if not args.local_source:download(manifest['release_base']+'/'+asset['name'], archive, asset)
            if not valid(archive, asset):raise ValueError('Archive checksum mismatch: '+str(archive))
            unpack(archive, asset['files'])
            print('INSTALLED', asset['name'], flush=True)
    print('All requested resources verified.', flush=True)


if __name__ == '__main__':
    main()
