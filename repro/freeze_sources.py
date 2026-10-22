"""Maintainer utility: record the reviewed portable source/checkpoint inventory."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files=[]
    for folder in ['lerobot_policy','repro']:
        for path in (ROOT/folder).rglob('*'):
            relative=path.relative_to(ROOT)
            if not path.is_file() or any(part in relative.parts for part in
                                        ['outputs','__pycache__','exports','artifacts','stack_artifacts']):continue
            if relative.as_posix()=='repro/source_manifest.json':continue
            if path.suffix in ['.py','.md','.json','.txt','.safetensors']:files.append(path)
    manifest=dict(description='Reviewed portable source/checkpoint inventory; historical hashes remain in original report.',
                  sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    (ROOT/'repro/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Recorded',len(files),'files')


if __name__=='__main__':main()
