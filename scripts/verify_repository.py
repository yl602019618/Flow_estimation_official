"""检查归档完整性与 Markdown 本地链接；不启动声学、CFD 或反演。"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-hashes', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = []
    count = 0
    for doc in root.rglob('*.md'):
        if any(part in {'.git', '.venv', 'outputs'} for part in doc.relative_to(root).parts):
            continue
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            target = target.split('#')[0]
            if not (doc.parent/target).exists():
                errors.append(f'{doc.relative_to(root)}: missing link {target}')
    for name in ('data_manifest.json', 'results_manifest.json'):
        for entry in json.loads((root/'docs'/name).read_text()):
            path = root/entry['path']
            count += 1
            if not path.is_file():
                errors.append(f'missing {entry["path"]}')
                continue
            if path.stat().st_size != entry['bytes']:
                errors.append(f'size mismatch {entry["path"]}')
            if not args.skip_hashes:
                with path.open('rb') as stream:
                    hasher = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(block)
                    digest = hasher.hexdigest()
                if digest != entry['sha256']:
                    errors.append(f'hash mismatch {entry["path"]}')
    print(json.dumps({'checked_artifacts': count, 'hashes_checked': not args.skip_hashes,
                      'errors': errors}, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
