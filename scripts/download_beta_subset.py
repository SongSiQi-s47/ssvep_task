"""Download a declared subset of BETA subjects with the repository's checksum checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import download_public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subjects', nargs='+', help='BETA subject IDs, e.g. S2 S25')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    names = {f'{subject}.mat' for subject in args.subjects}
    manifest = json.loads((download_public.ROOT / 'configs/download_manifest.json').read_text())
    items = [item for item in manifest if item['dataset'] == 'beta' and item['name'] in names]
    if {item['name'] for item in items} != names:
        parser.error('Unknown BETA subject ID in request')
    state_path = download_public.ROOT / 'logs/download_status.json'
    if state_path.exists():
        download_public.STATE.update(json.loads(state_path.read_text(encoding='utf-8')))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        status = list(pool.map(download_public.fetch, items))
    failed = [item['name'] for item, ok in zip(items, status) if not ok]
    if failed:
        raise SystemExit(f'Download failed: {failed}')


if __name__ == '__main__':
    main()
