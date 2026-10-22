"""Fetch original data only. Resume partial files; verify published checksums."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import requests
from scipy.io import whosmat

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
STATE = {}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def status(key, **values):
    with LOCK:
        STATE.setdefault(key, {}).update(values)
        save_json(ROOT / 'logs/download_status.json', STATE)


def fetch(item):
    dataset, name = item['dataset'], item['name']
    key = dataset + '/' + name
    target = ROOT / 'data/raw/public' / dataset / name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    status(key, status='downloading', expected_bytes=item.get('size'), url=item['url'])
    for attempt in range(6):
        try:
            if not target.exists():
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                url = item['url']
                if attempt % 2 and dataset == 'beta':
                    url = 'https://huggingface.co/datasets/Bingchuan/BETA/resolve/main/' + name
                with requests.get(url, headers=headers, stream=True, timeout=(20, 45)) as response:
                    if response.status_code == 416 and item.get('size') == offset:
                        partial.replace(target)
                    else:
                        response.raise_for_status()
                        append = response.status_code == 206 and offset > 0
                        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Unexpected Content-Range')
                        total = offset if append else 0
                        last = time.monotonic()
                        with partial.open('ab' if append else 'wb') as stream:
                            for block in response.iter_content(1024 * 1024):
                                if block:
                                    stream.write(block)
                                    total += len(block)
                                if time.monotonic() - last >= 5:
                                    status(key, downloaded_bytes=total)
                                    last = time.monotonic()
                        if item.get('size') is not None and partial.stat().st_size != item['size']:
                            raise ValueError('File size differs from published metadata')
                        partial.replace(target)
            if item.get('size') is not None and target.stat().st_size != item['size']:
                target.replace(partial)
                raise ValueError('Existing file has incorrect size; retained as partial')
            md5, sha256 = digest(target)
            if item.get('md5') and md5 != item['md5']:
                target.replace(target.with_suffix(target.suffix + '.invalid'))
                raise ValueError('Published MD5 mismatch; retained invalid file')
            variables = whosmat(target) if target.suffix == '.mat' else None
            if variables is not None and not variables:
                raise ValueError('MAT file contains no variables')
            if dataset == 'nakanishi2015' and variables is not None:
                if not any(n == 'eeg' and tuple(shape) == (12, 8, 1114, 15)
                           for n, shape, dtype in variables):
                    raise ValueError('Unexpected Nakanishi EEG tensor')
            status(key, status='verified', downloaded_bytes=target.stat().st_size,
                   md5=md5, sha256=sha256, variables=variables, error=None)
            print('VERIFIED', key, target.stat().st_size, flush=True)
            return True
        except Exception as exc:
            status(key, status='retrying', error=str(exc), attempt=attempt + 1)
            print('RETRY', key, str(exc), flush=True)
            time.sleep(min(2 ** attempt, 8))
    status(key, status='failed')
    return False


def manifest():
    source = ROOT / 'configs/beta_source.json'
    if source.exists():
        article = json.loads(source.read_text(encoding='utf-8'))
    else:
        response = requests.get('https://api.figshare.com/v2/articles/12264401', timeout=45)
        response.raise_for_status()
        article = response.json()
        save_json(source, article)
    files = [{'dataset': 'beta', 'name': f['name'], 'size': f['size'],
              'md5': f['computed_md5'], 'url': f['download_url']}
             for f in article['files']]
    files.sort(key=lambda f: (f['name'].endswith('.mat'),
                             int(f['name'][1:-4]) if f['name'].endswith('.mat') else 0))
    files += [{'dataset': 'nakanishi2015', 'name': 'README.md',
               'url': 'https://raw.githubusercontent.com/mnakanishi/12JFPM_SSVEP/master/README.md'}]
    files += [{'dataset': 'nakanishi2015', 'name': f's{i}.mat',
               'url': f'https://raw.githubusercontent.com/mnakanishi/12JFPM_SSVEP/master/data/s{i}.mat'}
              for i in range(1, 11)]
    save_json(ROOT / 'configs/download_manifest.json', files)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    files = manifest()
    previous = ROOT / 'logs/download_status.json'
    if previous.exists():
        STATE.update(json.loads(previous.read_text(encoding='utf-8')))
    required = sum(x.get('size', 20_000_000) for x in files
                   if not (ROOT / 'data/raw/public' / x['dataset'] / x['name']).exists())
    if shutil.disk_usage(ROOT).free < required + 3_000_000_000:
        raise RuntimeError('Insufficient space: retain at least 3 GB for project work')
    print('START', len(files), 'files;', required, 'estimated bytes', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fetch, files))
    summary = {'verified': sum(results), 'total': len(files), 'complete': all(results)}
    import verify_public
    summary['mat_structure_and_finite_values_passed'] = verify_public.main()
    summary['complete'] = summary['complete'] and summary['mat_structure_and_finite_values_passed']
    save_json(ROOT / 'logs/download_summary.json', summary)
    print('SUMMARY', summary, flush=True)
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
