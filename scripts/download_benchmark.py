"""Download a preregistered six-subject pilot from MOABB's Zenodo mirror."""
import concurrent.futures
import json
from pathlib import Path
import threading

import requests
import download_public as downloader

ROOT=Path(__file__).resolve().parents[1]
SUBJECTS=[1,7,14,21,28,35]


def main():
    path=ROOT/'configs/benchmark_source.json'
    if path.exists():record=json.loads(path.read_text(encoding='utf-8'))
    else:
        response=requests.get('https://zenodo.org/api/records/14865172',timeout=45)
        response.raise_for_status();record=response.json()
        downloader.save_json(path,record)
    wanted={f'S{i}.mat' for i in SUBJECTS}|{'Readme.txt','64-channels.loc','Sub_info.txt'}
    files=[{'dataset':'benchmark','name':f['key'],'url':f['links']['self'],
            'size':f['size'],'md5':f['checksum'].split(':')[1]}
           for f in record['files'] if f['key'] in wanted]
    if len(files)!=len(wanted):raise ValueError('Required pilot files absent from repository')
    files.sort(key=lambda x:(x['name'].endswith('.mat'),x['name']))
    downloader.save_json(ROOT/'configs/benchmark_download_manifest.json',files)
    status=ROOT/'logs/benchmark_download_status.json'
    if status.exists():downloader.STATE.update(json.loads(status.read_text()))
    def update(key,**values):
        with downloader.LOCK:
            downloader.STATE.setdefault(key,{}).update(values)
            downloader.save_json(status,downloader.STATE)
    downloader.status=update
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(downloader.fetch,files))
    downloader.save_json(ROOT/'logs/benchmark_download_summary.json',
                         {'verified':sum(results),'total':len(results),'complete':all(results),'subjects':SUBJECTS})
    print('BENCHMARK_DOWNLOAD',sum(results),len(results),flush=True)


if __name__=='__main__':main()
