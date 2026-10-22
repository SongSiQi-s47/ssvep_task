"""Resumable ranged download of the official CPU wheel; publisher SHA256 required."""
from pathlib import Path
import concurrent.futures
import hashlib
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
URL='https://download-r2.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-win_amd64.whl'
SHA='c52adc3bd526ff7fc5bdcc58fc4a76e6214d09951b646ffe5256ebd7ba60cef1'
WORK=ROOT/'work/torch-wheel';WORK.mkdir(parents=True,exist_ok=True)
OUT=WORK/'torch-2.14.1+cpu-cp312-cp312-win_amd64.whl'


def main():
    r=requests.get(URL,headers={'Range':'bytes=0-0'},timeout=45)
    r.raise_for_status()
    if r.status_code!=206:raise ValueError('Server does not support bounded ranges')
    total=int(r.headers['Content-Range'].split('/')[-1]);chunk=4*1024*1024
    def fetch(start):
        end=min(total-1,start+chunk-1);p=WORK/f'{start}.part'
        if p.exists() and p.stat().st_size==end-start+1:return p
        for attempt in range(10):
            try:
                offset=p.stat().st_size if p.exists() else 0
                if offset==end-start+1:return p
                position=start+offset
                # Offset query prevents a cached response for a different byte range.
                with requests.get(URL+f'?segment={position}',headers={'Range':f'bytes={position}-{end}'},
                                  stream=True,timeout=(20,35)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {position}-{end}/{total}':
                        raise ValueError('Incorrect ranged response')
                    with p.open('ab' if offset else 'wb') as f:
                        for block in response.iter_content(262144):f.write(block)
                if p.stat().st_size!=end-start+1:raise ValueError('Truncated segment')
                print('CHUNK',start,end,flush=True);return p
            except Exception as exc:
                print('RETRY',start,attempt,str(exc)[:120],flush=True);time.sleep(min(attempt+1,5))
        raise RuntimeError(f'Could not fetch segment {start}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts=list(pool.map(fetch,range(0,total,chunk)))
    sha=hashlib.sha256()
    with OUT.open('wb') as f:
        for p in parts:
            b=p.read_bytes();f.write(b);sha.update(b)
    if sha.hexdigest()!=SHA:raise ValueError('Official wheel checksum mismatch')
    print('VERIFIED_WHEEL',OUT,total,flush=True)


if __name__=='__main__':main()
