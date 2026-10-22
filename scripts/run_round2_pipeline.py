"""One background job: acquire independent training data, then run frozen protocol."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/experiment_v2.json').read_text())
RUN='v2-beta-subject-isolated' if CONFIG['training_source']=='beta' else 'v2-independent-benchmark'
DISTILL_RUN='v2-beta-algorithm-distilled'


def state(stage,**details):
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    (ROOT/'logs/round2.pipeline.json').write_text(json.dumps(
        {'stage':stage,'updated_at_Asia_Shanghai':now,'run':RUN,**details},ensure_ascii=False,indent=2),encoding='utf-8')
    print('PIPELINE_STAGE',stage,flush=True)


def main():
    try:
        if CONFIG['training_source']=='benchmark':
            state('downloading_independent_training_data')
            subprocess.run([sys.executable,'-u','scripts/download_benchmark.py'],cwd=ROOT,check=True)
            summary=json.loads((ROOT/'logs/benchmark_download_summary.json').read_text())
            if not summary['complete']:raise RuntimeError('Training data download incomplete; see checksum/download status')
        if not (ROOT/'runs'/RUN/'completed.json').exists():
            state('training_and_validation')
            subprocess.run([sys.executable,'-u','-m','ssvep.round2','--run',RUN],cwd=ROOT,check=True)
        if not (ROOT/'runs'/RUN/'completed.json').exists():raise RuntimeError('Experiment completion flag missing')
        if CONFIG['training_source']=='beta':
            if not (ROOT/'runs'/DISTILL_RUN/'completed.json').exists():
                state('algorithm_teacher_distillation')
                subprocess.run([sys.executable,'-u','-m','ssvep.beta_algorithm_distill','--base-run',RUN,'--run',DISTILL_RUN],cwd=ROOT,check=True)
            state('complete',results=str(ROOT/'runs'/DISTILL_RUN/'summary.json'));return
        state('complete',results=str(ROOT/'runs'/RUN/'summary.json'))
    except Exception as exc:
        state('failed',error=str(exc));raise


if __name__=='__main__':main()
