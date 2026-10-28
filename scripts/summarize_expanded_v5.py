"""Write the fixed-protocol v5 accuracy report from completed run records."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/v5-expanded-eegnet'
config = json.loads((RUN/'preregistered_config.json').read_text(encoding='utf-8'))
dev = json.loads((RUN/'development_results.json').read_text(encoding='utf-8'))
confirm = json.loads((RUN/'confirmation_results.json').read_text(encoding='utf-8'))
assert json.loads((RUN/'completed.json').read_text())['complete']
seeds = config['seeds']


def pct(x):
    return f'{100*x:.2f}%'


def row(data, name):
    fixed = data['fbcca']['subject_macro_accuracy']
    six = np.array([data[f'eegnet_six_{s}']['subject_macro_accuracy'] for s in seeds])
    fourteen = np.array([data[f'eegnet_fourteen_{s}']['subject_macro_accuracy'] for s in seeds])
    return f'| {name} | {pct(fixed)} | {pct(six.mean())} | {pct(fourteen.mean())} | {100*(fourteen-six).mean():+.2f} pp | {100*(fourteen.mean()-fixed):+.2f} pp |'


lines = [
    '# v5 扩大实验：固定 FBCCA 与 EEGNet',
    '',
    '运行状态：完成。六名与十四名 BETA 训练受试者分别训练同一 EEGNet 结构；每组种子 2026/2027/2028。',
    '两组均使用固定 1.5 秒首窗、窗口内预处理、每 epoch 768 次抽样、50 epoch 与相同优化器。',
    'BETA 开发受试者 S6/S20/S51 用于选 checkpoint；新增 BETA 五名受试者在所有模型冻结后一次性评分。',
    'D1/D2 只用于冻结后的历史赛方样例复核；不参与训练、归一化统计拟合、早停或模型选择。',
    '',
    '| 数据集 | 固定 FBCCA | EEGNet 六人均值 | EEGNet 十四人均值 | 十四人−六人 | 十四人−FBCCA |',
    '|---|---:|---:|---:|---:|---:|',
    row(dev, 'BETA 已暴露开发，3 人/96 试次'),
    row(confirm['new_beta_test'], 'BETA 新确认，5 人/160 试次'),
    row(confirm['D1_trials'], 'D1 历史样例，48 试次'),
    row(confirm['D2_trials'], 'D2 历史样例，48 试次'),
    '',
    '## 新 BETA 确认集逐受试者',
    '',
    '| 受试者 | FBCCA | EEGNet 六人三种子均值 | EEGNet 十四人三种子均值 | 十四人−六人 | 十四人−FBCCA |',
    '|---|---:|---:|---:|---:|---:|',
]
test = confirm['new_beta_test']
subjects = sorted(test['fbcca']['subjects'])
differences = []
for subject in subjects:
    fixed = test['fbcca']['subjects'][subject]
    six = np.array([test[f'eegnet_six_{s}']['subjects'][subject] for s in seeds])
    fourteen = np.array([test[f'eegnet_fourteen_{s}']['subjects'][subject] for s in seeds])
    differences.append([float((fourteen-six).mean()), float(fourteen.mean()-fixed)])
    lines.append(f'| {subject} | {pct(fixed)} | {pct(six.mean())} | {pct(fourteen.mean())} | {100*(fourteen-six).mean():+.2f} pp | {100*(fourteen.mean()-fixed):+.2f} pp |')

differences = np.asarray(differences)
rng = np.random.default_rng(20261023)
indices = rng.integers(len(subjects), size=(20000, len(subjects)))
boot = differences[indices].mean(axis=1)
intervals = np.quantile(boot, [0.025, 0.975], axis=0) * 100
lines += [
    '',
    '## 配对变化与解释',
    '',
    f'- 新 BETA 受试者宏平均：十四人训练相对六人训练 {100*differences[:,0].mean():+.2f} pp；按受试者重抽样的描述性 95% 区间 [{intervals[0,0]:+.2f}, {intervals[1,0]:+.2f}] pp。',
    f'- 新 BETA 受试者宏平均：十四人 EEGNet 相对固定 FBCCA {100*differences[:,1].mean():+.2f} pp；相同方法的描述性 95% 区间 [{intervals[0,1]:+.2f}, {intervals[1,1]:+.2f}] pp。',
    f'- 逐受试者方向：十四人 EEGNet 比六人 EEGNet 更好 {int((differences[:,0]>0).sum())}/{len(subjects)} 人；比固定 FBCCA 更好 {int((differences[:,1]>0).sum())}/{len(subjects)} 人。',
    '- 上述区间仅对这五名受试者做重抽样，不能覆盖数据集差异、反复开发选择和正式赛题的不确定性；三种子是训练重复，不当作独立受试者。',
    '- D1/D2 已有历史暴露且来源受试者身份不确定。它们只能作为迁移风险提示，不能据此再调窗口、挑种子或宣布盲测提升。',
    '- 正式赛题格式、通道排列和刺激频率映射仍应以届时元信息核对；当前不支持外推到任意正式赛题。',
    '',
    '可复核文件：`configs/expanded_v5.json`、`ssvep/expanded_v5.py`、`runs/v5-expanded-eegnet/`。',
]
out = ROOT / 'docs/expanded_v5_results.md'
out.write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(out)
