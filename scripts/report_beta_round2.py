"""Summarize completed BETA-only experiments without changing model selection."""
from pathlib import Path
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(r'C:\Users\64443\Documents\Codex\2026-10-14\siunners-chatgpt-conversation-6ace5c53-2184-83e8\outputs')


def main():
    base=ROOT/'runs/v2-beta-subject-isolated';dist=ROOT/'runs/v2-beta-algorithm-distilled'
    for run in [base,dist]:
        if not json.loads((run/'completed.json').read_text())['complete']:raise ValueError('Incomplete experiment')
    a=json.loads((base/'results.json').read_text());b=json.loads((dist/'results.json').read_text())
    roles=json.loads((base/'subject_roles.json').read_text())
    seeds=['2026','2027','2028']
    sources={'频率共享 CNN 教师':[a[s]['frequency_teacher'] for s in seeds],
             'EEGNet 直接训练学生':[a[s]['student'] for s in seeds],
             'FBCCA 教师蒸馏学生':[b[s] for s in seeds]}
    def stat(rows,field):
        values=np.array([r[field]['accuracy'] for r in rows])*100
        return f'{values.mean():.2f}% ± {values.std():.2f}'
    lines=['# 第二轮 BETA 实验报告','',
           '用户确认协议：赛方 D1/D2 仅用于验证；公开 BETA 可以训练，按受试者隔离。以下均为实际完成结果。','',
           '## 数据与比较范围','',
           '- 24 名已校验 BETA：14 人训练、5 人开发验证、5 人内部测试；固定人员划分，3 个初始化种子。',
           '- 原始试次数：训练 448、开发 160、内部测试 160；训练窗口在划分后生成。',
           '- D1/D2 各 48 段，共 96 段，仅在全部模型冻结后评分，不参加训练、蒸馏、校准、早停和模型选择。',
           '- 固定 1 秒、250 Hz、六通道；本报告不能与先前训练使用过 D1 的 v1 数字直接作改进对照。',
           '- D1/D2 曾被历史实验查看，因此这里称历史样例验证，不称全新盲测。','',
           '## 三种子结果','',
           '均值 ± 标准差为三个初始化结果的描述性统计（总体标准差，ddof=0），不是置信区间。FBCCA 为无需训练的固定结果。','',
           '| 算法 | BETA 内部测试，160 段 | D1/D2 验证，96 段 |',
           '|---|---:|---:|']
    for name,rows in sources.items():lines.append(f'| {name} | {stat(rows,"public_holdout")} | {stat(rows,"reserved_evaluation")} |')
    lines.append(f'| FBCCA 基线 | {a["fbcca"]["public_holdout"]["accuracy"]*100:.2f}% | {a["fbcca"]["reserved_evaluation"]["accuracy"]*100:.2f}% |')
    lines+=['','## 蒸馏的实际收益','']
    for field,label in [('public_holdout','BETA 内部测试'),('reserved_evaluation','D1/D2 验证')]:
        gain=np.mean([b[s][field]['accuracy']-a[s]['student'][field]['accuracy'] for s in seeds])*100
        lines.append(f'- {label}中，FBCCA 教师蒸馏相对直接训练学生，平均变化 {gain:+.2f} 个百分点。')
    lines+=['- CNN 教师在三个 BETA 开发集结果中均弱于学生，按预设门槛跳过神经教师蒸馏。',
            '- FBCCA 开发准确率为 63.75%，高于三个直接训练学生。它的原始分数以开发集负对数似然选择全局倍率 8；不是把算法分数直接称作已校准概率。',
            '- 蒸馏温度 T=2、软目标权重 α=0.25；软目标仅来自 BETA 训练窗口。',
            '- 蒸馏有小幅平均改善，尚未超过 D1/D2 上的 FBCCA。三次种子不足以认定统计显著性。','',
            '## 每种子的直接对照','',
            '| 种子 | 直接学生 D1 | 蒸馏学生 D1 | 直接学生 D2 | 蒸馏学生 D2 |',
            '|---|---:|---:|---:|---:|']
    for s in seeds:
        lines.append('| '+s+' | '+' | '.join(f'{v*100:.2f}%' for v in [a[s]['student']['D1_trials']['accuracy'],b[s]['D1_trials']['accuracy'],a[s]['student']['D2_trials']['accuracy'],b[s]['D2_trials']['accuracy']])+' |')
    lines+=['','## 模型选择与交付','',
            '- 默认算法在各种子中均由 BETA 开发集选择，当前为 FBCCA。没有因 D1/D2 某个种子分数更高而改选。',
            '- 单独导出神经网络候选：蒸馏学生 seed=2026，依据它在 BETA 开发集的 61.875% 选择；这是神经网络候选中的选择，未超过默认 FBCCA 的开发分数。',
            '- 学生 6,484 个参数。导出的 TorchScript 约 70.5 KB，可用 CPU 加载；参数权重与原始模型已在批量 1 和 8 下比对一致。',
            '- GPU 模型计算中位时延约 1 毫秒，来自原始模型测试；不包含采样和预处理，也不是 CPU 导出模型的端到端时延。','',
            '导出文件 `SSVEP_v2_student_inference.pt`：输入 float32 `[batch,6,250]`，电极顺序 CP3/CPZ/CP4/PO3/POZ/PO4；输出八类 logits，argmax 对应 0–7。该文件为 CPU 推理导出；GPU 使用原始模型与训练权重。','',
            '输入必须先按项目流程处理：完整原始试次做四阶 6–90 Hz 离线零相位带通，每通道去均值并除以该试次标准差，再取首个一秒窗口。不可把未处理的 CSV 直接传入。数据单位和正式格式仍须按赛方资料核对。','',
            '## 人员划分','']
    for role,subjects in roles.items():lines.append(f'- {role}: '+', '.join(subjects))
    lines+=['','## 可复核记录','',
            f'- 主实验：`{base}`。',f'- 算法教师蒸馏：`{dist}`。',
            '- 各运行保存来源 SHA-256、受试者/试次清单、开发曲线、最优权重、逐试次预测及部署选择依据。',
            '- 数据用途与受试者隔离等共 10 项测试通过；CPU 保存/重载结果与模型输出比对通过。',
            '- 澄清前的独立 Benchmark 先导实验已经完成并保留，不混入本报告数字。','',
            '下一轮建议在公开数据内部固定新的研究问题：短窗长度、滤波方式、公开数据规模和跨来源稳健特征。D1/D2 继续只作阶段性验证，不据其分数逐轮搜索参数。']
    text='\n'.join(lines)+'\n'
    (ROOT/'docs/round2_beta_results.md').write_text(text,encoding='utf-8')
    (OUT/'第二轮BETA实验报告.md').write_text(text,encoding='utf-8')
    shutil.copy2(ROOT/'docs/round2_beta_protocol.md',OUT/'第二轮实验方案.md')
    print('Report saved:',OUT/'第二轮BETA实验报告.md')


if __name__=='__main__':main()
