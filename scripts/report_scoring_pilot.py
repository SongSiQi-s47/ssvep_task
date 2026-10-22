import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
run=root/'runs/v4-scoring-pilot-cpu'
assert (run/'completed.json').exists()
r=json.loads((run/'results.json').read_text())
d=json.loads((run/'frozen_screen_decision.json').read_text())
lines=['# 第四阶段：固定 1.5 秒算法详细比较','',
       '日期：2026-10-17。全部四次 CPU 训练已完成；所有结果属于 BETA 筛选开发集。','',
       '## 数据与公平性','',
       '训练：S2/S5/S10/S15/S18/S25，192 个原始试次，先切 375 点窗口再预处理，步长 125 点，共 320 个训练窗口。开发：S6/S20/S51，各 32 段，共 96 段，只取首个窗口。原始试次先按受试者隔离，训练与开发不交叉。',
       '窗长采用前一步冻结的 1.5 秒；BETA 去除预刺激及 0.14 秒延迟。不读取 D1/D2、五名内部留出或 S13/S23 确认受试者。文件 SHA-256 与既有来源清单比对通过。','',
       '两个训练模型均采用种子 2026/2027，每 epoch 同样抽取 768 个窗口、batch 64，共 50 epoch/600 次更新；AdamW 学习率 0.001、权重衰减 0.001，梯度裁剪 5。两模型使用完全相同的抽样索引，不作数据增强；EEGNet 保留结构内 dropout。不同架构不声称初始权重相同或计算预算相同。选最早达到最高开发宏平均准确率的 checkpoint。',
       '共享评分器为三个 FBCCA 子带分数加三个相对谐波能量，各候选频率共用 6 个线性权重。特征标准化只在训练窗口拟合。FBCCA 无训练参数；学习评分器仍需执行 FBCCA 特征提取，不能把 6 个权重等同于低计算成本。','',
       '## 实际结果','',
       '| 算法/种子 | 宏平均准确率 | S6 | S20 | S51 | 参数 | 最佳 epoch | CPU 计算中位/p95 ms |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for k,v in r.items():
    t=v['cpu_compute_ms']
    lines.append(f"| {k} | {v['macro_accuracy']*100:.2f}% | "+' | '.join(f"{v['subjects'][s]*100:.2f}%" for s in ['S6','S20','S51'])+f" | {v.get('parameters',0)} | {v.get('selected_epoch','—')} | {t['median']:.2f}/{t['p95']:.2f} |")
lines+=['','## 种子汇总与晋级','']
for arm in ['eegnet','shared_scorer']:
    a=[r[f'{arm}_{s}']['macro_accuracy']*100 for s in [2026,2027]]
    lines.append(f"- {arm}：两种子均值 {np.mean(a):.2f}%，范围 {min(a):.2f}%–{max(a):.2f}%；训练耗时分别为 "+'/'.join(f"{r[f'{arm}_{s}']['training_seconds']:.1f}" for s in [2026,2027])+' 秒（不含公共特征缓存构建）。')
lines+=['',f"共享评分器相对每个种子中 FBCCA/EEGNet 的较强者：{d['gains_pp'][0]:+.2f}、{d['gains_pp'][1]:+.2f} 个百分点，平均 {d['mean_gain_pp']:+.2f}。",
        '预注册门槛：两种子均不退步且平均至少 +3 个百分点。结果：'+('通过筛选，尚需扩大训练与独立确认，不能直接替换模型。' if d['eligible'] else '未通过；停止扩大此评分方案，不降低门槛，不启动后续确认。'),'',
        '## 错误互补性诊断','',
        '下表只用开发真值诊断，两者任一正确的比例是不可部署的理想上界，不是融合模型准确率。', '',
        '| 种子 | FBCCA 正确/EEGNet 错误 | EEGNet 正确/FBCCA 错误 | 两者均错 | 任一正确上界 |','|---|---:|---:|---:|---:|']
manifest=json.loads((run/'window_manifest.json').read_text())['dev']
# Original class labels are encoded in saved per-trial data sequence: beta loader orders 8 classes x4 blocks per subject.
y=np.tile(np.repeat(np.arange(8),4),3)
assert len(manifest)==len(y)
for v in r.values():
    cm=np.zeros((8,8),dtype=int)
    np.add.at(cm,(y,np.array(v['prediction'])),1)
    np.testing.assert_array_equal(cm,v['confusion'])
for seed in [2026,2027]:
    a=np.array(r['fbcca']['prediction'])==y;b=np.array(r[f'eegnet_{seed}']['prediction'])==y
    lines.append(f'| {seed} | {int((a&~b).sum())} | {int((b&~a).sum())} | {int((~a&~b).sum())} | {(a|b).mean()*100:.2f}% |')
lines+=['','## 耗时与验证限制','',
        '时延均为同一 CPU、单数值计算线程、batch=1，预热后交错执行 3 遍×96 试次，各设置 288 次。包括窗口预处理及各自推理/特征提取，不含磁盘读取、训练或信号等待；本次批量与单样本输出已检查一致。BETA 需观测到刺激后 1.64 秒才能得到当前窗口；本实验为窗口内离线零相位处理，不是流式因果验证。',
        '最初 CUDA 运行在第一次优化器更新前因 cuDNN 初始化错误中止，失败目录保留；随后两种子全部 CPU 从头重训。GPU 自动检查也遇到初始化错误。CPU 环境测试共 20 项，其中 19 项通过、1 项 GPU 测试跳过，不能声称 GPU 验证通过。',
        '开发集已用于窗长选择与多轮模型选择，存在适应性暴露；三个受试者与两个种子不足以确认泛化或统计显著性。训练窗口不是独立受试者样本。所有候选分数均应按同窗长同预处理比较，不与历史不同协议的分数相减。','',
        '共享评分器采用随机初始化及固定 600 步预算；未以 FBCCA 权重暖启动，未做凸优化收敛验证。阴性结果只说明当前预注册训练方式未通过，不证明线性评分的最优解必然较差。后续如研究收敛问题，应另立实验，而非延长本轮训练直到开发分数过线。','',
        '## 文件与状态','',
        '完整运行：runs/v4-scoring-pilot-cpu；包含运行前配置、数据指纹、窗口索引、抽样序列、特征标准化统计、每 epoch 曲线、权重、逐试次分数、混淆矩阵、耗时与冻结筛选决定。',
        '本轮不替换正式部署算法或导出学生。失败的 GPU 尝试位于 runs/v4-scoring-pilot，仅供故障记录，不作为结果。']
text='\n'.join(lines)+'\n'
(root/'docs/scoring_pilot_v4_results.md').write_text(text,encoding='utf-8')
output=Path('C:/Users/64443/Documents/Codex/2026-10-14/siunners-chatgpt-conversation-6ace5c53-2184-83e8/outputs/第四阶段算法详细比较.md')
output.write_text(text,encoding='utf-8')
print(json.dumps({'eligible':d['eligible'],'scores':{k:v['macro_accuracy'] for k,v in r.items()}},indent=2))
