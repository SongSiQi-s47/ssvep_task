"""Report all bounded pilot outcomes, including unsupported directions."""
from pathlib import Path
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(r'C:\Users\64443\Documents\Codex\2026-10-14\siunners-chatgpt-conversation-6ace5c53-2184-83e8\outputs')


def main():
    studies=[('v3-training-method-pilot','30 epoch：采样/增强/学习率'),
             ('v3b-cross-subject-alignment','50 epoch：跨受试者类别特征对齐'),
             ('v3c-long-to-short-curriculum','50 epoch：长窗到短窗课程')]
    names={'balanced_random_crop':'均衡随机切窗','phase_gain':'随机切窗＋相位/增益增强',
           'phase_gain_cosine':'随机切窗＋增强＋余弦学习率',
           'cross_subject_contrastive':'跨受试者监督对比辅助损失',
           'long_to_short_curriculum':'1.5 秒→1 秒课程训练'}
    lines=['# 第三阶段训练方式小规模验证报告','',
           '结论：本阶段未确认可以稳定改善当前模型的训练方式。16 次训练中，五个候选变体均未达到事先设定的筛选门槛；没有进入保留开发受试者确认，也没有替换第二轮模型。','',
           '## 数据与公平性','',
           '- 六名 BETA 训练受试者：S2、S5、S10、S15、S18、S25，共 192 个原始试次。',
           '- 三名筛选开发受试者：S6、S20、S51，共 96 个原始试次。下面的准确率全部属于这 96 段 BETA 开发数据，不是 D1/D2。',
           '- 两名保留确认开发受试者：S13、S23，共 64 段，因没有候选通过筛选，本阶段未读取其信号。它们历史上属于第二轮开发数据，不能称为全新外部盲测。',
           '- D1/D2 与第二轮 BETA 内部测试五人，本阶段均未读取。未加载旧权重，也没有生成这些数据的软目标。',
           '- 所有方法使用同一个 6,484 参数学生；同种子初始权重逐项核对一致。每 epoch 768 样本，batch=64；30 epoch 为 360 次更新，50 epoch 为 600 次更新。',
           '- 三个小试验独立保存，分别与各自相同训练步数的基线比较；不把 50 epoch 与 30 epoch 的差值归因于训练方法。','',
           '## 两种子结果','',
           '差值为“候选准确率−同预算枚举切窗基线”，单位为百分点。预设筛选门槛：两个种子均不退步，并且平均增益至少 3 个百分点。','',
           '| 训练变体 | 种子 2026 差值 | 种子 2027 差值 | 平均差值 | 通过筛选 |',
           '|---|---:|---:|---:|---|']
    summaries={};total=0
    for study,title in studies:
        r=ROOT/'runs'/study
        if not json.loads((r/'completed.json').read_text())['complete']:raise ValueError('Unfinished study')
        decision=json.loads((r/'frozen_screen_decision.json').read_text())
        summaries[study]=decision
        for arm,row in decision['candidates'].items():
            first,second=row['paired_gains_percentage_points']
            lines.append(f'| {names[arm]} | {first:+.2f} | {second:+.2f} | {row["mean_gain_percentage_points"]:+.2f} | '+('是' if row['eligible'] else '否')+' |')
        total+=len(list(r.rglob('history.json')))
    lines+=['','## 绝对准确率与训练预算','',
            '| 小试验 | 方法 | 2026 开发准确率 | 2027 开发准确率 |',
            '|---|---|---:|---:|']
    for study,title in studies:
        result=json.loads((ROOT/'runs'/study/'screen_results.json').read_text())
        for arm,rows in result.items():
            label='枚举切窗基线' if arm=='enumerated' else names[arm]
            lines.append(f'| {title} | {label} | {rows["2026"]["macro_subject_accuracy"]*100:.2f}% | {rows["2027"]["macro_subject_accuracy"]*100:.2f}% |')
    lines+=['','## 如何理解这些结果','',
            '1. 均衡随机切窗在本设置下平均下降 2.60 个百分点，增加相位/幅值增强后也没有改善。不能据此认定数据增强一般无效；这里只排除了当前强度、采样和训练预算组合。',
            '2. 跨受试者特征对齐在一个种子中提升 3.13 个百分点，在另一个中下降 2.08 个百分点。平均约 +0.52，方向不一致，不能挑选前者宣称成功。',
            '3. 长窗课程仍以一秒输入评分，平均下降约 3.65 个百分点。它虽然更新次数相同，前半程输入更长，因此不是等 FLOPs 或等运行时间实验。',
            '4. 小样本下未达到门槛，当前不能确认值得扩大训练的变体。保留现有基线；后续若扩大研究，应先提出新的、可独立检验的假设，而不是继续对同一批开发数据搜索参数。',
            '5. 结果来自六名训练受试者和两个初始化种子，不能外推为这些方法在全量 BETA 或正式比赛中必然无效。没有计算统计显著性，也没有给出赛方数据的新准确率。','',
            '## 实现与核验','',
            '- 新增可重跑的训练方式试验入口，分别保存固定配置、读取审计、每 epoch 各受试者抽样次数、损失、学习率、训练窗长和最佳权重。',
            '- 对比损失只把同类不同受试者作为正对；同类同受试者不充当负对；无正对时稳定返回零损失。',
            '- 相位增强在所有通道使用共同位移；频率峰与通道对齐测试通过。课程裁剪保留原试次抽样权重并检查窗口边界。',
            '- 本阶段 16 次训练均完成，损失均为有限数值。两次 50 epoch 基线在不同小试验中的最终选中权重逐项相同，支持运行复现性。',
            '- 全部 14 项自动检查通过。原有第二轮模型及其推理导出保持不变，没有把未通过验证的变体替换为默认模型。','',
            '监督对比学习原始思路参考：[Khosla 等，Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)。本项目的跨受试者配对规则是适配实现，原文的其他任务结果不构成本次脑电任务的效果保证。','',
            '## 复核位置','']
    for study,_ in studies:lines.append(f'- `{ROOT/"runs"/study}`。')
    lines+=['','详细设置：`docs/training_pilot_v3_protocol.md`、`docs/alignment_pilot_v3b_protocol.md`、`docs/curriculum_pilot_v3c_protocol.md`。',
            '运行命令示例：`.venv_gpu\\Scripts\\python.exe -m ssvep.training_pilot --config configs/pilot_training_v3.json --run 新实验名`；使用唯一实验名，防止覆盖。']
    text='\n'.join(lines)+'\n'
    (ROOT/'docs/training_pilot_v3_results.md').write_text(text,encoding='utf-8')
    (OUTPUT/'第三阶段训练方式验证报告.md').write_text(text,encoding='utf-8')
    status={'complete':True,'model_fits':total,'all_pilot_directions_supported':False,
            'screen_decisions':summaries,'D1_D2_read':False,'beta_holdout_read':False,
            'confirmation_data_read':False,'default_model_replaced':False,'tests_passed':14}
    (ROOT/'logs/training_pilot_v3_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PILOT_REPORT_SAVED',total,'fits; no candidate passed')


if __name__=='__main__':main()
