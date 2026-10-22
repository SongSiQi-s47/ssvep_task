# ssvep_task

SSVEP 八分类算法研究。当前主线为 **限定观察窗口的 FBCCA 基线 → 有限学习模型对照 → 达标后再考虑蒸馏与压缩**。项目保留最初 EEGNet＋多尺度注意力 CNN＋知识蒸馏设想、后续路线修订和阴性实验结果。

本仓库包含完整赛方样例 **D1.csv、D2.csv**，以及公开数据下载脚本和来源清单。D1/D2 仅用于阶段验证，公开 BETA 可按受试者隔离训练。正式比赛数据格式、标签映射与参考样本规则仍须核对元信息。

## 当前结果

最近一次固定 1.5 秒、窗口内预处理的算法对照：六名 BETA 训练受试者，三名筛选开发受试者共 96 段，两个初始化种子。

| 方法 | 开发准确率 | CPU 计算中位耗时 |
|---|---:|---:|
| 固定 FBCCA | 68.75% | 21.31 ms |
| EEGNet 适配学生，两种子均值 | 48.96% | 约 5.56 ms |
| 共享线性评分器，两种子均值 | 57.29% | 约 22.09 ms |

这些是反复使用的公开开发集结果，不是独立测试结论。耗时包含预处理和特征提取，不含信号等待。该轮未读取 D1/D2、内部留出或确认数据，学习评分器未过筛；本仓库没有用它替换部署模型。

- [固定窗口算法详细比较](docs/scoring_pilot_v4_results.md)
- [观察窗口与预处理审计](docs/window_audit_v4_results.md)：同批开发数据 1 秒 56.25%、1.5 秒 68.75%；这是观察长度的收益。
- [第二轮完整 BETA 训练与蒸馏结果](docs/round2_beta_results.md)
- [第三阶段训练方式阴性结果](docs/training_pilot_v3_results.md)

## 路线与预想分支

| 路线 | 当前状态 | 继续条件 |
|---|---|---|
| FBCCA 物理频率/谐波方法 | 当前主基线 | 保持统一窗口与预处理对照 |
| EEGNet＋谐波特征轻量学生 | 保留候选 | 先提高跨受试者准确率，再考虑部署替代 |
| 多尺度注意力 CNN 教师→学生蒸馏 | 最初设想保留，近期暂停 | 教师须先在隔离公开开发/确认中稳定强于学生 |
| FBCCA 软目标蒸馏 | 第二轮已验证，小幅平均收益 | 需要更强教师与实际效率需求才扩大 |
| 子带/谐波共享评分融合 | 首个六权重配置未过筛 | 新的收敛或建模假设须另立实验，不追改本轮门槛 |
| 神经与 FBCCA 预测融合 | 预想分支，尚未确认收益 | 公开数据证明可学习的错误互补，不能把理想择优上界当效果 |
| 每类一个参考样本校准 | 预想分支 | 先在 BETA 按 block 隔离 support/query 模拟，正式规则明确后另行启用 |
| 因果流式识别 | 预想工程分支 | 独立验证滤波状态、归一化与端到端延迟 |

完整计划：[技术路线 v4](docs/technical_route_v4.md)；原始设想：[模型路线 v1](docs/model_route_v1.md)。v1 曾使用 D1 训练，是历史探索；不得复用其权重冒充现行“D1/D2 仅验证”协议。

## 数据集位置

| 数据 | 仓库相对位置 | 是否上传原始数据 |
|---|---|---|
| 赛方 D1 | [data/raw/competition/D1.csv](data/raw/competition/D1.csv) | 完整原始 CSV |
| 赛方 D2 | [data/raw/competition/D2.csv](data/raw/competition/D2.csv) | 完整原始 CSV |
| 比赛说明 | [data/raw/competition/competition_requirements.pdf](data/raw/competition/competition_requirements.pdf) | 原始 PDF |
| BETA | `data/raw/public/beta/` | 本地另行下载，保存来源与校验清单 |
| Nakanishi 2015 | `data/raw/public/nakanishi2015/` | 本地另行下载 |
| Tsinghua Benchmark | `data/raw/public/benchmark/` | 本地另行下载，具体文件路径以脚本/清单为准 |
| D1/D2 派生试次数组 | `data/processed/competition/*_trials.npz` | 本地由脚本生成 |

数据说明、公开数据来源及本地原位置：[数据目录](data/README.md)。D1/D2 原文件校验值见 [上传清单](data/competition_manifest.json)。每份 CSV 均为 48,000 行数据、48 个原始试次、六个脑电通道及 taskID/stimID；文件字节未改写。

## 运行

建议 Python 3.10。新建环境后在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_competition.py
python -m unittest discover -s tests
```

下载公开数据：

```powershell
python scripts/download_public.py --workers 4
python scripts/download_benchmark.py
```

完整公开下载可能达数 GB；不要将全部新增受试者自动用于训练。按已冻结的角色与配置管理数据，不把重叠窗口当成独立受试者。当前已完成实验的来源、角色和结果保存在 `runs/`；大模型权重、缓存、运行环境和公开原始数据未纳入 Git。

最近两项实验入口：

```powershell
python -m ssvep.window_audit
python -m ssvep.scoring_pilot
```

它们依赖已下载 BETA 和 `runs/v2-beta-subject-isolated/` 中的分组/指纹记录，且拒绝覆盖已有运行目录。重跑需另设运行目录并保留原结果，不能删除原记录后称为新验证。评分器实验显式使用 CPU；GPU 在最近一次运行中初始化失败。最近 CPU 测试 19 项通过、1 项 GPU 检查跳过。

历史 `Run-Experiment.ps1` 为 v1 入口，现行数据用途策略会阻止其将赛方数据用于训练。历史推理导出与 v4 输入范围不同；不能直接混用权重和预处理。`scripts/report_*.py` 中部分本机交付路径属于历史工作流，异地运行应调整报告输出位置。

## 目录

- `ssvep/`：数据、算法、训练、推理与验证代码。
- `configs/`：来源清单、数据用途策略、参数与实验配置。
- `docs/`：路线选择、赛方参数依据、协议及结果报告。
- `runs/`：已完成及失败实验的可审计记录；以状态文件区分，失败记录不是有效结果。
- `scripts/`、`tests/`：下载核验、报告及自动检查。

公开数据来源：BETA [Figshare](https://doi.org/10.6084/m9.figshare.12264401)、Nakanishi [作者仓库](https://github.com/mnakanishi/12JFPM_SSVEP)、Benchmark [Zenodo 镜像](https://zenodo.org/records/14865172)。赛方材料与各公开数据的权利及许可分别归原发布方，本仓库不额外授予其使用许可。
