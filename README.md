# ssvep_task

SSVEP 八分类算法研究。当前主线为 **整段 FBCCA 准确率优先 → 外部被试确认 → 赛方格式适配**。项目保留最初 EEGNet＋多尺度注意力 CNN＋知识蒸馏设想、后续路线修订和阴性实验结果。

本仓库包含完整赛方样例 **D1.csv、D2.csv**，以及公开数据下载脚本和来源清单。D1/D2 仅用于阶段验证，公开 BETA 可按受试者隔离训练。正式比赛数据格式、标签映射与参考样本规则仍须核对元信息。

## 最终赛题参数核查

用户提供的最终要求摘页只明确 **N 组数据、八类 0–7、逐试次识别、总体 ACC 与标准结果文件**；250 Hz、8–15 Hz、六通道和 1.5 秒窗口分别来自备赛样例、直播答疑和本仓库开发决策，必须在正式数据元信息中复核。详见 [最终要求与训练参数核查](docs/final_requirements_training_audit.md)。v5–v7 权重只能经 [匹配窗口预处理的新推理入口](ssvep/expanded_inference.py)使用；旧通用入口为历史全试次预处理协议。新入口写出的 CSV 仅供开发检查，不是赛题标准 Excel 模板。

## 当前结果

**v10 优化核验**：后部电极五谐波方案在开发集提高 3.13 pp，但在全新五名 BETA 被试上由 v9 的 88.13% 降至 84.38%，D2 亦下降一条，因此未替换默认方法。详见 [v10 阴性结果](docs/spatial_harmonic_v10_results.md)。

**v9 当前默认**：整段 FBCCA 在五名全新 BETA 被试上 138/160（86.25%），高于原固定 1.5 秒的 121/160（75.63%），5/5 人提高；冻结后验证 D1 47/48、D2 48/48。详情及限制见下方 v9 专节。

**v8 EEGNet 时域尺度改进**：按 EEGNet 原论文在 128 Hz 下约 0.5 秒的卷积时间尺度，将当前 250 Hz 模型的两层时域核从 63/15 点改为 125/31 点，其余 v7 训练配置不变。在预先指定的五名新 BETA 受试者上，三种子均值从 59.17% 提高到 68.12%，五人均有改善；固定 FBCCA 为 83.12%，故仍未过预设的替代门槛。D1/D2 仅在新权重冻结后作历史样例复核。详见 [v8 结果与协议](docs/expanded_v8_results.md)。


**v6/v7 续训结论**：D1/D2 在训练前通过原始 CSV → 48 试次 → 六通道 1.5 秒窗口 → FBCCA/EEGNet 八类分数的完整演算检查，无需调整样例格式。BETA 训练受试者从 14 扩到 26 人后，新增五人确认上 EEGNet 三种子均值 55.00%，固定 FBCCA 53.75%；但该 EEGNet 比原 14 人模型略低 0.63 个百分点。只提高 26 人训练更新量后，另一组新增五人上 EEGNet 45.21%、原 26 人模型 43.33%、FBCCA 72.50%。两轮新增确认组难度差异大，不能将单次超越解释为稳定泛化；默认仍保留 FBCCA。D1/D2 仅用于冻结后的历史验证，不能用于选参数。该轮默认仍为 FBCCA；v9 后已改用整段。见 [v6 报告](docs/expanded_v6_results.md)和 [v7 报告](docs/expanded_v7_results.md)。

**v5 扩大实验（固定 1.5 秒）**：BETA 训练受试者从 6 人增至 14 人，同一 EEGNet 结构和训练预算使用三个种子。在预先指定、此前未用于本项目开发的 5 名 BETA 受试者（160 个原始试次）上，EEGNet 宏平均准确率从 40.21% 提升到 46.67%（+6.46 个百分点），固定 FBCCA 为 52.50%。因此扩大训练带来了跨受试者改善，但 EEGNet 仍未稳定超过 FBCCA。D1/D2 仅在六个模型全部冻结后作历史样例复核：14 人 EEGNet 三种子均值分别为 43.75%/81.94%，FBCCA 为 68.75%/81.25%；两份小样例方向不一致，不能作为调参或盲测依据。详见 [v5 扩大实验报告](docs/expanded_v5_results.md)。

**v4 筛选实验**：

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
| FBCCA 物理频率/谐波方法 | v9 整段评分为当前默认 | 正式格式与参数到手后核对和适配 |
| EEGNet＋谐波特征轻量学生 | 保留候选 | 先提高跨受试者准确率，再考虑部署替代 |
| 多尺度注意力 CNN 教师→学生蒸馏 | 最初设想保留，近期暂停 | 教师须先在隔离公开开发/确认中稳定强于学生 |
| FBCCA 软目标蒸馏 | 第二轮已验证，小幅平均收益 | 需要更强教师与实际效率需求才扩大 |
| 子带/谐波共享评分融合 | 首个六权重配置未过筛 | 新的收敛或建模假设须另立实验，不追改本轮门槛 |
| 神经与 FBCCA 预测融合 | 预想分支，尚未确认收益 | 公开数据证明可学习的错误互补，不能把理想择优上界当效果 |
| 每类一个参考样本校准 | 预想分支 | 先在 BETA 按 block 隔离 support/query 模拟，正式规则明确后另行启用 |
| 因果流式识别 | 预想工程分支 | 独立验证滤波状态、归一化与端到端延迟 |

完整计划：[技术路线 v4](docs/technical_route_v4.md)；原始设想：[模型路线 v1](docs/model_route_v1.md)。v1 曾使用 D1 训练，是历史探索；不得复用其权重冒充现行“D1/D2 仅验证”协议。

## 当前默认：整段 FBCCA（v9）

准确率优先的 v9 实验在 BETA 三名开发被试上选择整段评分，并在五名此前未参与选型的 BETA 被试上确认：固定 1.5 秒 FBCCA 为 121/160（75.63%），整段为 138/160（86.25%），五人均提高。冻结方法后，D1 为 47/48（97.92%）、D2 为 48/48（100%）；两份旧样例只用于验证，无训练、调参或支持集校准。详见[完整协议和逐人结果](docs/duration_fbcca_v9_results.md)及[冻结配置](configs/expanded_v9.json)。正式数据格式和信号参数仍需以赛方实际文件核对，样例分数不是正式榜单成绩。

无标签 NPZ 输入（`X`、`channels`、`sampling_rate_hz`、`task_id`）可用 `python -m ssvep.duration_inference --input <input.npz> --output <predictions.csv>` 生成开发用 CSV。当前输入契约为 250 Hz、六通道 CP3/CPZ/CP4/PO3/POZ/PO4、每段至少 375 点；正式赛题需要按实际元数据接入读取器并填写赛方标准结果文件。

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

v6/v7 续训依次使用 `configs/expanded_v6.json`、`configs/expanded_v7.json`，入口分别为 `python -m ssvep.expanded_v6 train` / `confirm` 和 `python -m ssvep.expanded_v7 train` / `confirm`。训练前样例兼容性检查：`python -m scripts.smoke_competition_v6`。每轮确认结果均拒绝覆盖；已保存的运行目录无需重跑。

v5 实验使用 [冻结配置](configs/expanded_v5.json) 中的指定 BETA 人员。下载子集后分别运行 `python -m ssvep.expanded_v5 train` 和 `python -m ssvep.expanded_v5 confirm`；确认阶段只接受训练阶段写出的完整冻结记录，并拒绝覆盖已有确认结果。两阶段运行记录位于 `runs/v5-expanded-eegnet/`，报告可由 `python scripts/summarize_expanded_v5.py` 重建。

历史 `Run-Experiment.ps1` 为 v1 入口，现行数据用途策略会阻止其将赛方数据用于训练。历史推理导出与 v4 输入范围不同；不能直接混用权重和预处理。`scripts/report_*.py` 中部分本机交付路径属于历史工作流，异地运行应调整报告输出位置。

## 目录

- `ssvep/`：数据、算法、训练、推理与验证代码。
- `configs/`：来源清单、数据用途策略、参数与实验配置。
- `docs/`：路线选择、赛方参数依据、协议及结果报告。
- `runs/`：已完成及失败实验的可审计记录；以状态文件区分，失败记录不是有效结果。
- `scripts/`、`tests/`：下载核验、报告及自动检查。

公开数据来源：BETA [Figshare](https://doi.org/10.6084/m9.figshare.12264401)、Nakanishi [作者仓库](https://github.com/mnakanishi/12JFPM_SSVEP)、Benchmark [Zenodo 镜像](https://zenodo.org/records/14865172)。赛方材料与各公开数据的权利及许可分别归原发布方，本仓库不额外授予其使用许可。
