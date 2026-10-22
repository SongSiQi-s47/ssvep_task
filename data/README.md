# 数据目录与使用边界

## 已随仓库完整提供

- `raw/competition/D1.csv`：5,956,291 字节。
- `raw/competition/D2.csv`：5,927,734 字节。
- `raw/competition/competition_requirements.pdf`：比赛原文。
- `processed/competition/trial_index.csv`：原始试次索引。

D1/D2 来自此前对话中的赛方附件，原始字节完整保留。每份 48,000 行数据、六通道 CP3/CPZ/CP4/PO3/POZ/PO4，另有 taskID/stimID。各 48 段、每段 1,000 点。暂采用 250 Hz、0–7 对应 8–15 Hz；这是备赛映射，正式元信息待复核。D1/D2 的受试者关系未明确，不能称为已验证的跨受试者划分。

用途：**仅验证**。禁止用于训练、生成蒸馏软目标、伪标签、早停、模型选择或参考样本校准。历史 v1 探索记录不构成现行协议的独立验证。

运行 `python scripts/prepare_competition.py` 生成 `processed/competition/D1_trials.npz`、`D2_trials.npz`。派生文件未上传，原始 CSV 已上传。

## 公开补充数据（另行下载）

| 数据集 | 本地相对目录 | 来源配置与入口 |
|---|---|---|
| BETA | `raw/public/beta/` | `configs/beta_source.json`、`configs/download_manifest.json`；`scripts/download_public.py` |
| Nakanishi 2015 | `raw/public/nakanishi2015/` | 同一公开下载清单与脚本 |
| Benchmark | `raw/public/benchmark/` | `configs/benchmark_source.json`、`configs/benchmark_download_manifest.json`；`scripts/download_benchmark.py` |

原工作机项目根目录为 `C:\Users\64443\Desktop\ssvep_project`，公开数据在该目录的上述相对路径。克隆后以本仓库位置作为根目录，不能依赖其他电脑的原绝对路径。

当前本地 BETA 已校验 24 人，原数据共 70 人；Nakanishi 10 人，Benchmark 6 人。已获取不等于全部数据集完整。下载状态以发布者校验值及脚本生成记录为准，不把 `.part` 当成可用数据。

BETA 可训练，必须保持受试者隔离；第二轮原分组见 `runs/v2-beta-subject-isolated/subject_roles.json`。本轮小规模训练与筛选名单见 `runs/v4-scoring-pilot-cpu/preregistered_config.json`。已有开发与测试集合有历史暴露，不称全新盲测。Nakanishi 电极/目标频率不完全匹配，不能直接补零混入赛题八分类。

所有来源与现行策略见 `configs/data_usage_policy.json`。数据原发布方的许可与比赛使用规则分别适用。
