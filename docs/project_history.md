# SSVEP 项目

## 最新比较：1.5 秒算法筛选（2026-10-17）

四次 CPU 训练完成。相同 96 段 BETA 筛选开发数据上：FBCCA 68.75%，EEGNet 两种子均值 48.96%，共享评分器 57.29%。评分器未通过晋级，停止扩大此配置；未读取 D1/D2、内部留出或确认数据，未替换部署模型。详情见 docs/scoring_pilot_v4_results.md，运行目录 runs/v4-scoring-pilot-cpu。CPU 测试 19 项通过、1 项 GPU 检查跳过；本轮 GPU 初始化失败单独记录。


## 最新验证：观察窗口审计（2026-10-17）

已完成限定窗口预处理与 FBCCA 对照：三名 BETA 筛选开发受试者共 96 段，1 秒 56.25%，1.5 秒 68.75%。冻结后续算法比较窗口为 375 点；收益属于观察长度变化。未读取 D1/D2、内部留出或确认开发受试者，未替换部署模型。详细报告：docs/window_audit_v4_results.md。

## 当前技术路线：v4（2026-10-16）

主线修订为：评估口径修正 → FBCCA 基线 → 小型可学习评分融合 → 有条件蒸馏。保留 EEGNet 候选，暂停弱 CNN 教师及第三阶段未过筛的训练变体。下一轮先验证只使用指定窗口信号的预处理与 1.0/1.5 秒窗口，再做有限算法筛选。路线与晋级规则见 docs/technical_route_v4.md；本次未启动新训练，未改变部署模型。


## 最新阶段：训练方式小规模验证已完成

第三阶段共完成 16 次小规模训练，对比均衡随机切窗、相位/幅值增强、余弦学习率、跨受试者特征对齐和长窗到短窗课程。各候选均未通过预设的双种子增益门槛，因此没有扩展至完整实验，也没有替换第二轮模型。

报告：`docs/training_pilot_v3_results.md`；可重跑入口：`ssvep/training_pilot.py`，配置为 `configs/pilot_training_v3.json`、`configs/pilot_alignment_v3b.json`、`configs/pilot_curriculum_v3c.json`。本阶段未读取 D1/D2、BETA 内部测试受试者或保留确认开发受试者。

## 当前有效协议：第二轮独立来源训练

用户已明确：D1/D2 只验证，BETA 可以训练并按受试者隔离。当前主实验 `v2-beta-subject-isolated` 使用 24 名 BETA，固定 14 人训练、5 人开发验证、5 人内部留出；D1/D2 全部 96 段仅在模型冻结后评分，不参与选择和校准。全部权重新初始化。算法教师蒸馏续实验为 `v2-beta-algorithm-distilled`。此前保守口径运行的 Benchmark 先导实验已完成，作为另一条历史记录保留。

新入口 `Start-Round2.ps1`，最新协议见 `docs/round2_beta_protocol.md`；澄清前的 Benchmark 协议保留于 `docs/round2_protocol.md`。运行状态为 `logs/round2.pipeline.json`，主实验和蒸馏结果位于上述独立目录。旧 `Run-Experiment.ps1` 属于历史 v1 协议，直接启动会被新的数据用途检查阻止。

以下 v1 模型路线与结果保留为探索历史，不构成新协议下的独立验证结论。

项目根目录：`C:\Users\64443\Desktop\ssvep_project`。

当前已开始首版模型构建：原始数据适配、CCA/滤波器组 CCA 基线、注意力多尺度 CNN 教师、EEGNet 适配学生、输出分布蒸馏以及每类一个参考样本的校准。代码位于 `ssvep/`，完整算法与验证协议见 `docs/model_route_v1.md`；每次实际训练结果保存在 `runs/`，只有含 `completed.json` 的实验才视为完成。

模型入口为 `Run-Experiment.ps1`，可用 GPU 训练环境为 `.venv_gpu`（复用本机 EEG 环境已安装的计算库，不修改原环境）。`.venv` 是新组件安装尝试的未完成环境，不用于实际训练。运行 `Get-Status.ps1` 查询的是数据下载进度，不是模型训练进度。

## 直播答疑补充（2026-10-14 归档）

用户新提供直播转写已归档于 `docs/sources/live_qa_original_lines.txt`；完整参数结论、原文行号及不确定项见 `docs/live_qa_parameters.md`。备赛暂采用 250 Hz、0–7 依次对应 8–15 Hz，正式数据核对元信息。电压单位原词为“微幅”，疑似微伏，尚不进行幅值换算。今年正式格式不是 CSV，具体结构未知；正式比赛每类提供一个带标签参考样本，可用于校准，组/受试者范围待确认。直播称 AI 辅助编程不限制，同时多处要求以规则书为准；相关工具与成品算法限制需结合现有 PDF 和后续正式通知理解。

## 已取回的赛方资料

- `data/raw/competition/D1.csv`、`D2.csv`：由此前 ChatGPT 对话附件复制，原始数值未修改。
- `data/raw/competition/competition_requirements.pdf`：比赛原文。
- 每份 CSV 为 48,000 行、8 列，脑电通道为 CP3、CPZ、CP4、PO3、POZ、PO4。
- `taskID` 在样例中是片段编号 0–47；每段 1,000 点。`stimID` 是 0–7 的刺激标记，每个标记在每份文件出现 6 段。根据数据结构暂将其视为样例类别，等待赛方说明确认语义。
- 共 96 段，8 类均衡、每类 12 段，无缺失值或非有限数值。
- `data/processed/competition/` 保存无滤波、无重采样的片段数组；每份 X 形状为 `[48, 6, 1000]`，通道中不含 taskID/stimID。
- 直播补充后的备赛参数为 250 Hz、0–7→8–15 Hz，因此当前 1,000 点样例可按 4 秒记录处理；不代表正式数据恒为 4 秒。幅值单位、刺激起点与延迟、D1/D2 的受试者关系仍未确认。

## 公开补充数据

1. BETA（主补充）：70 名受试者，64 通道、40 个目标、每人 4 个 block；原始 MATLAB 文件约 5.28 GB。采用论文指定的 Figshare 原始数据入口，保存发布者 MD5 和本地 SHA-256。
2. Nakanishi 2015（辅助实验）：作者仓库中的 10 人、8 通道、12 目标数据，256 Hz；每个 MATLAB 文件预期 `[12, 8, 1114, 15]`。其通道不能完整覆盖赛方的六通道，因此只作为独立辅助数据源，不能补零伪造同名电极。
3. Tsinghua Benchmark（候选）：已核查官方下载入口，但本机访问遇到 TLS 连接异常，尚未获取。先完成 BETA 和 Nakanishi，避免下载多套数据挤占训练空间。

公开数据的数字类别与赛题的 stimID 没有自动对应关系。后续按物理频率、通道、采样率及刺激起点建立适配规则；频率不匹配时保留各自分类头或进行表征预训练。直播补充明确可使用正式比赛包中赛方指定的带标签参考样本校准；其余无标签样本为预测目标，不能当作带真值的监督训练数据。BETA 包含八个暂定目标频率，可按实际元信息筛选。

## 下载和核验

```powershell
python scripts/download_public.py --workers 4
python scripts/prepare_competition.py
python scripts/verify_public.py
```

需要 Python 的 requests、numpy、scipy（当前本机环境已具备）。下载脚本可重跑：保留 `.part` 并请求断点续传；服务端不支持 Range 时从头覆盖临时文件；仅在文件大小和发布方 MD5 核验通过后标记 verified。Nakanishi 没有发布方 MD5，核验 MATLAB 结构并记录本地 SHA-256，不能将其称为发布方校验。

双击或运行 `Start-Download.ps1` 可在后台启动/恢复下载；重复运行会检查是否已有该项目的下载进程。运行 `Get-Status.ps1` 可查看实际校验完成数量和后台进程状态。

下载进度：`logs/download_status.json`。主下载程序最终结果：`logs/download_summary.json`；下载全部任务结束后会运行 MATLAB 数组及有限值核验，结果写入 `docs/public_validation.json`。若存在 `logs/nakanishi_download_status.json`，它记录优先获取 Nakanishi 的独立下载进度。存在 `.part` 只代表未完成下载。后台下载依赖电脑开机和网络连接；中断后可恢复。

BETA 主站在本机无法稳定建立连接，使用论文指定的 Figshare 入口；断线重试可切换至发布者的 Hugging Face 镜像，但仍以 Figshare 原始大小及 MD5 验证内容一致性。保留相位元数据 2022 年更正说明，后续处理不得直接使用未修正相位。

原始数据和来源清单分别位于 `data/raw/`、`configs/download_manifest.json`；BETA 原始 API 元数据快照位于 `configs/beta_source.json`。片段索引和核验信息见 `data/processed/competition/trial_index.csv` 与 `docs/competition_inventory.json`。

## 尚需赛方提供

- QQ 群（100628255）中的备赛/正式数据链接、百度网盘提取码与比赛时公布的解压密码。
- 250 Hz 与 0–7→8–15 Hz 的正式元信息复核、事件定义、电极布局、受试者/场次及参考样本所属组信息。
- 标准结果 Excel 模板及外部训练数据的具体使用规定。

比赛文件要求正式数据在 T0 前 4 小时经 QQ 群发布、T0 公布密码，因此不能通过普通公开网页替代获取。已取得的 D1/D2 仅按已提供样例管理，不声称正式数据已经齐全。

比赛原文对自主编程、外部自动化工具替代自主研发有明确限制。本阶段仅处理数据获取与核验；后续算法实现前应由参赛团队核对允许使用的工具范围。原文没有明确外部训练数据许可，下载并不意味着可直接用于正式参赛。

## 来源与引用

- 比赛说明：本项目归档的 PDF，报名网站 https://www.aicomp.cn/。
- 清华官方下载目录：https://bci.med.tsinghua.edu.cn/download.html
- BETA 数据 DOI：https://doi.org/10.6084/m9.figshare.12264401
- BETA 论文：https://doi.org/10.3389/fnins.2020.00627
- BETA API：https://api.figshare.com/v2/articles/12264401 （记录中许可为 CC BY 4.0，科研用途等说明保留在原始资料中）。
- Nakanishi 作者数据仓库：https://github.com/mnakanishi/12JFPM_SSVEP
- Nakanishi 论文：https://doi.org/10.1371/journal.pone.0140703 （论文许可与数据许可不能混为一谈；保存仓库说明，使用前核查数据许可）。

此前对话的工具读取返回了三个附件和用户提问，但助手方案正文仅为内容引用占位符，不能完整恢复初始路线图。本项目据实际附件和上述原始来源启动数据阶段；参数采用已归档来源，并明确标记推断与未知项。



