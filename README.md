# SegFormer-Potsdam：多光谱遥感语义分割与鲁棒性实验

面向高分辨率遥感影像的语义分割研究项目。项目以 **SegFormer-B2** 为基础，围绕 RGB 与近红外（NIR）特征融合、固定/动态融合权重、成像退化鲁棒性，以及双时相 HRSCD 数据集的配对审计与实验协议准备开展实验。

> **项目定位：** 这是一个持续迭代的科研代码仓库，不是已经完成的通用遥感软件。当前仓库的主要实验集中在 Potsdam RGB/NIR 语义分割；HRSCD 部分包含数据审计、影像配对和实验协议构建工具。使用时请以实际脚本参数、数据协议和实验记录为准。

仓库地址：[zero-60/segformer_potsdam](https://github.com/zero-60/segformer_potsdam)

---

## 目录

- [项目目标](#项目目标)
- [当前研究内容](#当前研究内容)
- [模型方法](#模型方法)
- [数据集与类别](#数据集与类别)
- [仓库结构](#仓库结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [运行实验](#运行实验)
- [评估口径](#评估口径)
- [实验输出](#实验输出)
- [复现建议](#复现建议)
- [已知限制与注意事项](#已知限制与注意事项)
- [参考资料](#参考资料)
- [许可证](#许可证)

## 项目目标

本项目服务于复杂地形及成像干扰条件下的遥感影像自动解译研究，重点探索：

1. **高分辨率语义分割：** 使用 SegFormer-B2 对遥感影像进行像素级地物分类。
2. **RGB+NIR 多光谱融合：** 在 RGB 主干特征的基础上，引入 NIR 特征残差，研究近红外信息对分类性能的贡献。
3. **固定权重与动态权重对比：** 对比固定 NIR 融合系数与根据特征质量动态调整的融合门控机制。
4. **成像退化鲁棒性：** 研究高斯噪声、模糊、曝光不足和雾化等受控退化对 RGB/NIR 分割性能的影响。
5. **双时相变化检测准备：** 对 HRSCD 的双时相影像、地类标签和变化标签进行完整性审计、配对检查与可复现协议构建，为后续变化检测实验建立数据基础。

长期应用方向包括矿山复杂地貌分割、植被覆盖度估算、多时相地表变化检测及修复成效评估。上述应用目标仍需通过对应数据、标签、实验和独立验证逐步确认；不能仅凭单一数据集上的分割指标直接认定已达到实际矿山场景的验收要求。

## 当前研究内容

### 1. Potsdam RGB 语义分割基线

以 SegFormer-B2 为主干，建立 RGB 语义分割基线，为多光谱融合实验提供对照。

### 2. RGB+NIR 固定残差融合

使用独立的 RGB 与 NIR 编码器提取多尺度特征，并通过 NIR 残差适配器将辅助模态特征注入 RGB 特征：

\[
F_i = F_{\mathrm{RGB},i} + g \cdot \mathrm{Adapter}_i(F_{\mathrm{NIR},i})
\]

其中，\(g\) 为固定融合系数，\(i\) 表示特征尺度。当前 clean 实验脚本默认使用 \(g=1.0\)，也支持通过命令行修改该系数。

### 3. 动态可靠性门控融合（RR-DARF）

动态融合版本为不同尺度的 NIR 残差学习门控系数，使模型能够根据特征质量调整辅助模态的影响。当前实现对门控 logit 进行平滑约束，并在相对可靠性实验中研究不同模态退化条件下的门控响应。

### 4. Joint Robust-4 受控退化实验

联合退化协议对 RGB 与 NIR 同时施加受控成像退化，避免出现“辅助模态始终干净、主模态却受到干扰”的不公平比较。退化类型包括：

- 高斯噪声（Gaussian noise）
- 高斯模糊（Gaussian blur）
- 曝光不足（Underexposure）
- 雾化（Atmospheric fog）

该实验中的雾化方式属于受控近似，不应直接视为经过传感器或大气物理标定的成像模型。

### 5. HRSCD 双时相数据审计

仓库提供 HRSCD 数据检查与协议构建脚本，用于检查五类数据的数量、配对关系、尺寸、波段/数据类型、地理参考信息及标签取值，并生成项目定义的数据清单与划分。

**注意：** 项目生成的 HRSCD train/val/test 划分属于项目自定义划分，不应描述为数据集官方划分。当前的数据审计工具也不等同于完整的变化检测训练与评估系统。

## 模型方法

### SegFormer-B2

项目使用 Hugging Face Transformers 中的 SegFormer 实现，并以 NVIDIA 发布的 SegFormer-B2 预训练权重作为初始化来源。具体权重名称、下载方式和加载检查以相应模型构建脚本为准。

### 固定 NIR 残差融合

- RGB 分支负责提供主要语义表征。
- NIR 分支提取单通道近红外特征。
- 多尺度 NIR 残差适配器将辅助特征映射到与 RGB 特征相匹配的通道空间。
- 使用固定系数 \(g\) 控制 NIR 残差贡献。

### 动态门控融合

动态模型使用质量门控为不同尺度的 NIR 残差分配可学习权重。实验应同时检查分割指标和门控行为，避免仅凭门控系数变大/变小就断言模型可靠性判断正确。

### 损失函数与训练策略

当前部分训练脚本采用交叉熵损失与 Lovász-Softmax 损失的组合，并包含预训练参数与新建模块的差异化学习率、梯度累积、学习率调度、检查点保存及随机状态恢复等机制。不同实验脚本的参数可能不同，请以对应脚本中的默认参数和命令行帮助为准。

## 数据集与类别

### ISPRS Potsdam

Potsdam 实验使用 RGBIR 正射影像中的 RGB 与 NIR 信息，训练输入采用 `512 × 512` 图块。验证/测试阶段使用固定滑窗推理，并将重叠窗口的 logits 融合回完整 tile 后再生成最终预测。

当前数据协议中使用的六类地物为：

| 类别 ID | 英文类别 | 说明 |
|---:|---|---|
| 0 | Impervious surfaces | 不透水表面 |
| 1 | Building | 建筑物 |
| 2 | Low vegetation | 低矮植被 |
| 3 | Tree | 树木 |
| 4 | Car | 汽车 |
| 5 | Clutter/background | 杂类/背景 |

标签中的 `255` 用作 ignore/sentinel 值，不作为普通地物类别参与相应指标计算。类别颜色、标签编码与数据协议必须保持一致。

数据集原始文件通常较大，仓库不应被假定已经包含完整原始影像。请按照 `data_pipeline/` 中的数据读取逻辑及 `data/processed/potsdam/` 下的协议元数据准备本地数据。

### HRSCD

HRSCD 实验准备工具默认识别以下五类目录：

```text
data/HRSCD/
├── images_2006/
├── images_2012/
├── labels_change/
├── labels_land_cover_2006/
└── labels_land_cover_2012/
```

五类文件需要根据稳定的影像/瓦片标识完成配对。不要只按文件排序或文件名中的年份直接配对；应先运行审计脚本，并检查输出中的缺失配对、重复键、尺寸差异、地理参考差异和标签值问题。

HRSCD 标签中的类别定义与 Potsdam 不同，不可直接复用 Potsdam 的类别 ID、颜色映射或植被覆盖度计算口径。

## 仓库结构

以下为仓库的主要目录与代表性文件；目录中可能持续增加实验脚本。

```text
segformer_potsdam/
├── data/                       # 数据与处理后的协议元数据
├── data_pipeline/              # Potsdam 数据读取、采样和数据加载
│   └── potsdam_dataset.py
├── models/                     # 模型结构
│   ├── segformer_b2_rgbnir_fixed_clean.py
│   └── segformer_b2_darf_rr.py
├── tools/                      # 训练、评估、数据审计和协议构建工具
│   ├── train_model_b2_rgbnir_fixed_clean.py
│   ├── eval_model_b2_rgbnir_fixed_clean.py
│   ├── train_model_b2_darf_rr_joint_robust4.py
│   ├── inspect_hrscd_dataset.py
│   └── build_hrscd_cd0_protocol.py
├── evaluation/                 # 评估相关代码
├── references/                 # 参考资料
├── outputs/                    # 训练、评估和审计输出
├── src/                        # 项目辅助代码
├── requirements-lock.txt       # 当前环境的依赖锁定文件
└── README.md
```

## 环境配置

建议在独立的 Conda 或 Python 虚拟环境中运行。

```bash
git clone https://github.com/zero-60/segformer_potsdam.git
cd segformer_potsdam

conda create -n segformer-potsdam python=3.11 -y
conda activate segformer-potsdam

python -m pip install --upgrade pip
```

### 安装依赖

仓库中的 `requirements-lock.txt` 包含固定版本的依赖，其中 PyTorch / torchvision 是 CUDA 12.8 构建版本。若运行环境与该 CUDA 构建不匹配，**不要直接照搬整份锁定文件**；请先安装与本机驱动、CUDA 环境或 CPU 环境匹配的 PyTorch，再安装其余兼容依赖。

在兼容当前锁定版本的 CUDA 环境中，可使用：

```bash
pip install -r requirements-lock.txt
```

检查 PyTorch 是否能识别 GPU：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

首次构建模型时，预训练权重可能需要从 Hugging Face 下载。请确保运行环境可以访问权重来源，或提前配置好本地缓存。

## 数据准备

### Potsdam

1. 按照项目数据协议准备原始 RGBIR 影像与标签。
2. 确认数据读取代码需要的协议元数据存在，包括：
   - `data/processed/potsdam/labels_manifest.json`
   - `data/processed/potsdam/tile_split.json`
   - `data/processed/potsdam/rgbir_semantics.json`
   - `data/processed/potsdam/dataset_protocol.json`
3. 确认 RGBIR 影像、标签、tile ID、数据划分和类别映射相互一致。
4. 若修改了数据、划分、标签或预处理规则，应重新审计协议并记录变更；不要在不同实验之间无记录地更换数据口径。

### HRSCD

先检查数据：

```bash
python tools/inspect_hrscd_dataset.py \
  --root data/HRSCD \
  --output outputs/dataset_audit/HRSCD
```

通过审计后，生成项目定义的变化检测数据清单与划分：

```bash
python tools/build_hrscd_cd0_protocol.py \
  --root data/HRSCD \
  --output outputs/hrscd_cd0
```

只有在检查输出无错误、配对关系和标签编码均已确认后，才考虑使用 `--freeze` 冻结协议：

```bash
python tools/build_hrscd_cd0_protocol.py \
  --root data/HRSCD \
  --output outputs/hrscd_cd0 \
  --freeze
```

不要在审计存在错误时冻结协议。若输入数据发生变化，应先保存/归档旧协议，再按脚本要求重新生成。

## 运行实验

以下命令均应在仓库根目录执行。

### 1. 训练 clean RGB+NIR 固定融合模型

默认使用固定 NIR 残差融合系数 `g=1.0`：

```bash
python tools/train_model_b2_rgbnir_fixed_clean.py
```

自动恢复最近检查点：

```bash
python tools/train_model_b2_rgbnir_fixed_clean.py --resume auto
```

例如，将固定融合系数改为 `0.5`：

```bash
python tools/train_model_b2_rgbnir_fixed_clean.py --fixed-nir-strength 0.5
```

该 clean 实验不施加合成噪声、模糊、曝光不足或雾化退化；其余数据采样和几何增强仍遵循脚本定义的协议。

### 2. 评估 clean RGB+NIR 模型

请将 checkpoint 路径替换为实际生成的文件：

```bash
python tools/eval_model_b2_rgbnir_fixed_clean.py \
  --checkpoint outputs/training/b2_rgbnir_fixed_clean_g1p0_seed20260917/checkpoints/final.pt \
  --split val \
  --device cuda
```

如需在 CPU 上运行，可尝试 `--device cpu`；但完整 tile 推理的速度和内存占用可能较高。

### 3. 训练 RR-DARF Joint Robust-4 模型

```bash
python tools/train_model_b2_darf_rr_joint_robust4.py \
  --variant darf_rr
```

该实验按脚本定义的联合退化协议处理 RGB 与 NIR，并学习动态 NIR 门控。若要比较不同模型，请确保数据划分、随机种子、训练轮数、损失函数和评估口径一致，并在实验记录中注明任何不同设置。

### 4. 查看脚本支持的参数

若脚本版本更新导致参数不同，优先查看帮助信息：

```bash
python tools/train_model_b2_rgbnir_fixed_clean.py --help
python tools/eval_model_b2_rgbnir_fixed_clean.py --help
python tools/train_model_b2_darf_rr_joint_robust4.py --help
python tools/inspect_hrscd_dataset.py --help
python tools/build_hrscd_cd0_protocol.py --help
```

## 评估口径

### 语义分割指标

建议至少报告：

- **mIoU：** 各类别 IoU 的平均值。
- **Per-class IoU：** 每一类别的 IoU，用于发现小类或易混类别上的退化。
- **混淆矩阵：** 检查类别间的主要误分关系。
- **完整 tile 评估：** 对 Potsdam 验证/测试影像采用滑窗推理，将重叠窗口 logits 融合到完整 tile 后再计算指标。

不要将 patch 级指标与完整 tile 级指标直接混为一谈；比较实验时应使用相同的评估流程。

### 植被覆盖度

分割精度与覆盖度估算误差是不同的评价维度。若用于矿山植被覆盖度评估，应先明确：

1. 哪些标签类别计入“植被”；
2. GT 与预测是否使用相同的有效像素范围及 ignore 规则；
3. 覆盖度按单张原始影像、完整 tile 还是其他空间单元计算；
4. 使用像素计数比例还是百分数/百分点表示；
5. 报告逐影像误差、平均绝对误差，以及必要时的全局汇总误差。

可按以下方式定义单幅影像的覆盖度：

\[
C = \frac{N_{\mathrm{vegetation}}}{N_{\mathrm{valid}}} \times 100\%
\]

其中，\(N_{\mathrm{vegetation}}\) 为被判定为植被的有效像素数，\(N_{\mathrm{valid}}\) 为纳入统计的有效像素总数。覆盖度误差应在同一影像和同一统计口径下计算：

\[
E = |C_{\mathrm{pred}} - C_{\mathrm{GT}}|
\]

当 \(C\) 以百分数表示时，\(E\) 的单位为**百分点（pp）**。如果要判断误差是否满足 ±10% 的项目要求，必须先确认验收规范中的“10%”指百分点误差还是相对误差；两者不能互换。

**不要把 mIoU ≥ 90% 当作覆盖度误差 ≤ 10% 的等价条件。** 应分别计算和报告分割指标与覆盖度误差。

## 实验输出

训练与评估脚本通常会在 `outputs/` 下保存实验产物。具体子目录由脚本参数和实验配置决定。建议保留：

- 模型 checkpoint（例如 `latest.pt`、`final.pt`）
- 训练日志与逐 epoch 指标
- 实验配置、随机种子和协议版本
- 验证/测试指标与混淆矩阵
- 预测可视化与覆盖度核对表
- HRSCD 审计报告、配对清单和协议文件

训练输出和原始数据可能占用较大磁盘空间。提交代码或同步仓库前，请检查 `.gitignore` 与 Git 状态，避免误提交原始影像、缓存、权重和大型中间文件。

## 复现建议

1. **固定数据版本：** 保存数据清单、类别映射、划分文件和协议版本。
2. **固定随机种子：** 记录脚本使用的 seed；比较模型时尽量使用相同 seed 和相同数据协议。
3. **一次只改变一个主要因素：** 例如先比较固定融合系数，再比较动态门控，最后比较退化鲁棒性。
4. **保留基线：** RGB-only、clean RGB+NIR fixed 和 dynamic RR-DARF 的实验记录应分别保存，避免结果覆盖。
5. **区分验证集和测试集：** 选择模型与调参使用验证集；测试集用于最终报告。
6. **记录硬件环境：** 保存 Python、PyTorch、Transformers、CUDA、GPU 型号和显存信息。
7. **检查门控行为：** 对动态融合模型，除总体指标外，还应检查不同尺度门控的分布、稳定性和与模态质量变化的关系。
8. **避免过度解读：** Potsdam 上的实验结果不能直接替代矿山实景数据上的验证；迁移到实际矿山影像前，应完成独立数据测试与标签口径核对。

## 已知限制与注意事项

- 项目仍处于研究迭代阶段，脚本、参数和输出目录可能随实验调整。
- `requirements-lock.txt` 当前包含 CUDA 12.8 对应的 PyTorch 构建，不一定适用于所有机器。
- Potsdam 与 HRSCD 的类别体系、影像模态和标签编码不同，不应直接混用。
- HRSCD 数据审计和协议构建工具用于保证数据配对与实验可追溯性；不能单独证明变化检测模型有效，也不能替代后续的变化检测训练与评价。
- 受控噪声/模糊/曝光/雾化实验是合成退化实验，不完全等同于真实无人机、传感器、天气或地形造成的成像误差。
- 本 README 不预设模型已经达到特定精度或覆盖度验收指标。所有结果应以实际运行日志和独立评估为准。
- 请遵守所使用数据集的原始许可与引用要求。

## 参考资料

- **SegFormer:** Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. [论文（arXiv）](https://arxiv.org/abs/2105.15203)
- **预训练模型：** [nvidia/segformer-b2-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)
- **项目代码：** [zero-60/segformer_potsdam](https://github.com/zero-60/segformer_potsdam)

使用 Potsdam、HRSCD 或其他数据集开展研究时，请在论文、报告和公开结果中按数据集官方要求进行引用。

## 许可证

当前仓库尚未在 README 中声明具体开源许可证。除非仓库维护者另行添加明确的 LICENSE 文件，否则请不要默认该代码可以不受限制地复制、修改或再分发。
