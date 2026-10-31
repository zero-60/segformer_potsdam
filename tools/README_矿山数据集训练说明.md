# TLSNMSP 矿山 RGB+NIR 训练适配包

本包为现有 `D:\MRISE\segformer_potsdam` 项目新增文件，不覆盖原有 Potsdam 代码和检查点。

## 已知正式训练环境

- 项目根目录：`D:\MRISE\segformer_potsdam`
- 数据根目录：`D:\MRISE\dataset\TLSNMSP`
- Conda 环境：`mine-seg`
- GPU：NVIDIA GeForce RTX 4060
- 现有依赖检查：`pip check` 通过，torch / transformers / cv2 / albumentations / tifffile / pandas / scipy / sklearn / matplotlib / accelerate / einops / tensorboard 可导入。

代码使用项目已有的 `models/segformer_b2_rgbnir_fixed_clean.py`，以及 `models/segformer_rgb.py` 等依赖。没有要求新建环境，也没有引入 rasterio 依赖。

## 包含文件

将压缩包内的 `models` 和 `tools` 文件夹合并复制到项目根目录：

- `models/segformer_b2_rgbnir_meris.py`：新建七类 RGB+NIR 模型封装；不修改 Potsdam 原模型。
- `tools/prepare_meris_tlsnmsp.py`：建立可复现的 train/val/test 清单、核验原始标签与转换标签、计算训练集 NIR 均值和标准差。
- `tools/train_model_b2_rgbnir_meris.py`：训练、验证、最终测试、断点保存与恢复。
- `tools/run_meris_prepare.bat`：一键生成数据协议。
- `tools/run_meris_smoke_test.bat`：只做一个批次的前向与反向检查。
- `tools/run_meris_train.bat`：按默认参数启动训练。

## 第一步：复制新文件

解压后，将压缩包内 `models` 和 `tools` 两个目录合并复制到：

`D:\MRISE\segformer_potsdam`

Windows 提示合并目录时选择合并即可。文件名均为本任务新增文件，不应覆盖现有同名文件。如果目标目录已经存在同名文件，请先备份并检查版本。

## 第二步：激活现有环境

在 Anaconda Prompt 或 PowerShell 中执行：

```powershell
conda activate mine-seg
cd D:\MRISE\segformer_potsdam
python -m pip check
```

本包不要求创建新环境。准备脚本使用项目环境已有的 `cv2`、`tifffile`、`numpy`；训练脚本使用已有 `torch`、`transformers`。

## 第三步：准备数据协议

```powershell
python tools\prepare_meris_tlsnmsp.py --data-root "D:\MRISE\dataset\TLSNMSP"
```

默认输出目录：

`D:\MRISE\segformer_potsdam\data\processed\meris_tlsnmsp`

将生成：

- `manifest_all.csv`
- `train.csv`、`val.csv`、`test.csv`
- `dataset_config.json`
- `class_pixel_counts_by_split.csv`
- `preparation_report.txt`

基于当前审计结果，预期按航带组划分为 126/9/41 张有标签影像：

- train：100、101、103、106、107
- val：102
- test：108

这是项目自定义划分，不是数据集官方划分。脚本会逐像素检查原始彩色 `mask` 和 `mask_converted`，如果发现未知颜色、标签不一致、缺文件或尺寸不一致，会停止，不会静默忽略。

NIR 从五波段 TIFF 的第 5 波段读取。NIR 均值/标准差只从训练组影像按固定步长抽样计算，避免使用验证集或测试集的统计量。原始影像不会被修改。

## 第四步：先跑 smoke test

```powershell
python tools\train_model_b2_rgbnir_meris.py --smoke-test
```

该命令会构建模型、读取一个 512×512 RGB/NIR/标签样本，执行前向、计算损失并反向传播，然后退出；不会进入长时间训练，也不会写训练检查点。

默认使用现有的 `nvidia/segformer-b2-finetuned-ade-512-512` 预训练 B2 编码器，七类分类头重新初始化。首次运行如果本机缓存没有该权重，需确保可以访问 Hugging Face，或预先把该模型下载到 Transformers 缓存。

如果正式训练电脑上另有已训练好的 Potsdam RGB+NIR 固定残差模型 checkpoint，可以额外传入 `--init-checkpoint "绝对路径\final.pt"`。脚本只加载形状完全匹配的非分类头参数，并保留新任务的七类分类头随机初始化。该选项只适用于结构兼容的固定残差 RGB+NIR checkpoint；不要直接传入 RGB-only 或 DARF 动态门控 checkpoint。没有可用 checkpoint 时，保持默认即可。

预期日志包含 `[SMOKE PASS]`、RGB shape `(1, 3, 512, 512)`、NIR shape `(1, 1, 512, 512)`、logits shape `(1, 7, 512, 512)` 和有限损失值。若 CUDA 显存不足，先把错误信息发回，不要直接启动正式训练。

## 第五步：开始正式训练

```powershell
python tools\train_model_b2_rgbnir_meris.py --epochs 40 --batch-size 1 --grad-accum-steps 4 --num-workers 0
```

默认配置：

- SegFormer-B2 双编码器 RGB+NIR 固定残差融合，`g=1.0`
- 7 类输出
- RGB ImageNet 标准化
- NIR 使用 train-only DN 均值/标准差 z-score
- 每张训练影像每个 epoch 随机采一个 512×512 patch，并同步执行旋转/翻转增强
- 验证/测试采用固定 512×512 窗口，按全体像素累计混淆矩阵
- Cross Entropy + 0.5 × Lovasz-Softmax
- AMP 在 CUDA 可用时启用，梯度检查点默认尝试开启
- Windows DataLoader 默认 `num_workers=0`
- 根据验证集 mIoU 保存 `best.pt`，并保存 `latest.pt` 和训练日志
- 默认不使用类别权重，便于先得到一个可比较的基线；后续可使用 `--class-weight-mode sqrt_inverse` 开展类别不平衡对照实验

如果你确实有兼容的 Potsdam 固定残差 checkpoint，示例命令如下（将路径替换为实际文件）：

```powershell
python tools\train_model_b2_rgbnir_meris.py --init-checkpoint "D:\MRISE\segformer_potsdam\outputs\training\你的实验目录\checkpoints\final.pt" --smoke-test
```

如果训练中断后需要从最新检查点恢复：

```powershell
python tools\train_model_b2_rgbnir_meris.py --resume auto --epochs 40 --batch-size 1 --grad-accum-steps 4 --num-workers 0
```

默认输出目录：

`D:\MRISE\segformer_potsdam\outputs\training\meris_b2_rgbnir_fixed_g1_seed20261028`

## 重要说明

1. 原始标签 ID `1..7` 在数据加载时映射为模型索引 `0..6`；标签文件不会被改写。
2. 原始 ID 对应：1 建筑物，2 硬化地表，3 水体，4 裸地，5 树木，6 低矮植被，7 背景/其他目标。
3. 本任务新建七类分类头，不会沿用 Potsdam 六类分类头的类别语义；预训练编码器仍从 ADE20K B2 初始化。
4. 当前 split 按航带组分开，降低相邻影像泄漏风险；但验证集只有 9 张图，验证指标方差可能较大，报告中需要说明。
5. 影像尺寸一致不等于像素级配准已经完全验证。正式研究应抽查 RGB/NIR/GT 的细小边界，特别是道路、水体边缘和小型建筑。
6. 该训练脚本不会自动覆盖原有 Potsdam checkpoint。所有输出写入新的 `outputs/training/meris_...` 目录。
7. 若 smoke test 失败，请将完整终端错误、`preparation_report.txt` 和 `dataset_config.json` 发回，再继续修正；不要跳过 smoke test。
