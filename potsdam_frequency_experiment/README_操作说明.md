# Potsdam 双B2固定NIR融合＋频率内容门控实验 v1

## 代码依据和设计边界

参考仓库 https://github.com/zero-60/segformer_potsdam ，检查版本39e59f1fbe407c9457330a567527b63eaa98ebe4。

对照 models/segformer_b2_rgbnir_fixed_clean.py：保留双B2编码器，NIR首层为预训练RGB卷积核的通道均值；保留四尺度、零输出初始化NIR bottleneck残差适配器与fixed=1.0；采用原SegFormer解码头。数据采样的SHA256候选坐标/D4规则与 data_pipeline/potsdam_dataset.py 相同；新加载器读取实际协议中的样本量，不再写死1152。

本包是独立实验实现，不覆盖仓库文件，不加载旧固定模型训练权重作为频率实验初始化。`baseline`和`frequency`具有相同共享模块初始化、采样/增强/优化配置；频率模块在共享模块创建后初始化。数值运算不保证跨GPU/软件版本逐位一致。

本次只改变频率模块，不额外加入类别感知、类别加权CE或退化增强，以便判断频率模块本身的作用。若要研究“频率＋类别感知”的交互，需另做清楚的消融比较。

## 查阅的论文与对应思路

1. Linwei Chen等，Frequency-aware Feature Fusion for Dense Image Prediction，TPAMI 2024。
   https://arxiv.org/abs/2408.12879
   https://github.com/Linwei-Chen/FreqFusion
   论文包含自适应低通、高通和偏移重采样，处理类内不一致与边界模糊。本包借鉴高低频分开处理与自适应融合的思路，没有实现它的动态滤波器生成或偏移重采样。
2. Xin Li等，Frequency-Guided Denoising Network for Semantic Segmentation of Remote Sensing Images，IEEE TGRS 2026，64卷，DOI 10.1109/TGRS.2025.3648408。
   https://hub.hku.hk/handle/10722/379254
   通过作者机构记录与摘要核对其频率敏感特征、多阶段融合与边缘保持动机；这里没有声称完整复现DRB、FDM、FFM或FCFM，也没有加入其去噪结构。

这是受文献启发的简化FFT内容门控残差实验，不能命名为论文官方FreqFusion/FreDNet，也不能直接把该组合视作已验证的创新。

## 模型框架

RGB → B2编码器 → 四尺度RGB特征。

NIR → B2编码器 → 四尺度NIR特征 → bottleneck Adapter。

每尺度先融合 `F = F_RGB + 1.0 × Adapter(F_NIR)`。

frequency模式下，每尺度F先反射扩展，再进行float32 FFT，使用平滑高斯低通掩码得到L，令H=F−L。频率截止参数cutoff默认0.15 cycles/feature-pixel，固定不学习。低/高频经各自深度卷积处理，内容门控根据当前特征为每个通道、空间位置输出两个softmax权重；再经1×1映射，以受限残差强度回加F。四尺度结果进入原SegFormer解码头，得到六类logits并双线性插值。

公式：`F' = F + tanh(alpha) × Out(GELU(gL × DWConv(L) + gH × DWConv(H)))`。

门控是按内容、通道和位置调整高低频贡献；没有按GT类别指定权重。alpha初始0.1；输出投影小值随机初始化，保证第一步频率门控和卷积能获得梯度。反射扩展减轻FFT周期边界影响，不能保证消除切片接缝；全局上下文仍受512裁剪限制。高频也可能含噪声，不预设增强一定提升精度。

baseline模式完全跳过频率模块。

## 放置路径

将本文件夹解压放到项目根目录，最终有：

```text
/home/panyao/ty/MRISE/segformer_potsdam/potsdam_frequency_experiment/train.py
/home/panyao/ty/MRISE/segformer_potsdam/potsdam_frequency_experiment/validate.py
```

默认路径相对项目根目录解析：

| 项目 | 默认路径 |
|---|---|
| 新协议 | data/processed/potsdam_frequency_quick_v1 |
| 原始数据根 | data/raw/potsdam |
| RGBIR | data/raw/potsdam/_expanded/4_Ortho_RGBIR |
| 预训练 | ../hf-cache/segformer-b2-finetuned-ade-512-512 |
| 训练输出 | outputs/training/potsdam_quick_频率或基线_seed20260917 |

预训练默认绝对位置即 /home/panyao/ty/MRISE/hf-cache/segformer-b2-finetuned-ade-512-512。所有预训练加载local_files_only=True，不联网。若影像实际位置不同，用 --rgbir-dir 修改；--raw-root必须能与manifest中的gt_relpath正确拼接。

## 第一步：真实数据Smoke Test

```bash
conda activate mine-seg
cd /home/panyao/ty/MRISE/segformer_potsdam
python potsdam_frequency_experiment/train.py --variant baseline --smoke-test
python potsdam_frequency_experiment/train.py --variant frequency --smoke-test
```

预期：train_tiles=6、val_tiles=2、samples/epoch=192、crop=512，随后出现SMOKE PASS；频率版本还输出正的梯度和。Smoke Test只前向、反向，不开始30轮训练，不写入正式输出。ADE20K分类器150→6的形状不匹配提示是预期现象。

如需先完成一个真实完整验证来确认融合流程，可用单轮试运行（独立目录），它会训练1轮并验证2张整图：

```bash
python potsdam_frequency_experiment/train.py --variant frequency --epochs 1 --val-every 1 --out outputs/training/potsdam_quick_frequency_trial
```

如显存不足，先确认batch-size=1；AMP默认开启。双B2本身仍有两套编码器；缩小数据量减少训练时间，并不等价于降低单batch显存。

## 第二步：公平比较两组模型

依次运行，避免同卡并发：

```bash
python potsdam_frequency_experiment/train.py --variant baseline
python potsdam_frequency_experiment/train.py --variant frequency
```

默认30轮、batch=1、累积4、每5轮及最后一轮验证。每轮192样本，48次计划优化器更新；30轮共1440次，前5%线性warmup，随后poly衰减。AMP溢出跳过更新时调度器也跳过。初始共享预训练参数LR=3e-5，适配器/新分类器/频率模块LR=3e-4，AdamW、weight_decay=0.01、梯度裁剪1.0。损失 `CE + 0.5×Lovasz`：CE在512分辨率计算，Lovasz沿用仓库在原始1/4 logits计算的方式。这个小预算采用累积4，区别于旧完整训练默认累积16；本包两组保持相同。

保存best.pt、last.pt、train_log.jsonl、run_config.json、dataset_protocol.json和每次验证指标。best仅按两张验证图聚合mIoU选择，不使用测试集。若两条曲线30轮仍上升，后续应双方等预算重跑或扩展方案，不能只给一个模型增加训练量。

恢复中断：

```bash
python potsdam_frequency_experiment/train.py --variant frequency --resume outputs/training/potsdam_quick_frequency_seed20260917/last.pt
```

恢复包含优化器、调度器、AMP scaler，从下一个epoch开始。每轮独立固定随机种子，支持按已保存的epoch边界复现；不支持batch中途精确恢复。恢复时epochs等核心配置必须与原运行相同；若改变总轮数，学习率方案也会改变，应作为新实验另起输出目录。

## 第三步：独立整图验证

```bash
python potsdam_frequency_experiment/validate.py \
  --weights outputs/training/potsdam_quick_frequency_seed20260917/best.pt \
  --split val --save-predictions
```

无需在验证命令中指定variant，代码从权重内记录的训练配置重建对应结构。baseline权重同样可验证。

验证严格检查协议SHA256与checkpoint一致。输出默认为 outputs/evaluation/训练目录名_val_best/；再次运行应指定新的 --out，避免覆盖结果。每张6000图使用256个窗口，两张共512。使用磁盘临时memmap累加六类float32 logits，单图约0.9GB临时空间；计数矩阵在CPU内存约72MB。临时文件在每张结束后清理；预测图按需保存，最终GPU上不保留整张图logits。

输出metrics.json包括：全split混淆矩阵汇总的六类mIoU、逐类IoU、OA、Kappa；每张图指标；植被GT/预测覆盖度、带符号误差和绝对误差（百分点）；植被二分类混淆矩阵（行GT，列预测；0非植被、1植被）。vegetation_MAE_pp为逐图覆盖度绝对误差平均。不能只看平均误差，还应检查每张图是否满足±10个百分点要求。六类矩阵顺序：不透水面、建筑、低矮植被、树木、汽车、杂类；植被=2+3。

零union类别的IoU记null，mIoU排除这类；不是把缺失类别当0。所有unknown标签颜色触发错误，避免把错误GT悄悄映射成背景。

模块选定后才运行原14张测试：将 --split val 改成 --split test。完整测试共3584窗口，明显比小验证慢；不要反复以测试结果筛选模型。

## 已完成的本地验证与限制

本地CPU合成小模型检查通过：共享模块初始化一致；双编码器前向/反向；频率门控、滤波器非零梯度；FFT高低频重构和常量输入；滑窗融合、混淆矩阵、覆盖度及预测导出；checkpoint重载一致；真实协议metadata指纹；192样本加载器；候选坐标SHA256与仓库实现一致。

没有访问你的服务器原始TIFF、本地ADE20K权重或4090，没有完成真实B2 CUDA训练。selfcheck.py是离线CPU合成检查，不代表服务器Smoke Test已通过。可在本机现有环境执行 python potsdam_frequency_experiment/selfcheck.py；通常无需安装新包，依赖torch、transformers、numpy、tifffile，预测可视化用Pillow。
