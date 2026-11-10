# Potsdam 频率模块快速实验协议 v1

本包基于你上传的 potsdam.zip 生成，不包含原始 TIFF、模型或训练脚本。新协议用于比较原基线与频率模块，筛选有效思路后再恢复完整协议。原数据和原协议不需要删除或覆盖。

## 1. 固定划分

| 用途 | 原协议 | 快速协议 |
|---|---:|---:|
| 训练大图 | 18 | 6 |
| 验证大图 | 6 | 2 |
| 测试大图 | 14 | 14（封存，模块筛选时不运行） |
| 暂不使用的开发集大图 | 0 | 16 |
| 每张训练图每轮随机裁剪 | 64 | 32 |
| 每轮训练切片 | 1152 | 192 |
| 一次完整验证窗口 | 1536 | 512 |

训练：2_12、4_12、5_10、6_9、6_11、7_9。

验证：4_10、5_11。

测试仍是原14张；详见 protocol/test.txt。unused.txt 中的16张本阶段不进入任何 DataLoader。

训练6张只从原训练18张中选择，验证2张只从原验证6张中选择。使用已有标签类别像素统计，穷举候选组合，先最小化六类别占比最大绝对偏差，以平均绝对偏差作为次级排序，不使用模型结果或测试标签选图。每个集合均覆盖六类别，但不保证每个512切片都包含六类。

训练子集与原训练集的最大类别占比偏差为0.2303个百分点；验证子集为4.1216个百分点。因此，小验证集只适合初步筛选；完整数据效果仍需复核。已看过的小验证集不应再当作独立测试集。

## 2. 保持的处理方式

- 图像为 RGBIR，通道顺序 R/G/B/NIR，六分类及原颜色映射不变。
- 训练裁剪512×512，保留原D4增强、cat_max_ratio=0.75、最多10次候选及原回退方式。
- 保留原数据随机种子20260917，不缩小图像分辨率，不改变频率模块观察的像素尺度。
- 验证按6000×6000整图滑窗：512窗口、384步长、末端补窗口，每张256个窗口，平均logit融合，不启用TTA。
- 主指标由全验证集像素混淆矩阵汇总后算六类mIoU，不平均切片mIoU。建议同时记录逐类IoU、OA、每张植被覆盖度误差，植被为类别2+3。覆盖度误差以百分点记录。
- GT继续只使用 labels_manifest.json 的 canonical路径。尤其4_12需要使用5_Labels_for_participants版本。

## 3. 文件用途

- protocol/：已经生成的小规模协议，可独立放到项目新目录。
- source_protocol/：上传的四个原始JSON，供重现与追溯。
- build_quick_protocol.py：重现划分，并可在服务器上重新计算六张训练影像的NIR均值/标准差。
- protocol/quick_protocol_report.json：选择方法、集合和元数据检查结果。

labels_manifest.json 仍保留38张图的canonical标签索引。它是路径字典，不是训练名单。训练代码必须以 tile_split.json 的 splits.train/val/test 为准；不能遍历所有manifest记录生成训练集。新 tile_split.json 的 tiles 中，未使用影像标为 unused。

旧协议的裁剪审计、采样日程指纹已失效；新协议未冒用旧PASS。滑窗坐标本身未改变。元数据核验通过不等于服务器原始文件已重新核验。

## 4. 在Ubuntu服务器重新生成（推荐）

把压缩包解压到项目根目录，使下列脚本存在：

/home/panyao/ty/MRISE/segformer_potsdam/potsdam_frequency_quick_v1/build_quick_protocol.py

终端执行：

```bash
conda activate mine-seg
cd /home/panyao/ty/MRISE/segformer_potsdam
python potsdam_frequency_quick_v1/build_quick_protocol.py \
  --out data/processed/potsdam_frequency_quick_v1 \
  --rgbir-dir data/raw/potsdam/_expanded/4_Ortho_RGBIR
```

最后一个路径使用你此前记录的展开目录；若你把数据放到了其他位置，请替换为实际包含 top_potsdam_2_12_RGBIR.tif 等文件的4_Ortho_RGBIR目录。脚本会检查六张输入图的尺寸、通道和类型。需要现有环境中的numpy、tifffile；不会安装软件。输出目录必须尚不存在，以免覆盖已有实验。

本命令只读取6张训练TIFF并输出新元数据，无须移动或复制6000×6000影像，也无须提前导出所有切片。NIR均值、标准差使用6张训练图的全部像素重新计算。基线和频率模型必须共用本次生成的同一份协议。

如果只想重现随包协议、不读取影像：

```bash
python potsdam_frequency_quick_v1/build_quick_protocol.py \
  --out data/processed/potsdam_frequency_quick_v1
```

不提供 --rgbir-dir 时，使用原18张训练图计算的固定NIR参考统计（mean=0.3854390870128298，std=0.14614233541588012；均在除255之后）。它没有使用原验证或测试图，但含本次未使用的训练影像，因此不是严格“仅六张训练图估计”的统计。新协议已明确标注这个差异。推荐采用上面的重算方式；不要在基线和频率模型之间切换统计。

## 5. 让训练脚本真正使用新协议

上传包只有协议，没有当前训练代码，不能确认你现有训练脚本的命令参数，也不能保证它自动读取新增的 experiment_plan 字段。以下配置需要落到实际训练脚本中；只生成JSON不会自动改变训练。

1. 将训练脚本的协议目录改为 data/processed/potsdam_frequency_quick_v1，保持raw数据根目录指向原数据。split文件和dataset_protocol必须同时来自新目录，不能混用旧文件。
2. 确认训练名单读自 tile_split.json 的 splits.train，验证名单读自 splits.val；每轮每图32次，合计192个训练样本。覆盖脚本中可能写死的18图、64次、1152步等常量。
3. 规范化读新 dataset_protocol.json 中 normalization；类别权重若由训练标签计算，需按新训练6图重新统计，并在两个模型间固定。
4. 首轮推荐30 epochs，每5轮完整验证一次，并确保最后一轮验证。按相同验证mIoU选best；test不参与选模型。30轮是起始预算，不保证收敛，若两条验证曲线仍明显上升，应同等增加预算。
5. 学习率调度的总步数要按新DataLoader长度重算。batch_size、梯度累积保持两个模型相同；每轮优化器更新数依据实际batch数及累积逻辑确定，不能把192样本直接当192次更新。
6. 新建不同输出目录，例如 outputs/training/potsdam_quick_baseline_seed20260917 与 outputs/training/potsdam_quick_frequency_seed20260917。不要覆盖原训练输出，也不要只给频率模型继承额外训练过的权重。

训练启动日志至少应核对：

```text
protocol = .../data/processed/potsdam_frequency_quick_v1
train_tiles = 6
val_tiles = 2
crops_per_tile_per_epoch = 32
train_samples_per_epoch = 192
crop = 512 x 512
val_windows_per_tile = 256
val_windows_total = 512
```

先各跑一次单batch Smoke Test，确认图像通道、标签ID、logit插值、loss为有限值、反向传播和频率分支参数梯度正常；随后进行一次完整小验证，确认全图融合及指标聚合可用。若原代码有冻结协议的指纹校验，应对新协议重新审计，不能简单把状态改成PASS绕过。

## 6. 比较顺序和时间预期

先在本快速协议上重新训练基线，再用完全相同条件训练频率模块。原18图训练结果不能直接作为本6图协议的消融对照。先用同一随机种子作初步判断，有改善后再考虑多种子与完整协议复核。

同轮数、batch和模型开销相近时，训练样本工作量为原来的1/6，一次验证窗口量为原来的1/3。真实耗时还包括数据加载、频率分支、验证与保存，不能据此保证6倍提速。每5轮验证一次可进一步降低验证成本；最终比较要保留相同验证频率。

小数据结果只说明该预算下的表现。频率模块没有改善时，也要结合是否收敛、类别IoU及边界接缝现象判断，不能仅凭一次小样本实验认定模块无效。
