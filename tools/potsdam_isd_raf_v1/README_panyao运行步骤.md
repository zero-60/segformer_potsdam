# Potsdam双B2 + ISDNet RAF适配实验 v1

## 范围与来源

ISDNet是完整超高分辨率分割框架，不是单个插件。本包只提取并适配其关系感知融合（Relation-Aware Fusion，RAF），比较现有双B2基线与“基线＋RAF”，不叠加FFT模块。

官方来源：https://github.com/cedricgsh/ISDNet

核对提交：7341335677fd6201457b321e1e6368feb7ad647e。

参考类：mmseg/models/decode_heads/isdhead.py中的ChannelAtt、RelationAwareFusion；third_party中附Apache-2.0许可与原始文件供核对，不作为运行依赖。

官方完整ISDNet包括低分辨率深分支、高分辨率浅分支、拉普拉斯残差、两级RAF及训练辅助任务。本包未复现这些完整结构，也没有超分辨率重建或结构蒸馏损失。这里的深、浅特征来自现有双B2的第四级与第一级，不是论文的两条图像分辨率分支。因此不能将本实验称为完整ISDNet，也不能引用论文的FPS作为本实现速度。

## 模型连接

RGB与NIR各自经过B2，四个尺度沿用：

F_i = RGB_i + 1.0 × Adapter_i(NIR_i)。

取高分辨率F_1（512输入时128×128、64通道）和深层F_4（16×16、512通道）。先分别投影到256和128通道，再送入RAF。RAF分支各自产生特征与全局平均池化的通道描述子；将128通道分16组，构造16×16关系矩阵，通过两个MLP修正各分支通道注意力，再上采样深层特征与浅层特征融合。经输出映射回64通道，以可学习残差加入F_1。其余尺度不变，最后进入原SegFormer解码头。

残差强度为tanh(scale)，scale初始0.1，输出投影采用正常初始化；官方RAF内部g1/g2保持0初始化。第一次反向传播关系MLP可能为零梯度，g1/g2更新后才开始参与，这是官方零初始关系系数的结果，并非模块断开。

移植调整：以PyTorch Conv+GroupNorm+ReLU替代mmcv ConvModule/SyncBN，使单GPU、batch=1以及池化后1×1描述子能够正常训练。这里只用一级RAF，保留原解码头，外加输入输出投影及残差连接；这些区别已在代码注明。

模型仍只看到512×512裁剪，无法获取切片外地物。若目标是解决6000整图的全局上下文缺失，需要另行改造输入与训练协议；本次仅判断浅深层关系融合是否有效，不能宣称已经解决接缝。

## 安装：全部代码在tools下

把包内tools目录合并到：

/home/panyao/ty/MRISE/segformer_potsdam/tools

不覆盖现有pf_*或频率实验代码。本包使用独立isd_*文件，可独立运行。

默认路径：

| 项目 | 路径 |
|---|---|
| 项目根目录 | /home/panyao/ty/MRISE/segformer_potsdam |
| 协议 | data/processed/potsdam_frequency_quick_v1 |
| RGBIR | data/raw/potsdam/_expanded/4_Ortho_RGBIR |
| GT根目录 | data/raw/potsdam |
| 本地B2权重 | /home/panyao/ty/MRISE/hf-cache/segformer-b2-finetuned-ade-512-512 |
| 缓存 | data/processed/potsdam_frequency_fast_cache |
| RAF输出 | outputs/training/potsdam_isd_raf_v1_seed20260917 |
| 配对基线输出 | outputs/training/potsdam_isd_baseline_v1_seed20260917 |

继续使用服务器已经按6张训练图重算NIR统计的协议。若服务器缓存已存在且来源匹配，会直接复用；从其他机器复制过来的缓存因绝对路径/时间戳变化可能重建，这是预期行为。不需要重新划分数据。

依赖现有mine-seg中的torch、transformers、numpy、tifffile、Pillow，不需要安装mmcv/mmseg，不需要下载ISDNet的STDC权重，不修改当前Conda环境。预训练仅加载本地文件。

## 1. 环境与离线合成检查（可选）

```bash
conda activate mine-seg
cd /home/panyao/ty/MRISE/segformer_potsdam
python tools/selfcheck_potsdam_isd_raf.py
```

selfcheck用随机小模型和合成输入，检查模块连接，不代表真实数据通过。

## 2. 缓存准备和真实数据Smoke Test

```bash
python tools/train_potsdam_isd_raf.py --prepare-cache-only
python tools/train_potsdam_isd_raf.py --variant raf --smoke-test
```

预期6张训练、2张验证、每轮192样本；输出RAF gradient_abs_sum>0及SMOKE PASS。新分类器150→6的加载提示正常。缓存首次构建可能耗时；随后复用。

## 3. 从头训练RAF

```bash
python tools/train_potsdam_isd_raf.py --variant raf
```

默认30轮，batch=1，梯度累积4，workers=2，CPU线程4，AMP开启，6张训练每图32次裁剪，损失CE+0.5Lovasz；预训练参数LR=3e-5、新适配器/分类器/RAF参数LR=3e-4。Warmup占总更新数5%，随后poly；每5轮及最后一轮整图验证。按验证全局混淆矩阵mIoU保存best.pt，并每轮保存last.pt。

独占显卡依次运行实验。如果worker异常，可用 --workers 0；验证显存不足可用 --eval-batch-size 1（默认2）。输出目录必须为空，或明确使用resume恢复同一实验。

先不要加载旧频率模型best.pt作为RAF初始化；两者结构不同，也会改变消融训练预算。

## 4. 在panyao服务器跑配对基线

```bash
python tools/train_potsdam_isd_raf.py --variant baseline
```

这组与RAF实验共享初始化、协议、采样、损失和训练预算。Windows上既有基线68.816%只能作为历史参考，不宜用不同机器的软件/硬件差异下0.0x个百分点变化判断收益。推荐比较本脚本同一环境下的两组结果。

## 5. 剩余时间显示

训练每16个batch输出，可用 --log-every 8调频率：

- epoch_eta：当前轮剩余训练时间。
- train_eta_excl_val：全部剩余训练时间，不含验证。
- total_eta_est：剩余训练＋后续计划验证的估计时间；首次验证完成前显示pending_first_validation，此后使用已完成验证的平均耗时估计。未包含权重保存、启动与缓存时间；运行速度变化时会调整。
- data_wait_ms、step_wall_ms：每轮累计平均数据等待/步骤墙钟耗时，不等同于独立CUDA核计时。

每轮末尾还显示RAF residual_scale、relation_g1、relation_g2，并写进train_log.jsonl。不能只根据系数大小断言模块有效，还需看验证提升及预测图。

## 6. 验证与续训

```bash
python tools/validate_potsdam_isd_raf.py \
  --weights outputs/training/potsdam_isd_raf_v1_seed20260917/best.pt \
  --split val --save-predictions \
  --out outputs/evaluation/potsdam_isd_raf_v1_val
```

验证从checkpoint配置自动识别baseline/raf，严格核对协议SHA256。仍按512窗口/384步长融合6000整图，输出六类mIoU、OA、Kappa、逐图植被覆盖度误差和混淆矩阵。融合默认CPU RAM累加，不把整图logits放GPU。test保持封存，不用于筛模块。

断点恢复：

```bash
python tools/train_potsdam_isd_raf.py \
  --variant raf \
  --resume outputs/training/potsdam_isd_raf_v1_seed20260917/last.pt
```

仅从已保存epoch边界恢复；总轮数、学习率、variant等必须一致。不能使用旧频率checkpoint恢复RAF。重启后total_eta_est要重新观察一次验证才估计验证耗时。

## 测试范围

本地CPU小模型合成测试检查：共享初始化与旧baseline输出完全一致；batch1 GroupNorm；前向/损失/反向；RAF深浅投影与关系MLP梯度；不同推理batch近似一致；权重保存重载。真实服务器原始TIFF、B2权重与4090显存/耗时仍需要上述Smoke Test和训练日志验证。本包不承诺精度提升。
