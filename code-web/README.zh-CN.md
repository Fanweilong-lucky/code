# 目标感知全局–局部红外–可见光图像配准

对应论文：**Target-aware global–local infrared–visible image registration for small unmanned aerial vehicle targets**。

本目录是独立的**纯代码发布包**，包含 V32 配准网络、分阶段训练、推理、点 TRE 评价、配置模板和合成数据测试。

## 发布范围

- 不包含数据集、样本图、标注、真实样本编号或训练/验证/测试名单。
- 不包含已训练权重、实际校准文件、逐样本变换或实验输出。
- 不包含原实验 Git 历史、本机绝对路径或旧的100对样本包。
- 代码不会自动下载数据或模型。数据集受保密限制。

公开代码并不意味着仅凭本目录即可复现论文数值：仍需另行获得授权数据及对应模型、校准参数，或在授权数据上重新训练。

## 环境与使用

环境、命令和参数见英文 [README](README.md)。本次本地检查使用 PyTorch 2.8.0+cu126、NumPy 2.0.2、OpenCV 4.13.0；运行时使用 CPU，没有重新训练论文模型。

先安装适合计算机硬件的 PyTorch 2.8.0，再安装 `requirements.txt`。无需数据即可检查：

```bash
python scripts/verify_code_only.py
python -m unittest discover -s tests -v
```

测试图像、假标注和随机模型均在运行时临时生成，不是研究数据，测试后自动清理。

## 训练与推理

真实数据、私人配置及运行输出均应保存在仓库外。数据接口说明在 [docs/DATA_INTERFACE.md](docs/DATA_INTERFACE.md)。

1. 复制配置模板到仓库外，填写自己获得授权的数据路径。
2. 运行 `scripts/train_v31.py` 完成前三阶段训练。
3. 运行 `scripts/train_v32.py`，从已训练 V31 初始化并仅训练目标局部相关分支。
4. 使用生成的 V32 权重和相应校准文件运行 `scripts/register.py`。
5. 使用私有目标中心标注运行 `scripts/evaluate.py`。

`core/v31.py` 是 V32 所需的基础网络和训练依赖，不能因文件名含 V31 就删除。代码保留原模型结构、损失、几何变换与训练参数；推理入口改为显式传入自己的数据、权重和校准路径。

推理需要可见光目标框先验，采用 YOLO 格式，选择最大的类别0目标框。模型计算尺寸为480×384，输出为960×768灰度红外配准图。推理不读取人工红外中心或人工配准真值。

## 上传 GitHub

只上传本目录内的内容，不要上传原实验目录或旧“公开”样本目录。文件排除规则和纯代码检查脚本已提供；可启用本地 Git 提交钩子。GitHub 自动检查只能在提交到平台后运行，不能撤回已经上传的敏感内容，因此上传前必须先在本地检查。

```bash
git init
git config core.hooksPath .githooks
git add .
python scripts/verify_code_only.py --tracked
```

尚未代作者选择开源许可证，亦未虚构 DOI、接收状态或仓库地址。正式发布前应由作者确定代码授权方式并填写核实后的文献信息。
