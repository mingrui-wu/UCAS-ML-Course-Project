# Gemini 智算平台 Video-QA Demo 操作指南

本指南介绍如何使用学校 Gemini 智算平台，运行基于 Qwen2.5-VL-3B 的 Video-QA 模型微调 Demo。

课程已经准备好所需的 Docker 镜像、预训练模型和示例数据。同学们只需要创建自己的项目、启动开发环境，并执行提供的训练命令，即可完成一次 QLoRA 微调实验。

### 课程提供的资源

| 资源 | 名称或配置 |
| --- | --- |
| 平台入口 | SEP 门户 → 常用系统 → 智算平台 |
| Docker 镜像 | `ml-course-llamafactory` |
| 镜像 Tag | `20261104194308` |
| 预训练模型 | `Qwen2.5-VL-3B-Instruct` |
| GPU | `H20-8G-small` |
| 示例数据 | `gemini_demo/data/` |

以上镜像和模型已共享至课程空间，**无需自行下载模型、安装 LLaMA-Factory 或配置 Python 环境**。

## 1. 获取 Demo 代码

1. 打开课程 GitHub 仓库，点击 **Code → Download ZIP**，下载并解压仓库。
2. 找到其中的 `gemini_demo/` 文件夹，将整个文件夹重新压缩为 `gemini_demo.zip`。
3. 保留 ZIP 文件，后续创建 Gemini 项目时上传。

请确保压缩包内包含以下结构：

```text
gemini_demo/
├── configs/
├── scripts/
├── data/
└── README.md
```

其中 `data/` 已包含用于 Demo 的短视频及 Video-QA 数据，不需要另外上传数据集。

## 2. 登录平台并创建项目

1. 登录学校 **SEP 门户**，通过 **常用系统 → 智算平台** 进入 Gemini。
2. 确认当前位于课程所在的空间。
3. 在左侧点击 **项目 → 创建项目**。
4. 按照下表填写项目配置：

| 配置项 | 选择 |
| --- | --- |
| 项目名称 | 自行命名，例如 `videoqa-学号` |
| 代码 | 本地代码 |
| 镜像 | `ml-course-llamafactory` |
| 镜像 Tag | `20261104194308` |
| 预训练模型 | `Qwen2.5-VL-3B-Instruct` |
| 数据 | 不需要额外挂载 |
| 公开性 | 私有 |

5. 点击 **创建**，在上传代码界面选择第 1 步准备的 `gemini_demo.zip`，完成上传。

创建成功后，可以在项目的 **代码** 页面查看已上传的文件。

注意：每位同学使用自己的项目进行实验，不需要修改教师提供的共享镜像和预训练模型。

## 3. 启动开发环境

1. 进入刚创建的项目，点击左侧 **开发 → 初始化开发环境实例**。
2. 选择以下配置：

| 配置项 | 选择 |
| --- | --- |
| GPU | `H20-8G-small`，1 卡 |
| 镜像 | 创建项目时选择的课程镜像 |
| 开发者工具 | JupyterLab |
| SSH / 开放端口 | 保持默认 |
| 最长运行时间 | 建议 1 小时 |

3. 确认配置后，点击 **确定**，启动开发环境。
4. 等待环境状态变为 **运行中**。
5. 点击 **进入开发环境**，打开 **JupyterLab**。

首次启动时，平台可能需要一定时间拉取镜像。启动完成后，在 JupyterLab 中打开一个 **Terminal**，执行后面的命令。

## 4. 进入 Demo 目录

Gemini 会将项目代码挂载至 `/gemini/code`。

如果上传后仍然是 `gemini_demo.zip` 文件，先在 Terminal 中解压：

```bash
cd /gemini/code
python -m zipfile -e gemini_demo.zip .
```

如果已经能看到 `gemini_demo/` 文件夹，可以跳过解压。

然后进入工程：

```bash
cd /gemini/code/gemini_demo
```

设置训练数据和模型位置：

```bash
export DATA_DIR="$PWD/data"
export MODEL_DIR=/gemini/pretrain
```

至此，训练所需的代码、模型和示例数据均已准备好。

## 5. 运行 Video-QA QLoRA 微调

本 Demo 使用 **Qwen2.5-VL-3B-Instruct** 作为基座模型，采用 4-bit QLoRA 进行参数高效微调。

示例数据包含 12 段短视频，其中 27 条 Video-QA 样本用于训练，9 条用于后续测试。数据仅用于验证完整训练流程，不代表真实的空间推理任务难度。

### 5.1 快速训练测试

先执行一个仅包含 2 个优化步骤的训练任务：

```bash
bash scripts/train.sh qlora_smoke_8g
```

正常情况下，终端会输出训练 Loss、进度信息，并在结束时显示：

```text
Training completed.
```

训练结果保存在：

```text
outputs/qlora_smoke_8g/
```

### 5.2 完整训练一轮

快速测试成功后，执行：

```bash
bash scripts/train.sh qlora_one_epoch_8g
```

这次会使用全部 27 条训练样本，完成 1 个 Epoch。

使用课程提供的默认配置时，预计完成 **7 个 optimizer steps**。训练时间取决于实际 GPU 资源及平台运行状态。

训练结束后，模型的 LoRA Adapter 将保存在：

```text
outputs/qlora_one_epoch_8g/
```

## 6. 检查训练结果

运行：

```bash
python scripts/benchmark.py outputs/qlora_one_epoch_8g
```

检查输出中是否显示训练完成，并且 `Global steps` 为 `7`。

还可以检查保存的模型文件：

```bash
ls -lh outputs/qlora_one_epoch_8g/
```

其中应包含：

```text
adapter_config.json
adapter_model.safetensors
trainer_state.json
```

`adapter_model.safetensors` 是本次微调产生的 LoRA 权重，并不包含完整的基座模型。

**如果训练正常结束，并成功生成 Adapter 文件，即视为完成本次 Demo。**

## 7. 结束实验

完成后，请返回 Gemini 平台的 **开发** 页面，主动停止开发环境，避免继续占用 GPU 算力资源。

本 Demo 主要用于熟悉学校智算平台的使用方式，以及体验大语言模型微调的基本流程。后续课程项目将涉及训练数据构建、模型微调和性能评估，具体要求请参考课程 Tutorial 及其他模块说明。