# 基于 LeRobot 的 ACT 跨环境泛化实验

本仓库对应期末作业题目二，包含：

- `splitA` 训练 `A-only`
- `splitABC` 训练 `ABC-joint`
- `splitD` zero-shot 离线评测

## Environment

根目录提供提交版环境文件 [`environment.yml`](environment.yml)。

创建环境：

```bash
conda env create -f environment.yml
conda activate lerobot
```

根据机器单独安装匹配的 PyTorch CUDA 版本，例如：

```bash
pip install torch torchvision torchaudio
```

随后安装 LeRobot：

```bash
cd /path/to/lerobot
pip install -e .
```

## Data

本仓库不分发 CALVIN 原始数据。实验使用的 LeRobot 格式数据可从 Hugging Face 下载：

- [xiaoma26/calvin-lerobot](https://huggingface.co/datasets/xiaoma26/calvin-lerobot/tree/main)

该数据集公开提供 `splitA`、`splitB`、`splitC`、`splitD` 四个目录。README 中的命令默认你已将数据下载到本地，例如：

```text
/path/to/calvin-lerobot/
├── splitA
├── splitB
├── splitC
├── splitD
└── splitABC
```

其中 `splitABC` 用于多环境联合训练，需要由 `splitA`、`splitB`、`splitC` 在本地整理得到。

训练与评测脚本默认依赖每个 split 下的：

- `data/`
- `meta/tasks.parquet`

关键字段名：

- `observation.images.image`
- `observation.images.wrist_image`
- `observation.state`
- `action`

## Train

### `A-only@2000`

```bash
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitA \
  --dataset.root=/path/to/calvin-lerobot/splitA \
  --batch_size=32 \
  --steps=2000 \
  --num_workers=4 \
  --save_freq=20000 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitA_steps2000
```

### `ABC-joint@2000`

```bash
CUDA_VISIBLE_DEVICES=3 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitABC \
  --dataset.root=/path/to/calvin-lerobot/splitABC \
  --batch_size=32 \
  --steps=2000 \
  --num_workers=4 \
  --save_freq=20000 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitABC_steps2000
```

### `A-only@10000`

```bash
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitA \
  --dataset.root=/path/to/calvin-lerobot/splitA \
  --batch_size=32 \
  --steps=10000 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitA_steps10000
```

### `ABC-joint@10000`

```bash
CUDA_VISIBLE_DEVICES=3 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitABC \
  --dataset.root=/path/to/calvin-lerobot/splitABC \
  --batch_size=32 \
  --steps=10000 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitABC_steps10000
```

## Test

### 视觉偏移与 replan

```bash
python work/eval_shift.py \
  --split-root splitA=/path/to/calvin-lerobot/splitA \
  --split-root splitD=/path/to/calvin-lerobot/splitD \
  --repo-id splitA=calvin_splitA \
  --repo-id splitD=calvin_splitD \
  --checkpoints \
    A10000=checkpoints/A_only_10000/pretrained_model \
    ABC10000=checkpoints/ABC_joint_10000/pretrained_model \
  --samples-per-family 128 \
  --replan-pairs-per-family 64 \
  --batch-size 32 \
  --device cuda:0 \
  --seed 0 \
  --output outputs/results/visual_shift_with_replan_10000.json
```

### 离线动作误差

```bash
python work/eval_metrics.py \
  --dataset-root /path/to/calvin-lerobot/splitD \
  --repo-id splitD \
  --batch-size 32 \
  --max-samples 6400 \
  --device cuda:0 \
  --output outputs/results/eval_task2_metrics_full.json \
  --checkpoints \
    A2000=checkpoints/A_only_2000/pretrained_model \
    ABC2000=checkpoints/ABC_joint_2000/pretrained_model \
    A10000=checkpoints/A_only_10000/pretrained_model \
    ABC10000=checkpoints/ABC_joint_10000/pretrained_model
```

## Files

- 环境文件：`environment.yml`
- 视觉偏移与 replan 脚本：`work/eval_shift.py`
- 离线动作误差脚本：`work/eval_metrics.py`
- 训练曲线整理脚本：`work/extract_wandb_history.py`
- 2000 step 评测汇总脚本：`work/run_task2_rerun_eval_suite.py`
