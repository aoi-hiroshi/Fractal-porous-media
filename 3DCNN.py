"""经典三维卷积网络：二值数字岩心 -> 渗透率回归（PyTorch >= 2.4）。

默认模型：三组 [Conv3d(3, stride=1, padding=1) -> BatchNorm3d -> ReLU
                    -> MaxPool3d(3, stride=1, padding=0)]，通道 1/8/16/32；
          空间均值汇聚 -> Linear(32, 32) -> ReLU -> Dropout -> Linear(32, 1)。
局部卷积、池化的核均为 3×3×3，步长均为 1，没有隐藏降采样或输入缩放。
100³ 输入经过三个池化层得到 98³、96³、94³，最后的空间均值是回归读出，
不是另一个局部池化核。stride=1 会增加内存和计算量，也限制多尺度感受野。

数据：默认读取本文件上一级目录下 Fractral/manifest.csv、gt.csv 和 dataset。
RAW：uint8，C-order [z, y, x]；原文件 0=孔隙，网络输入转换成 1=孔隙。
标签：默认 log10((kx+ky+kz)/3)，单位 mD；也可用 --target xyz 回归三方向。
仅按 block_number 对齐，禁止用 gt 中历史遗留的 original_block_number 重映射。
空间划分沿用 XGB：z=0..5 训练，6..7 验证，8..9 测试，当前为 597/197/198。
本脚本只用训练集拟合，验证集早停；不同于 XGB.ipynb 最后 train+val 重拟合，
严谨比较时应统一两者的最终拟合协议，不能只看相同的测试集。
无分形辅助输入、无 PINN、无数据增强、无测试集参与调参。

在安装 numpy、pandas、torch 的环境中运行（从任意工作目录均可）：
    python 3DCNN.py --mode check              # 默认：数据审计和一个真实样本前向/反向
    python 3DCNN.py --mode train              # 正式训练，结束后仅评估最佳模型
    python 3DCNN.py --mode train --target xyz # 可选的三方向回归
    python 3DCNN.py --mode test --checkpoint /absolute/path/to/best.pt
未指定 --mode 时只检查，不启动完整训练。运行产物写到本文件旁 runs 的独立目录。
如需 GPU，请按 https://pytorch.org/get-started/locally/ 安装适合驱动的 CUDA 版本。

本文件独立实现常规 Conv/ReLU/Pool/FC 结构，并非复现某篇论文的训练结果。
层参数参考官方文档（没有复制第三方模型源码）：
https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time
import uuid
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "缺少 PyTorch；请在运行本脚本的 Python 环境中安装 torch >= 2.4。\n"
        "GPU 安装命令请见 https://pytorch.org/get-started/locally/"
    ) from exc


K_COLUMNS = [f"permeability_{axis}_mD" for axis in "xyz"]
STATUS_COLUMNS = [f"status_{axis}" for axis in "xyz"]
SPLITS = ("train", "validation", "test")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Fractral"


# ==================== 一、经典 3D CNN ====================
class CNN3D(nn.Module):
    """输入 [B, 1, Z, Y, X]，输出 [B, 1] 或 [B, 3]，值为标准化的 log10(k)。"""

    def __init__(self, channels=(8, 16, 32), out_features=1, dropout=0.2):
        super().__init__()
        if len(channels) != 3 or any(c <= 0 for c in channels):
            raise ValueError("channels 必须包含三个正整数。")
        if out_features not in (1, 3) or not 0 <= dropout < 1:
            raise ValueError("输出维数必须为 1 或 3，dropout 必须在 [0, 1) 内。")
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=1, padding=0),
            ))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.regressor = nn.Sequential(
            nn.Linear(channels[-1], 32), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(32, out_features),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 8:
            raise ValueError("输入必须为 [B, 1, Z, Y, X]，且三个空间尺寸均 >= 8。")
        for block in self.blocks:
            x = block(x)
        # 全局平均读出避免 Flatten 后出现数千万乃至数亿个全连接参数。
        x = x.mean(dim=(2, 3, 4))
        return self.regressor(x)


# ==================== 二、数据读取与空间划分 ====================
def read_table(path):
    frame = pd.read_csv(path, dtype={"block_number": str, "original_block_number": str})
    ids = frame["block_number"].str.strip()
    if ids.isna().any() or not ids.str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"{path.name}: block_number 有缺失或非整数编号。")
    # 保留四位编号；先去除多余前导零，使 1、0001 被识别为同一个 ID。
    frame["block_number"] = ids.map(lambda value: f"{int(value):04d}")
    if frame["block_number"].duplicated().any():
        raise ValueError(f"{path.name}: block_number 重复，不能一对一对齐。")
    return frame


def load_records(data_dir: Path, shape=(100, 100, 100)):
    """不读 features.csv，也不修改 CSV；清单负责 RAW 路径及原始空间位置。"""
    data_dir = data_dir.resolve()
    manifest = read_table(data_dir / "manifest.csv")
    gt = read_table(data_dir / "gt.csv")
    if set(manifest.block_number) != set(gt.block_number):
        raise ValueError("manifest 与 gt 的 block_number 集合不同，禁止静默丢样本。")
    # 原始编号仅用于记录差异，绝不参与合并。
    gt = gt.rename(columns={"original_block_number": "original_block_number_gt"})
    frame = manifest.merge(gt, on="block_number", validate="one_to_one")
    if "original_block_number_gt" in frame:
        mismatch = (frame.original_block_number != frame.original_block_number_gt).sum()
        if mismatch:
            warnings.warn(f"gt 的 original_block_number 有 {mismatch} 个历史差异；"
                          "已按 block_number 对齐，没有重新映射或删除样本。")
    numeric = frame[K_COLUMNS + STATUS_COLUMNS + ["porosity"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("标签、孔隙率或 LBM 状态包含缺失值/非有限值。")
    if (frame[K_COLUMNS] <= 0).any().any():
        raise ValueError("存在非正渗透率，不能直接取 log10；请核实标签，不自动裁剪。")
    if (frame[STATUS_COLUMNS] != 0).any().any():
        raise ValueError("存在非零 LBM 状态，必须先核实，不能默默当成有效标签。")
    if not frame.porosity.between(0, 1).all():
        raise ValueError("孔隙率必须在 [0, 1] 内。")
    coords = np.asarray(frame.block_index_zyx.map(json.loads).tolist())
    if coords.shape != (len(frame), 3) or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("block_index_zyx 必须是三个整数组成的 [z,y,x]。")
    if (coords < 0).any() or (coords > 9).any():
        raise ValueError("默认空间划分仅适用于 10×10×10 分块；请先明确新数据的划分。")
    if len(np.unique(coords, axis=0)) != len(frame):
        raise ValueError("manifest 存在重复的原始空间块坐标。")
    frame[["z_index", "y_index", "x_index"]] = coords
    frame["split"] = np.where(coords[:, 0] <= 5, "train",
                              np.where(coords[:, 0] <= 7, "validation", "test"))
    expected_bytes = int(np.prod(shape))  # uint8 每体素一字节。
    resolved_paths = []
    for raw_path in frame.raw_path:
        path = (data_dir / raw_path).resolve()
        if not path.is_relative_to(data_dir):
            raise ValueError(f"RAW 路径越出数据目录：{path}")
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"RAW 缺失或字节数不符（应为 {expected_bytes}）：{path}")
        resolved_paths.append(str(path))
    if len(set(resolved_paths)) != len(frame):
        raise ValueError("多个 block_number 指向同一 RAW 文件。")
    frame["resolved_raw_path"] = resolved_paths
    for split in SPLITS:
        if (frame.split == split).sum() < 2:
            raise ValueError(f"{split} 少于两个样本，无法按预期评估 R²。")
    return frame.sort_values("block_number").reset_index(drop=True)


def target_log10(frame, target):
    k = frame[K_COLUMNS].to_numpy(dtype=np.float64)
    # 必须先算 mD 的算术均值，再取对数，不是三个 log(k) 的平均。
    return np.log10(k.mean(axis=1, keepdims=True) if target == "mean" else k)


class RockDataset(Dataset):
    def __init__(self, frame, shape, target, mean, std):
        self.frame = frame.reset_index(drop=True).copy()
        self.shape = tuple(shape)
        self.targets = ((target_log10(frame, target) - mean) / std).astype(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        raw = np.fromfile(row.resolved_raw_path, dtype=np.uint8).reshape(self.shape)
        pore = raw == 0
        # 对每个实际读入的样本核对孔隙率，尽早发现文件/标签错配或黑白反转。
        if not np.isclose(pore.mean(), row.porosity, rtol=0, atol=5e-7):
            raise ValueError(f"{row.block_number}: RAW 孔隙率与 gt 不一致。")
        image = torch.from_numpy(pore.astype(np.float32)).unsqueeze(0)
        return image, torch.from_numpy(self.targets[index]), row.block_number


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(frame, args, mean, std, device, training=False):
    return DataLoader(
        RockDataset(frame, args.shape, args.target, mean, std),
        batch_size=args.batch_size, shuffle=training, drop_last=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )


# ==================== 三、训练、验证与评价 ====================
def amp_context(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def train_epoch(model, loader, optimizer, scaler, device, amp, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    seen = 0
    group_samples = 0
    for step, (images, labels, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_context(device, amp):
            prediction = model(images)
            loss = nn.functional.mse_loss(prediction, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失出现 NaN/Inf，请检查学习率或使用 --no-amp。")
        count = len(images)
        # 按真实样本数累计，最后不足 accumulation 个小批量也有正确的梯度权重。
        scaler.scale(loss * count).backward()
        total_loss += loss.item() * count
        seen += count
        group_samples += count
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_samples)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
    return total_loss / seen


@torch.inference_mode()
def predict(model, loader, device, amp):
    model.eval()  # 验证/测试不更新 BatchNorm，关闭 Dropout。
    all_targets, all_predictions, ids = [], [], []
    for images, labels, block_ids in loader:
        with amp_context(device, amp):
            prediction = model(images.to(device, non_blocking=True))
        all_predictions.append(prediction.float().cpu().numpy())
        all_targets.append(labels.numpy())
        ids.extend(block_ids)
    truth, estimate = np.concatenate(all_targets), np.concatenate(all_predictions)
    if not np.isfinite(estimate).all():
        raise FloatingPointError("预测出现 NaN/Inf。")
    return truth, estimate, ids


def scalar_metrics(truth, estimate):
    residual = estimate - truth
    ss_total = np.square(truth - truth.mean()).sum()
    mse = float(np.square(residual).mean())
    return {"R2": float(1 - np.square(residual).sum() / ss_total) if ss_total > 0 else None,
            "MAE": float(np.abs(residual).mean()), "MSE": mse, "RMSE": float(np.sqrt(mse))}


def evaluate_metrics(truth_log, predicted_log, target):
    """同时报告 log10 和 mD 尺度；MAPE 只在有物理意义的正值 mD 尺度计算。"""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        truth_md, predicted_md = 10.0**truth_log, 10.0**predicted_log
    names = ["mean"] if target == "mean" else list("xyz")
    result = {}
    for index, name in enumerate(names):
        raw = scalar_metrics(truth_md[:, index], predicted_md[:, index])
        raw["MAPE_percent"] = float(100 * np.mean(np.abs(
            (predicted_md[:, index] - truth_md[:, index]) / truth_md[:, index])))
        result[name] = {"log10": scalar_metrics(truth_log[:, index], predicted_log[:, index]),
                        "mD": raw}
    if target == "xyz":
        # 三方向输出的算术均值仅作派生评价，不混同于直接优化平均渗透率。
        mean_metrics = evaluate_metrics(np.log10(truth_md.mean(axis=1, keepdims=True)),
                                        np.log10(predicted_md.mean(axis=1, keepdims=True)), "mean")
        result["derived_mean"] = mean_metrics["mean"]
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def data_fingerprint(data_dir):
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("gt.csv", "manifest.csv")}


def new_run_dir(output_dir, prefix):
    # 每次运行独立建目录，避免覆盖之前的模型和结果。
    path = output_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def export_evaluation(model, frame, args, mean, std, device, amp, output_dir):
    reports = {}
    for split in ("validation", "test"):
        part = frame.loc[frame.split == split]
        _, prediction, ids = predict(model, make_loader(part, args, mean, std, device), device, amp)
        predicted_log = prediction.astype(float) * std + mean
        # 指标的真值采用原始 CSV 的 float64，而不是经 float32 标准化后再还原的数值。
        truth_log = target_log10(part, args.target)
        reports[split] = evaluate_metrics(truth_log, predicted_log, args.target)
        table = pd.DataFrame({"block_number": ids})
        for index, name in enumerate(["mean"] if args.target == "mean" else list("xyz")):
            table[f"true_{name}_log10"] = truth_log[:, index]
            table[f"pred_{name}_log10"] = predicted_log[:, index]
            table[f"true_{name}_mD"] = 10.0**truth_log[:, index]
            table[f"pred_{name}_mD"] = 10.0**predicted_log[:, index]
        table.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    write_json(output_dir / "metrics.json", reports)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


def train(model, frame, args, device):
    train_frame = frame.loc[frame.split == "train"]
    y_train = target_log10(train_frame, args.target)
    mean, std = y_train.mean(axis=0), y_train.std(axis=0)
    if (std <= 1e-12).any():
        raise ValueError("训练标签方差为零或过小。")
    train_loader = make_loader(train_frame, args, mean, std, device, training=True)
    val_loader = make_loader(frame.loc[frame.split == "validation"], args, mean, std, device)
    amp = args.amp and device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir = new_run_dir(args.output_dir, f"cnn3d_{args.target}_seed{args.seed}")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"torch_version": str(torch.__version__), "numpy_version": np.__version__,
                   "pandas_version": pd.__version__, "device_used": str(device),
                   "target_mean": mean.tolist(), "target_std": std.tolist(),
                   "data_fingerprint": data_fingerprint(args.data_dir),
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "fit_protocol": "train_only; validation_early_stopping; test_after_selection"})
    write_json(output_dir / "config.json", config)
    frame[["block_number", "original_block_number", "z_index", "y_index", "x_index", "split"]].to_csv(
        output_dir / "split.csv", index=False)
    print(f"训练输出：{output_dir}", flush=True)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, amp, args.accumulation)
        truth, estimate, _ = predict(model, val_loader, device, amp)
        val_loss = float(np.square(estimate.astype(float) - truth).mean())
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": config,
                        "best_epoch": epoch, "validation_mse_standardized_log10": val_loss},
                       output_dir / "best.pt")
        else:
            stale += 1
        row = {"epoch": epoch, "train_mse_standardized_log10": train_loss,
               "validation_mse_standardized_log10": val_loss,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | "
              f"best={best_epoch} | {row['seconds']:.1f}s", flush=True)
        if stale >= args.patience:
            print(f"早停：连续 {args.patience} 个 epoch 验证损失未改善。")
            break
    # 全程不看测试表现；选好最佳验证权重之后，才进行最终测试。
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    export_evaluation(model, frame, args, mean, std, device, amp, output_dir)
    print(f"最佳 epoch={best_epoch}；结果已保存：{output_dir}")


# ==================== 四、命令行入口 ====================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("check", "train", "test"), default="check")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--shape", nargs=3, type=int, default=(100, 100, 100), metavar=("Z", "Y", "X"))
    parser.add_argument("--target", choices=("mean", "xyz"), default="mean")
    parser.add_argument("--channels", nargs=3, type=int, default=(8, 16, 32))
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8, help="梯度累计步数；不等于扩大 BatchNorm 的批次")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Windows 默认 0，避免多进程启动问题")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for key in ("batch_size", "accumulation", "epochs", "patience", "cpu_threads"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} 必须为正整数。")
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("workers/weight_decay 不能为负，lr 必须大于零。")
    if min(args.shape) < 8 or any(c <= 0 for c in args.channels) or not 0 <= args.dropout < 1:
        parser.error("shape 各维必须 >= 8，channels 必须为正，dropout 必须在 [0,1) 内。")
    if args.mode == "test" and args.checkpoint is None:
        parser.error("test 模式必须提供 --checkpoint。")
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无可用 CUDA；请安装 GPU 版或选择 --device cpu。")
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = None
    if args.mode == "test":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        saved = checkpoint["config"]
        # 推理必须复用训练时的模型结构、标签定义及训练集统计量。
        for key in ("shape", "target", "channels", "dropout"):
            setattr(args, key, saved[key])
        if data_fingerprint(args.data_dir) != saved["data_fingerprint"]:
            raise ValueError("CSV 与训练时不同；本 test 模式仅复测原测试集，不静默混用外部数据。")
    frame = load_records(args.data_dir, args.shape)
    print(f"空间划分：{frame.split.value_counts().reindex(SPLITS).to_dict()}")
    model = CNN3D(args.channels, 1 if args.target == "mean" else 3, args.dropout).to(device)
    print(model)
    print(f"设备：{device}；参数量：{sum(p.numel() for p in model.parameters()):,}")
    print("局部卷积和池化均为 3×3×3、stride=1；输入保持原始分辨率。")
    if args.mode == "check":
        # 遍历 RAW 仅核对 ID/孔隙率，不训练，不保存文件，也不计算任何测试性能指标。
        dataset = RockDataset(frame, args.shape, args.target, np.zeros(model.regressor[-1].out_features),
                              np.ones(model.regressor[-1].out_features))
        for index in range(len(dataset)):
            dataset[index]
        image, label, block_id = dataset[0]
        started = time.perf_counter()
        prediction = model(image.unsqueeze(0).to(device))
        loss = nn.functional.mse_loss(prediction, label.unsqueeze(0).to(device))
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                         for p in model.parameters()):
            raise FloatingPointError("检查失败：输出或梯度包含非有限数值。")
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"全部 {len(dataset)} 个 RAW 与标签孔隙率一致。")
        print(f"样本 {block_id}：输入 {tuple(image.unsqueeze(0).shape)}，输出 {tuple(prediction.shape)}；"
              f"前向/反向通过，用时 {time.perf_counter() - started:.2f}s。")
        print("这是功能检查，不是训练结果；未启动完整训练，未写入模型文件。")
    elif args.mode == "train":
        train(model, frame, args, device)
    else:
        model.load_state_dict(checkpoint["model_state"])
        mean, std = np.array(saved["target_mean"]), np.array(saved["target_std"])
        output_dir = new_run_dir(args.output_dir, "evaluation")
        export_evaluation(model, frame, args, mean, std, device,
                          args.amp and device.type == "cuda", output_dir)
        print(f"评估结果：{output_dir}")


if __name__ == "__main__":
    main()
