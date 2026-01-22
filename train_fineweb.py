"""
Training script for Deep Delta Learning on FineWeb-Edu dataset.
Compares original DDL vs corrected DDL.
"""

import os
import sys
import math
import time
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))


class FineWebEduDataset(IterableDataset):
    """Streaming dataset for FineWeb-Edu."""

    def __init__(self, split: str, block_size: int, tokenizer, buffer_size: int = 10000):
        self.split = split
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size

        # Load dataset in streaming mode
        # Use sample-10BT subset for faster experiments
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

    def __iter__(self):
        buffer = []

        for example in self.dataset:
            # Tokenize text
            tokens = self.tokenizer.encode(example["text"], allowed_special=set())
            buffer.extend(tokens)

            # Yield chunks when buffer is large enough
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                buffer = buffer[self.block_size:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


class FineWebEduMapDataset(Dataset):
    """Pre-tokenized dataset for FineWeb-Edu (faster iteration)."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y


def prepare_fineweb_data(num_tokens: int = 10_000_000, cache_dir: str = "data"):
    """Download and prepare FineWeb-Edu tokens."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"fineweb_edu_{num_tokens // 1_000_000}M.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}")
        return torch.load(cache_path)

    print(f"Preparing FineWeb-Edu dataset ({num_tokens:,} tokens)...")
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    all_tokens = []
    total = 0

    for example in dataset:
        tokens = tokenizer.encode(example["text"], allowed_special=set())
        all_tokens.extend(tokens)
        total += len(tokens)

        if total % 1_000_000 == 0:
            print(f"  Tokenized {total:,} tokens...")

        if total >= num_tokens:
            break

    tokens = torch.tensor(all_tokens[:num_tokens], dtype=torch.long)
    torch.save(tokens, cache_path)
    print(f"Saved {len(tokens):,} tokens to {cache_path}")

    return tokens


@dataclass
class TrainConfig:
    """Training configuration."""
    model_type: str = "corrected"
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    head_dim: int = 64
    block_size: int = 256
    ddl_value_channels: int = 4

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 5000
    warmup_iters: int = 200
    grad_clip: float = 1.0

    eval_interval: int = 200
    log_interval: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False

    # FineWeb specific
    num_tokens: int = 10_000_000  # 10M tokens
    val_ratio: float = 0.1

    output_dir: str = "outputs"


def create_model(config: TrainConfig, vocab_size: int):
    """Create DDL model."""
    if config.model_type == "original":
        from model.DDL import GPT, GPTConfig
    else:
        from model.DDL_corrected import GPT, GPTConfig

    model_config = GPTConfig(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        block_size=config.block_size,
        ddl_value_channels=config.ddl_value_channels,
        ddl_beta_init=1.0,
        using_groupnorm=False,
        use_qk_rmsnorm=True,
    )

    return GPT(model_config)


def get_lr(it: int, config: TrainConfig) -> float:
    """Learning rate with warmup and cosine decay."""
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * coeff * 0.1 + config.learning_rate * 0.1


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device, eval_iters=50):
    """Estimate loss on train and val."""
    model.eval()
    losses = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        count = 0
        loader_iter = iter(loader)

        for _ in range(eval_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item()
            count += 1

        losses[split] = total_loss / count

    model.train()
    return losses


def train(config: TrainConfig) -> dict:
    """Train DDL model on FineWeb-Edu."""
    print(f"\n{'='*60}")
    print(f"Training DDL Model: {config.model_type.upper()} on FineWeb-Edu")
    print(f"{'='*60}")

    device = config.device
    os.makedirs(config.output_dir, exist_ok=True)

    # Prepare data
    tokens = prepare_fineweb_data(config.num_tokens)
    n = len(tokens)
    n_val = int(n * config.val_ratio)
    n_train = n - n_val

    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:]

    train_dataset = FineWebEduMapDataset(train_tokens, config.block_size)
    val_dataset = FineWebEduMapDataset(val_tokens, config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"Train tokens: {n_train:,}, Val tokens: {n_val:,}")
    print(f"Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}")

    # GPT-2 vocab size
    vocab_size = 50304  # Padded for efficiency

    # Create model
    model = create_model(config, vocab_size)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if config.compile_model and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Training
    history = {"train_loss": [], "val_loss": [], "iter": [], "lr": []}
    train_iter = iter(train_loader)
    model.train()

    start_time = time.time()

    for it in range(config.max_iters):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(it, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        if it % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | time {elapsed:.1f}s")

        if it % config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, device)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["iter"].append(it)
            history["lr"].append(lr)
            print(f"  => eval: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")

    final_losses = estimate_loss(model, train_loader, val_loader, device)
    print(f"\nFinal: train_loss={final_losses['train']:.4f}, val_loss={final_losses['val']:.4f}")

    # Save history
    history_path = os.path.join(config.output_dir, f"fineweb_history_{config.model_type}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_comparison(orig_history, corr_history, output_dir):
    """Plot comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(orig_history["iter"], orig_history["train_loss"], "b-", label="Original DDL", linewidth=2)
    axes[0].plot(corr_history["iter"], corr_history["train_loss"], "r-", label="Corrected DDL", linewidth=2)
    axes[0].set_xlabel("Iteration", fontsize=12)
    axes[0].set_ylabel("Training Loss", fontsize=12)
    axes[0].set_title("FineWeb-Edu: Training Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(orig_history["iter"], orig_history["val_loss"], "b-", label="Original DDL", linewidth=2)
    axes[1].plot(corr_history["iter"], corr_history["val_loss"], "r-", label="Corrected DDL", linewidth=2)
    axes[1].set_xlabel("Iteration", fontsize=12)
    axes[1].set_ylabel("Validation Loss", fontsize=12)
    axes[1].set_title("FineWeb-Edu: Validation Loss", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fineweb_loss_comparison.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "fineweb_loss_comparison.pdf"), bbox_inches="tight")
    print(f"\nSaved: {output_dir}/fineweb_loss_comparison.png")

    # Summary
    print(f"\n{'='*50}")
    print("FINEWEB-EDU RESULTS")
    print(f"{'='*50}")
    print(f"Original:  Train={orig_history['train_loss'][-1]:.4f}, Val={orig_history['val_loss'][-1]:.4f}")
    print(f"Corrected: Train={corr_history['train_loss'][-1]:.4f}, Val={corr_history['val_loss'][-1]:.4f}")
    train_imp = (orig_history['train_loss'][-1] - corr_history['train_loss'][-1]) / orig_history['train_loss'][-1] * 100
    val_imp = (orig_history['val_loss'][-1] - corr_history['val_loss'][-1]) / orig_history['val_loss'][-1] * 100
    print(f"Improvement: Train={train_imp:+.1f}%, Val={val_imp:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Train DDL on FineWeb-Edu")
    parser.add_argument("--model", choices=["original", "corrected", "both"], default="both")
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--num_tokens", type=int, default=10_000_000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=20)

    args = parser.parse_args()

    base_config = TrainConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        head_dim=args.hidden_size // args.num_heads,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        num_tokens=args.num_tokens,
        device=args.device,
        output_dir=args.output_dir,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    histories = {}

    if args.model in ["original", "both"]:
        config = TrainConfig(**{**asdict(base_config), "model_type": "original"})
        histories["original"] = train(config)

    if args.model in ["corrected", "both"]:
        config = TrainConfig(**{**asdict(base_config), "model_type": "corrected"})
        histories["corrected"] = train(config)

    if "original" in histories and "corrected" in histories:
        plot_comparison(histories["original"], histories["corrected"], args.output_dir)


if __name__ == "__main__":
    main()
