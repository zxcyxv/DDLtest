"""
Training script for Deep Delta Learning experiments.
Compares original DDL vs corrected DDL on TinyShakespeare dataset.
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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent))


def download_tiny_shakespeare(data_dir: str = "data") -> str:
    """Download TinyShakespeare dataset."""
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "tiny_shakespeare.txt")

    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading TinyShakespeare from {url}...")
        urllib.request.urlretrieve(url, data_path)
        print(f"Saved to {data_path}")

    return data_path


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices: list[int]) -> str:
        return "".join([self.idx_to_char[i] for i in indices])


class TextDataset(Dataset):
    """Dataset for character-level language modeling."""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_type: str = "corrected"  # "original" or "corrected"
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    head_dim: int = 64
    block_size: int = 256
    ddl_value_channels: int = 4

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 5000
    warmup_iters: int = 100
    grad_clip: float = 1.0

    # Logging
    eval_interval: int = 100
    log_interval: int = 10
    save_interval: int = 1000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"


def create_model(config: TrainConfig, vocab_size: int):
    """Create DDL model based on config."""
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

    model = GPT(model_config)
    return model


def get_lr(it: int, config: TrainConfig) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * coeff * 0.1 + config.learning_rate * 0.1


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    eval_iters: int = 50,
) -> dict[str, float]:
    """Estimate loss on train and val sets."""
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
    """Train a DDL model and return training history."""
    print(f"\n{'='*60}")
    print(f"Training DDL Model: {config.model_type.upper()}")
    print(f"{'='*60}")

    # Setup
    device = config.device
    os.makedirs(config.output_dir, exist_ok=True)

    # Load data
    data_path = download_tiny_shakespeare(config.data_dir)
    with open(data_path, "r") as f:
        text = f.read()

    # Tokenize
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Split data
    n = len(data)
    train_data = data[: int(0.9 * n)]
    val_data = data[int(0.9 * n) :]

    # Create datasets
    train_dataset = TextDataset(train_data, config.block_size)
    val_dataset = TextDataset(val_data, config.block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    model = create_model(config, tokenizer.vocab_size)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Compile if requested
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

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "iter": [],
        "lr": [],
    }

    train_iter = iter(train_loader)
    model.train()

    start_time = time.time()
    best_val_loss = float("inf")

    for it in range(config.max_iters):
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Update learning rate
        lr = get_lr(it, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        _, loss = model(x, targets=y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # Logging
        if it % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | time {elapsed:.1f}s"
            )

        # Evaluation
        if it % config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, device)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["iter"].append(it)
            history["lr"].append(lr)

            print(
                f"  => eval: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

    # Final evaluation
    final_losses = estimate_loss(model, train_loader, val_loader, device)
    print(f"\nFinal: train_loss={final_losses['train']:.4f}, val_loss={final_losses['val']:.4f}")

    # Save history
    history_path = os.path.join(
        config.output_dir, f"history_{config.model_type}.json"
    )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_comparison(
    original_history: dict,
    corrected_history: dict,
    output_dir: str,
):
    """Plot comparison of training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    ax.plot(
        original_history["iter"],
        original_history["train_loss"],
        label="Original DDL",
        color="blue",
        alpha=0.8,
    )
    ax.plot(
        corrected_history["iter"],
        corrected_history["train_loss"],
        label="Corrected DDL",
        color="red",
        alpha=0.8,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[1]
    ax.plot(
        original_history["iter"],
        original_history["val_loss"],
        label="Original DDL",
        color="blue",
        alpha=0.8,
    )
    ax.plot(
        corrected_history["iter"],
        corrected_history["val_loss"],
        label="Corrected DDL",
        color="red",
        alpha=0.8,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "loss_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {plot_path}")

    # Also save as PDF
    pdf_path = os.path.join(output_dir, "loss_comparison.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    orig_final_train = original_history["train_loss"][-1]
    orig_final_val = original_history["val_loss"][-1]
    corr_final_train = corrected_history["train_loss"][-1]
    corr_final_val = corrected_history["val_loss"][-1]

    print(f"\nOriginal DDL:")
    print(f"  Final Train Loss: {orig_final_train:.4f}")
    print(f"  Final Val Loss:   {orig_final_val:.4f}")

    print(f"\nCorrected DDL:")
    print(f"  Final Train Loss: {corr_final_train:.4f}")
    print(f"  Final Val Loss:   {corr_final_val:.4f}")

    print(f"\nImprovement:")
    print(f"  Train Loss: {(orig_final_train - corr_final_train) / orig_final_train * 100:+.2f}%")
    print(f"  Val Loss:   {(orig_final_val - corr_final_val) / orig_final_val * 100:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train DDL models")
    parser.add_argument(
        "--model", type=str, choices=["original", "corrected", "both"], default="both",
        help="Which model(s) to train"
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    # Base config
    base_config = TrainConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        head_dim=args.hidden_size // args.num_heads,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        device=args.device,
        output_dir=args.output_dir,
        compile_model=args.compile,
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

    # Plot comparison if both models were trained
    if "original" in histories and "corrected" in histories:
        plot_comparison(histories["original"], histories["corrected"], args.output_dir)


if __name__ == "__main__":
    main()
