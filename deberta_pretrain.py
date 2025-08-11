from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamATan2

from ai2arc_dataset import AI2ArcDatasetConfig, create_ai2arc_dataloader
from utils.functions import load_model_class, get_model_source_path


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class DebertaPretrainConfig(pydantic.BaseModel):
    # Model and dataset
    arch: ArchConfig
    dataset_name: str = "allenai/ai2_arc"
    dataset_config: str = "ARC-Easy"
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 512

    # Training hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class DebertaTrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    optimizer_lr: float


def cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, step: int, warmup_steps: int,
                               total_steps: int, lr_min_ratio: float, base_lr: float):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        # Linear warmup
        lr = base_lr * step / warmup_steps
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = base_lr * lr_min_ratio + 0.5 * base_lr * (1 - lr_min_ratio) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


@hydra.main(version_base=None, config_path="config", config_name="cfg_deberta_pretrain")
def main(hydra_cfg: DictConfig) -> None:
    # Convert to pydantic config
    config = DebertaPretrainConfig(**hydra_cfg)

    # Set random seed
    torch.manual_seed(config.seed)

    # Setup wandb
    if config.run_name is None:
        config.run_name = coolname.generate_slug(2)

    if config.project_name is None:
        config.project_name = "deberta-hrm-experiments"

    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.model_dump()
    )

    print(f"🚀 Starting training: {config.run_name}")
    print(f"📊 Config: {config}")

    # Create dataset
    dataset_config = AI2ArcDatasetConfig(
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        max_length=config.max_length,
        batch_size=config.global_batch_size,
    )

    train_loader = create_ai2arc_dataloader(dataset_config, "train")
    eval_loader = create_ai2arc_dataloader(dataset_config, "validation")

    print(f"📚 Dataset loaded: {len(train_loader)} train batches, {len(eval_loader)} eval batches")

    # Load model
    model_class = load_model_class(config.arch.name)
    model_config_dict = {k: v for k, v in config.arch.model_dump().items() if k not in ['name', 'loss']}
    model = model_class(model_class.Config(**model_config_dict))

    # Load loss
    loss_class = load_model_class(config.arch.loss.name)
    loss_config_dict = {k: v for k, v in config.arch.loss.model_dump().items() if k != 'name'}
    loss_head = loss_class(model, **loss_config_dict)

    print(f"🤖 Model loaded: {config.arch.name}")
    print(f"📉 Loss loaded: {config.arch.loss.name}")

    # Count parameters
    total_params = sum(p.numel() for p in loss_head.parameters())
    trainable_params = sum(p.numel() for p in loss_head.parameters() if p.requires_grad)
    print(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Setup optimizer
    optimizer = AdamATan2(
        loss_head.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

    # Training state
    train_state = DebertaTrainState(
        model=loss_head,
        optimizer=optimizer,
        optimizer_lr=config.lr
    )

    # Training loop
    step = 0
    total_steps = config.epochs * len(train_loader)

    print(f"🏁 Training for {config.epochs} epochs ({total_steps} steps)")

    for epoch in range(config.epochs):
        # Training
        train_state.model.train()
        train_metrics = {}

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            # Move to device
            batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in batch.items()}

            # Forward pass
            carry, loss, metrics, outputs, halted = train_state.model(batch)

            # Backward pass
            train_state.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), 1.0)
            train_state.optimizer.step()

            # Update learning rate
            current_lr = cosine_schedule_with_warmup(
                train_state.optimizer, step, config.lr_warmup_steps,
                total_steps, config.lr_min_ratio, config.lr
            )
            train_state.optimizer_lr = current_lr

            # Track metrics
            for k, v in metrics.items():
                if k not in train_metrics:
                    train_metrics[k] = []
                train_metrics[k].append(v.item() if torch.is_tensor(v) else v)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}",
                'acc': f"{metrics.get('accuracy', 0):.3f}" if 'accuracy' in metrics else "N/A"
            })

            step += 1

            # Evaluation
            if config.eval_interval and step % config.eval_interval == 0:
                eval_metrics = evaluate(train_state.model, eval_loader)

                # Log to wandb
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "lr": current_lr,
                    **{f"train/{k}": sum(v)/len(v) for k, v in train_metrics.items()},
                    **{f"eval/{k}": v for k, v in eval_metrics.items()}
                })

                print(f"📊 Step {step}: Train Loss={sum(train_metrics['loss'])/len(train_metrics['loss']):.4f}, "
                      f"Eval Acc={eval_metrics.get('accuracy', 0):.3f}")

                train_metrics = {}  # Reset
                train_state.model.train()

        # End of epoch evaluation
        eval_metrics = evaluate(train_state.model, eval_loader)

        # Log to wandb
        wandb.log({
            "step": step,
            "epoch": epoch + 1,
            "lr": train_state.optimizer_lr,
            **{f"train/{k}": sum(v)/len(v) for k, v in train_metrics.items()},
            **{f"eval/{k}": v for k, v in eval_metrics.items()}
        })

        print(f"🎯 Epoch {epoch+1} Complete - Eval Accuracy: {eval_metrics.get('accuracy', 0):.3f}")

    print(f"✅ Training complete!")
    wandb.finish()


def evaluate(model: nn.Module, eval_loader: DataLoader) -> Dict[str, float]:
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, desc="Evaluating", leave=False):
            # Move to device
            batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in batch.items()}

            # Forward pass
            carry, loss, metrics, outputs, halted = model(batch)

            total_loss += loss.item()
            if 'accuracy' in metrics:
                total_accuracy += metrics['accuracy'].item()
            total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "accuracy": total_accuracy / total_batches
    }


if __name__ == "__main__":
    main()
