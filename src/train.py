import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root directory to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import logger, set_seed
from src.data_loader import get_data_loaders

# Import models
from models.efficientnet_b0 import EfficientNetB0
from models.efficientnet_b3 import EfficientNetB3
from models.xception import XceptionNet
from models.vit import VisionTransformer
from models.hybrid_deepshield import HybridDeepShield

def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """Instantiates and returns the selected model by name."""
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        return EfficientNetB0(pretrained=pretrained)
    elif model_name == "efficientnet_b3":
        return EfficientNetB3(pretrained=pretrained)
    elif model_name == "xception":
        return XceptionNet(pretrained=pretrained)
    elif model_name == "vit":
        return VisionTransformer(pretrained=pretrained)
    elif model_name == "hybrid":
        return HybridDeepShield(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

class Trainer:
    """Manages the model training, validation, checkpointing, and logging."""
    
    def __init__(self, model_name: str, resume: bool = True):
        self.model_name = model_name
        self.resume = resume
        set_seed(Config.SEED)
        
        # Load directories
        Config.create_dirs()

        # Instantiate Model and send to device
        self.model = get_model(self.model_name, pretrained=True)

        # On CPU: freeze the heavy EfficientNet backbone to speed up training.
        # Only train: frequency branch + CBAM attention + classifier head.
        # On GPU: train everything end-to-end for full accuracy.
        if Config.DEVICE == "cpu" and hasattr(self.model, "spatial_backbone"):
            for param in self.model.spatial_backbone.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"CPU mode: EfficientNet backbone FROZEN. "
                f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)"
            )

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training via DataParallel.")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(Config.DEVICE)
        
        # Optimizer, Loss and Scheduler — only trainable parameters
        # BCEWithLogitsLoss + label smoothing prevents overconfident predictions
        label_smooth = getattr(Config, 'LABEL_SMOOTHING', 0.0)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=None
        )
        self._label_smooth = label_smooth  # applied manually in train_epoch

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        # Cosine annealing with linear warmup via scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=Config.EPOCHS - getattr(Config, 'WARMUP_EPOCHS', 2),
            eta_min=1e-6
        )
        
        # Tensorboard Writer
        self.writer = SummaryWriter(log_dir=os.path.join(Config.OUTPUT_DIR, "logs", self.model_name))
        
        # Mixed Precision Scaler — use new API
        use_amp = (Config.DEVICE == "cuda" and Config.MIXED_PRECISION)
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Training state variables
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Checkpoint paths
        self.best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{self.model_name}_best.pth")
        self.latest_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{self.model_name}_latest.pth")
        
        # In-memory history tracking
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        if self.resume:
            self.load_checkpoint()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves a training checkpoint including states of model, optimizer, and scheduler."""
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None
        }
        
        # Save latest
        torch.save(checkpoint, self.latest_ckpt_path)
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.best_ckpt_path)
            logger.info(f"Saved new best model checkpoint to {self.best_ckpt_path}")

    def load_checkpoint(self):
        """Resumes training from the latest checkpoint if it exists."""
        if os.path.exists(self.latest_ckpt_path):
            try:
                logger.info(f"Loading checkpoint from {self.latest_ckpt_path} to resume training...")
                checkpoint = torch.load(self.latest_ckpt_path, map_location=Config.DEVICE)
                
                # Load state dicts
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.best_val_loss = checkpoint["best_val_loss"]
                self.start_epoch = checkpoint["epoch"] + 1
                
                if checkpoint.get("scaler_state_dict") and self.scaler.is_enabled():
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    
                # Load training history if it exists
                history_path = os.path.join(Config.REPORTS_DIR, f"{self.model_name}_history.json")
                if os.path.exists(history_path):
                    with open(history_path, "r") as f:
                        self.history = json.load(f)
                    
                logger.info(f"Resumed successfully from epoch {self.start_epoch}.")
            except Exception as e:
                logger.warning(f"Failed to resume checkpoint: {e}. Starting training from scratch.")
        else:
            logger.info("No checkpoint found. Starting training from scratch.")

    def train_epoch(self, train_loader, epoch: int):
        """Runs a single epoch of training."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).float().unsqueeze(1)

            # Keep original hard labels (0/1) for accuracy — smoothing is only for loss
            hard_labels = labels.clone()

            # Apply label smoothing: soft_label = label*(1-ε) + 0.5*ε
            if self._label_smooth > 0:
                labels = labels * (1.0 - self._label_smooth) + 0.5 * self._label_smooth

            self.optimizer.zero_grad()
            
            # Forward pass with AMP autocast
            with torch.amp.autocast('cuda', enabled=(Config.DEVICE == "cuda" and Config.MIXED_PRECISION)):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)   # smoothed labels for loss
                
            # Backward pass with Scaler
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIPPING)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIPPING)
                self.optimizer.step()
                
            total_loss += loss.item() * images.size(0)
            
            # Calculate accuracy against hard (0/1) labels — NOT smoothed labels
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == hard_labels).sum().item()
            total += hard_labels.size(0)
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total*100:.2f}%"})

            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        self.writer.add_scalar("Loss/train", epoch_loss, epoch)
        self.writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        
        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch: int):
        """Evaluates the model on the validation dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE).float().unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        self.writer.add_scalar("Loss/val", epoch_loss, epoch)
        self.writer.add_scalar("Accuracy/val", epoch_acc, epoch)
        
        return epoch_loss, epoch_acc

    def fit(self):
        """Executes the full training loop with early stopping, scheduling, and validation."""
        # Load dataloaders
        train_loader, val_loader, _ = get_data_loaders()
        
        if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
            logger.error("Dataset folders are empty. Run preprocessing first.")
            return
            
        logger.info(f"Beginning training loop for {self.model_name}...")
        
        for epoch in range(self.start_epoch, Config.EPOCHS):
            # 1. Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            # 2. Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            # 3. Schedule LR
            self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Learning_Rate", lr, epoch)
            
            logger.info(
                f"Epoch {epoch}/{Config.EPOCHS - 1} Summary: "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {lr:.6f}"
            )
            
            # Record and save history to JSON file
            self.history["train_loss"].append(float(train_loss))
            self.history["train_acc"].append(float(train_acc))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_acc"].append(float(val_acc))
            
            history_path = os.path.join(Config.REPORTS_DIR, f"{self.model_name}_history.json")
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=4)
            
            # Checkpoint & Early Stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
            self.save_checkpoint(epoch, is_best=is_best)
            
            if self.early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch} epochs of no improvement.")
                break
                
        self.writer.close()
        logger.info(f"Training for {self.model_name} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deepfake detection models.")
    parser.add_argument("--model", type=str, default="hybrid", choices=["efficientnet_b0", "efficientnet_b3", "xception", "vit", "hybrid"], help="Model to train")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch instead of resuming")
    args = parser.parse_args()
    
    trainer = Trainer(model_name=args.model, resume=not args.no_resume)
    trainer.fit()
