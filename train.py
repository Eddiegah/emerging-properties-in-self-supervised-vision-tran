import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import logging
from typing import Tuple
import json
import time

from model import DINOModel, DINOLoss, SimpleViT
from data_loader import build_loader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DINOTrainer:
    """Trainer class for DINO self-supervised learning."""
    
    def __init__(
        self,
        model: DINOModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        args: argparse.Namespace,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        
        # Optimizer
        # NOTE: Paper uses AdamW with learning rate scaling
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        # Loss function
        self.criterion = DINOLoss(
            student_temp=args.student_temp,
            teacher_temp=args.teacher_temp,
        )
        
        # Learning rate scheduler
        # NOTE: Paper uses cosine annealing schedule
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images_student, images_teacher) in enumerate(self.train_loader):
            images_student = images_student.to(self.device)
            images_teacher = images_teacher.to(self.device)
            
            # Forward pass
            student_output, teacher_output = self.model(images_student, images_teacher)
            
            # Compute loss
            loss = self.criterion(student_output, teacher_output)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update teacher network with momentum
            self.model.update_teacher(momentum=self.args.momentum_teacher)
            
            # Update centering buffer
            self.model.update_center(teacher_output, momentum=self.args.momentum_center)
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % self.args.log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"Epoch [{self.current_epoch + 1}/{self.args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for images_student, images_teacher in self.val_loader:
            images_student = images_student.to(self.device)
            images_teacher = images_teacher.to(self.device)
            
            student_output, teacher_output = self.model(images_student, images_teacher)
            loss = self.criterion(student_output, teacher_output)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop."""
        logger.info("Starting DINO training...")
        logger.info(f"Total epochs: {self.args.epochs}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            logger.info(
                f"Epoch {epoch + 1}/{self.args.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint()
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        
        save_path = Path(self.args.checkpoint_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = save_path / 'checkpoint_best.pt'
        else:
            path = save_path / f'checkpoint_epoch_{self.current_epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        logger.info(f"Checkpoint loaded from {path}")


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DINO training')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='vit_small',
                        help='Model architecture')
    parser.add_argument('--head-dim', type=int, default=65536,
                        help='Output dimension of projection head')
    
    # Training arguments
    # NOTE: Batch size 1024 from paper
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (paper uses 1024)')
    # NOTE: Learning rate 0.0005 * batchsize/256 from paper
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='Learning rate (scales with batch size)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate in cosine schedule')
    parser.add_argument('--epochs', type=int, default=100,  # NOTE: Paper trains longer
                        help='Number of training epochs')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs')
    
    # DINO specific arguments
    parser.add_argument('--student-temp', type=float, default=0.1,
                        help='Temperature for student softmax')
    parser.add_argument('--teacher-temp', type=float, default=0.04,
                        help='Temperature for teacher softmax')
    parser.add_argument('--momentum-teacher', type=float, default=0.99,
                        help='Momentum coefficient for teacher update')
    parser.add_argument('--momentum-center', type=float, default=0.9,
                        help='Momentum coefficient for center update')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin memory for data loader')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Checkpoint save interval')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from checkpoint')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build data loaders
    logger.info("Building data loaders...")
    train_loader, val_loader = build_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    
    # Build model
    # TODO: Replace SimpleViT with actual ViT implementation
    logger.info("Building model...")
    backbone = SimpleViT(
        image_size=224,
        patch_size=16,
        dim=384,
        depth=12,
        heads=6,
    )
    
    model = DINOModel(
        backbone=backbone,
        head_dim=args.head_dim,
    )
    
    # Create trainer
    trainer = DINOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
    )
    
    # Load checkpoint if resuming
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()