import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, List
import logging


logger = logging.getLogger(__name__)


class MultiCropAugmentation:
    """
    Multi-crop augmentation strategy from DINO paper.
    # NOTE: Creates multiple crops of the same image for student and teacher networks
    """
    
    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
        size: int = 224,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.size = size
        
        # Base augmentation pipeline
        self.base_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur((3, 3), sigma=(0.1, 2.0)),
        ])
        
        # Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def __call__(self, img):
        """
        Apply multi-crop augmentation.
        
        Returns:
            Tuple of (student_crop, teacher_crop) for DINO training
            # NOTE: Student network sees multiple crops, teacher sees global crops
        """
        # Two global crops for teacher and student
        crop1 = transforms.RandomResizedCrop(
            self.size,
            scale=self.global_crops_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )(img)
        
        crop2 = transforms.RandomResizedCrop(
            self.size,
            scale=self.global_crops_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )(img)
        
        # Apply base augmentations
        crop1 = self.base_transform(crop1)
        crop2 = self.base_transform(crop2)
        
        # Normalize
        crop1 = self.normalize(crop1)
        crop2 = self.normalize(crop2)
        
        return crop1, crop2


class DINO_ImageNet(Dataset):
    """
    ImageNet dataset wrapper for DINO training with multi-crop augmentation.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
    ):
        self.split = split
        self.transform = transform
        
        # Use torchvision ImageNet
        # NOTE: Requires manual download from https://www.image-net.org/
        self.dataset = datasets.ImageNet(
            root=root_dir,
            split=split,
            transform=None,  # We'll apply transform manually
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.dataset[idx]
        
        if self.transform:
            crop1, crop2 = self.transform(img)
        else:
            # Default: no augmentation, just return same image twice
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            crop1 = transform(img)
            crop2 = transform(img)
        
        return crop1, crop2


class SimpleImageNetDataset(Dataset):
    """
    Simple ImageNet dataset for testing/debugging.
    # TODO: Replace with actual ImageNet loading when dataset is available
    """
    
    def __init__(
        self,
        root_dir: str = './data',
        split: str = 'train',
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Fallback: use CIFAR-10 for testing
        try:
            self.dataset = DINO_ImageNet(
                root_dir=root_dir,
                split=split,
                transform=transform,
            )
        except Exception as e:
            logger.warning(f"Could not load ImageNet: {e}")
            logger.warning("Falling back to CIFAR-10 for debugging")
            
            cifar_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            
            self.dataset = datasets.CIFAR10(
                root=root_dir,
                train=(split == 'train'),
                download=True,
                transform=cifar_transform,
            )
            self.use_cifar = True
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        if hasattr(self, 'use_cifar') and self.use_cifar:
            img, _ = self.dataset[idx]
            # Return same image twice (no multi-crop for debugging)
            return img, img
        else:
            return self.dataset[idx]


def build_loader(
    data_path: str = './data',
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build data loaders for DINO training.
    
    Args:
        data_path: Path to dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Multi-crop augmentation
    train_augmentation = MultiCropAugmentation()
    
    # Train dataset
    train_dataset = SimpleImageNetDataset(
        root_dir=data_path,
        split='train',
        transform=train_augmentation,
    )
    
    # Val dataset (minimal augmentation)
    val_augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    val_dataset = SimpleImageNetDataset(
        root_dir=data_path,
        split='val',
        transform=val_augmentation,
    )
    
    # Data loaders
    # NOTE: Paper uses batch size 1024; adjust num_workers based on hardware
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test data loading
    logger.info("Testing data loaders...")
    
    train_loader, val_loader = build_loader(
        batch_size=32,
        num_workers=0,
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Inspect first batch
    for batch_idx, (images1, images2) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Student crops shape: {images1.shape}")
        print(f"  Teacher crops shape: {images2.shape}")
        break