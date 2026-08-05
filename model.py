import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Dict, List


class DINOHead(nn.Module):
    """
    Projection head for DINO self-supervised learning.
    Applies a series of linear layers followed by batch normalization.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = True,
        norm_last_layer: bool = True,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        
        num_layers = 3
        hidden_dims = [hidden_dim] * (num_layers - 2) + [bottleneck_dim]
        
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
            
            if i < len(hidden_dims) - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
        
        # Final projection layer
        layers.append(nn.Linear(in_dim, out_dim))
        
        if norm_last_layer:
            layers.append(nn.BatchNorm1d(out_dim, affine=False))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimpleViT(nn.Module):
    """
    Simplified Vision Transformer implementation for DINO.
    # NOTE: Full ViT implementation is complex; this is a starter version
    # For production, use timm.create_model('vit_small', ...) or similar
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_dim: int = 1536,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=depth,
        )
        
        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)  # (B, dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, dim)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        x = self.norm(x)
        
        return x[:, 0]  # Return cls token


class DINOModel(nn.Module):
    """
    DINO (Self-Supervised Learning with Vision Transformers) model.
    
    Implements a student-teacher architecture with:
    - Multi-crop augmentation strategy
    - Momentum encoder for teacher network
    - Centering and sharpening to avoid collapse
    - Cross-entropy loss between student and teacher predictions
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        head_dim: int = 65536,  # NOTE: output dimension of projection head
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn_in_head: bool = True,
    ):
        super().__init__()
        
        # Extract backbone output dimension
        # NOTE: Assumes backbone outputs features of this size
        backbone_dim = 384  # For ViT-Small, adjust for other architectures
        
        self.backbone = backbone
        self.student_head = DINOHead(
            in_dim=backbone_dim,
            out_dim=head_dim,
            use_bn=use_bn_in_head,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
        
        self.teacher_backbone = self._create_teacher_backbone(backbone)
        self.teacher_head = DINOHead(
            in_dim=backbone_dim,
            out_dim=head_dim,
            use_bn=use_bn_in_head,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
        
        # Teacher parameters are not updated by gradient descent
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        
        # Centering buffer for teacher
        self.register_buffer("center", torch.zeros(1, head_dim))
        
    @staticmethod
    def _create_teacher_backbone(backbone: nn.Module) -> nn.Module:
        """Creates a deep copy of the backbone for teacher network."""
        import copy
        return copy.deepcopy(backbone)
    
    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student network."""
        features = self.backbone(x)
        return self.student_head(features)
    
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher network (detached)."""
        with torch.no_grad():
            features = self.teacher_backbone(x)
            return self.teacher_head(features)
    
    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.99):
        """
        Update teacher network parameters using momentum.
        # NOTE: momentum typically 0.99 or 0.999
        """
        for param_s, param_t in zip(
            self.backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_t.data = param_t.data * momentum + param_s.data * (1 - momentum)
        
        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data = param_t.data * momentum + param_s.data * (1 - momentum)
    
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor, momentum: float = 0.9):
        """
        Update the centering buffer to prevent collapse.
        # NOTE: momentum for center update typically 0.9
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * momentum + batch_center * (1 - momentum)
    
    def forward(
        self,
        student_input: torch.Tensor,
        teacher_input: torch.Tensor,
    ) -> tuple:
        """
        Forward pass for DINO training.
        
        Args:
            student_input: Input for student network
            teacher_input: Input for teacher network
            
        Returns:
            Tuple of (student_output, teacher_output)
        """
        student_output = self.forward_student(student_input)
        teacher_output = self.forward_teacher(teacher_input)
        return student_output, teacher_output


class DINOLoss(nn.Module):
    """
    DINO loss function combining cross-entropy with temperature sharpening.
    """
    def __init__(
        self,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        # NOTE: teacher_temp is lower to create sharper targets
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DINO loss using cross-entropy with temperature scaling.
        
        Args:
            student_output: Student network output (B, head_dim)
            teacher_output: Teacher network output (B, head_dim)
            
        Returns:
            Loss scalar
        """
        # Temperature-scaled softmax
        student_prob = F.softmax(student_output / self.student_temp, dim=-1)
        teacher_prob = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        
        # Cross-entropy: H(teacher_prob || student_prob)
        loss = -(teacher_prob * torch.log(student_prob + 1e-8)).sum(dim=-1).mean()
        
        return loss