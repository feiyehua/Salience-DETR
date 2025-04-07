import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple, List


class StarReLU(nn.Module):
    """StarReLU激活函数 - 计算效率高且减轻分布偏差
    公式: max(s*x, 0) + max(s'*x, 0)
    其中s和s'是可学习参数
    """
    def __init__(self, scale_value=1.0, bias_value=0.0, channel_dim=None):
        super().__init__()
        self.scale_value = scale_value
        self.bias_value = bias_value
        self.channel_dim = channel_dim
        
        if channel_dim is not None:
            self.scale = nn.Parameter(torch.ones(channel_dim) * scale_value)
            self.bias = nn.Parameter(torch.ones(channel_dim) * bias_value)
        else:
            self.scale = nn.Parameter(torch.tensor(scale_value))
            self.bias = nn.Parameter(torch.tensor(bias_value))
    
    def forward(self, x):
        if self.channel_dim is not None:
            # 适用于多通道数据
            scale = self.scale.view(1, -1, 1, 1) if x.dim() == 4 else self.scale.view(1, -1)
            bias = self.bias.view(1, -1, 1, 1) if x.dim() == 4 else self.bias.view(1, -1)
        else:
            scale = self.scale
            bias = self.bias
            
        return torch.maximum(x * scale, torch.tensor(0.0, device=x.device)) + torch.maximum(x * bias, torch.tensor(0.0, device=x.device))


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.se_module = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        nn.init.kaiming_normal_(self.conv_mask.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        batch, channel, height, width = x.shape
        # spatial pool
        # b, 1, c, h * w
        input_x = x.view(batch, channel, height * width).unsqueeze(1)
        # b, 1, h * w, 1
        context_mask = self.conv_mask(x).view(batch, 1, height * width)
        context_mask = self.softmax(context_mask).unsqueeze(-1)
        # b, 1, c, 1
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)
        return self.se_module(context) * x


class ContextBlock(nn.Module):
    """ContextBlock module in GCNet.

    See 'GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond'
    (https://arxiv.org/abs/1904.11492) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        ratio (float): Ratio of channels of transform bottleneck
        pooling_type (str): Pooling method for context modeling.
            Options are 'att' and 'avg', stand for attention pooling and
            average pooling respectively. Default: 'att'.
        fusion_types (Sequence[str]): Fusion method for feature fusion,
            Options are 'channels_add', 'channel_mul', stand for channelwise
            addition and multiplication respectively. Default: ('channel_add',)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float,
        pooling_type: str = "att",
        fusion_types: tuple = ("channel_add",),
    ):
        super().__init__()
        assert pooling_type in ["avg", "att"]
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ["channel_add", "channel_mul"]
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, "at least one fusion should be used"
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == "att":
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if "channel_add" in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1),
            )
        else:
            self.channel_add_conv = None
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1),
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == "att":
            nn.init.kaiming_normal_(self.conv_mask.weight, mode="fan_in")
            nn.init.constant_(self.conv_mask.bias, 0)

        if self.channel_add_conv is not None:
            nn.init.constant_(self.channel_add_conv[-1].weight, 0)
            nn.init.constant_(self.channel_add_conv[-1].bias, 0)
        if self.channel_mul_conv is not None:
            nn.init.constant_(self.channel_mul_conv[-1].weight, 0)
            nn.init.constant_(self.channel_mul_conv[-1].bias, 0)

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        if self.pooling_type == "att":
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class AdaptiveFeatureCalibration(nn.Module):
    """Adaptive Feature Calibration module to dynamically adjust feature weights
    
    This module enhances the visibility of small targets by predicting channel-wise
    or spatial weights that emphasize features in regions likely to contain small targets.
    It is designed to be lightweight and can be placed after the backbone or within 
    the multiscale token mixer.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_levels: int = 4,
        hidden_dim: int = 128,
        use_spatial_weights: bool = True,
        use_channel_weights: bool = True,
        small_target_threshold: float = 64,  # Pixel area threshold for small targets
    ):
        """
        Args:
            embed_dim: The embedding dimension
            num_levels: Number of feature levels to process
            hidden_dim: Hidden dimension for the MLP
            use_spatial_weights: Whether to predict spatial weights
            use_channel_weights: Whether to predict channel weights
            small_target_threshold: Threshold to define small targets by area
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.use_spatial_weights = use_spatial_weights
        self.use_channel_weights = use_channel_weights
        self.small_target_threshold = small_target_threshold
        
        assert use_spatial_weights or use_channel_weights, "At least one of spatial or channel weights must be used"
        
        # Global context aggregation
        self.context_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels)
        ])
        
        # Channel attention
        if use_channel_weights:
            self.channel_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim // 2, embed_dim),
                    nn.Sigmoid()
                ) for _ in range(num_levels)
            ])
        
        # Spatial attention
        if use_spatial_weights:
            self.spatial_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
                    nn.Sigmoid()
                ) for _ in range(num_levels)
            ])
        
        # Scale prediction - auxiliary task to help guide the feature calibration
        # This subnet predicts whether a region contains small targets
        self.scale_pred_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.GroupNorm(4, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_levels)
        ])
        
        self.init_weights()
    
    def init_weights(self):
        # Initialize the last layer of spatial convs with small weights
        if self.use_spatial_weights:
            for m in self.spatial_convs:
                nn.init.constant_(m[-2].weight, 0.)
                nn.init.constant_(m[-2].bias, 0.)
        
        # Initialize the scale prediction with small weights
        for m in self.scale_pred_convs:
            nn.init.constant_(m[-2].weight, 0.)
            nn.init.constant_(m[-2].bias, 0.)
    
    def forward(
        self, 
        multi_level_features: List[torch.Tensor],
        targets: dict = None
    ):
        """
        Args:
            multi_level_features: List of feature maps at different scales [P2, P3, P4, P5]
            targets: Optional dictionary with ground truth boxes for training
            
        Returns:
            calibrated_features: List of calibrated feature maps
            scale_predictions: List of scale prediction maps (1 for small targets, 0 for large)
        """
        calibrated_features = []
        scale_predictions = []
        
        # Generate scale targets for training if targets are provided
        scale_targets = None
        if targets is not None and self.training:
            # Create scale targets based on box areas
            scale_targets = [torch.zeros_like(feat[:, 0:1]) for feat in multi_level_features]
            for batch_idx, target in enumerate(targets):
                boxes = target["boxes"]  # [N, 4] in cxcywh format
                areas = boxes[:, 2] * boxes[:, 3]  # Area in relative coordinates
                
                # Convert to absolute pixel areas based on feature map size
                for level_idx, feat in enumerate(multi_level_features):
                    h, w = feat.shape[-2:]
                    abs_areas = areas * h * w
                    
                    # Get small target masks (1 for small targets, 0 for large)
                    small_mask = (abs_areas < self.small_target_threshold).float()
                    
                    # TODO: Generate spatial scale targets from boxes
                    # This would require projecting boxes to feature maps at each level
                    # For simplicity, we'll skip this in this implementation
            
        # Process each level
        for level_idx, features in enumerate(multi_level_features):
            # Extract context features
            context = self.context_conv[level_idx](features)
            
            # Predict channel weights
            if self.use_channel_weights:
                channel_weights = self.channel_mlps[level_idx](context)
                channel_weights = channel_weights.view(features.shape[0], -1, 1, 1)
                features = features * channel_weights
            
            # Predict spatial weights
            if self.use_spatial_weights:
                spatial_weights = self.spatial_convs[level_idx](context)
                features = features * spatial_weights
            
            # Predict scale for auxiliary task
            scale_pred = self.scale_pred_convs[level_idx](context)
            scale_predictions.append(scale_pred)
            
            # Add to output
            calibrated_features.append(features)
        
        return calibrated_features, scale_predictions
