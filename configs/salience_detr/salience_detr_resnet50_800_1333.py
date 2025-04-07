from torch import nn
# from torchvision.ops import FrozenBatchNorm2d

from models.backbones.resnet import ResNetBackbone
from models.bricks.misc import FrozenBatchNorm2d
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.basic import StarReLU
from models.bricks.salience_transformer import (
    SalienceTransformer,
    SalienceTransformerDecoder,
    SalienceTransformerDecoderLayer,
    SalienceTransformerEncoder,
    SalienceTransformerEncoderLayer,
)
from models.bricks.set_criterion import HybridSetCriterion
from models.detectors.salience_detr import SalienceCriterion, SalienceDETR
from models.matcher.hungarian_matcher import HungarianMatcher
from models.necks.channel_mapper import ChannelMapper
from models.necks.repnet import RepVGGPluXNetwork

# mostly changed parameters
embed_dim = 256
num_classes = 91
num_queries = 900
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

# Small target enhancement parameters
small_target_threshold = 0.04  # 4% of image area
small_target_weight_factor = 2.0

# 高级损失函数参数
use_ciou_loss = True  # 使用CIoU损失代替GIoU损失
use_star_relu = True  # 使用StarReLU激活函数

# Uncertainty-based RoI screening parameters
use_uncertainty_roi_screening = True
uncertainty_weight = 0.5  # 不确定性在综合分数中的权重
top_k_per_level = [200, 150, 100, 50]  # 每个层级选择的RoI数量
adaptive_roi_layers = 2  # 自适应RoI处理的层数
use_nsa = True  # 是否使用NSA稀疏注意力机制

# instantiate model components
position_embedding = PositionEmbeddingSine(embed_dim // 2, temperature=10000, normalize=True, offset=-0.5)

backbone = ResNetBackbone(
    "resnet50", norm_layer=FrozenBatchNorm2d, return_indices=(1, 2, 3), freeze_indices=(0,)
)

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

# 选择激活函数
activation_fn = StarReLU() if use_star_relu else nn.ReLU(inplace=True)

transformer = SalienceTransformer(
    encoder=SalienceTransformerEncoder(
        encoder_layer=SalienceTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=activation_fn,
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
    ),
    neck=RepVGGPluXNetwork(
        in_channels_list=neck.num_channels,
        out_channels_list=neck.num_channels,
        norm_layer=nn.BatchNorm2d,
        activation=nn.SiLU if use_star_relu else nn.SiLU,
        groups=4,
    ),
    decoder=SalienceTransformerDecoder(
        decoder_layer=SalienceTransformerDecoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=activation_fn,
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
    level_filter_ratio=(0.4, 0.8, 1.0, 1.0),
    layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
    # Enable small object enhancements
    use_cross_scale_attention=True,
    use_adaptive_calibration=True,
    small_target_threshold=64,  # Pixel area threshold
    # Enable uncertainty-based RoI screening
    use_uncertainty_roi_screening=use_uncertainty_roi_screening,
    uncertainty_weight=uncertainty_weight,
    top_k_per_level=top_k_per_level,
    adaptive_roi_layers=adaptive_roi_layers,
    use_nsa=use_nsa
)

# Enhance matcher for small objects
matcher = HungarianMatcher(
    cost_class=2, 
    cost_bbox=5, 
    cost_giou=2,
    small_object_size=small_target_threshold,  # Size threshold for small objects
    small_object_iou_scale=0.9  # Lower IoU threshold for small objects (10% reduction)
)

weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
weight_dict.update({
    k + f"_{i}": v
    for i in range(transformer_dec_layers - 1)
    for k, v in weight_dict.items()
})
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
weight_dict.update({"loss_salience": 2})

# Enhanced criterion with small object optimization
criterion = HybridSetCriterion(
    num_classes, 
    matcher=matcher, 
    weight_dict=weight_dict, 
    alpha=0.25, 
    gamma=2.0,
    small_object_threshold=small_target_threshold,
    small_object_weight_factor=small_target_weight_factor,
    use_ciou_loss=use_ciou_loss
)
foreground_criterion = SalienceCriterion(noise_scale=0.0, alpha=0.25, gamma=2.0)
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

# combine above components to instantiate the model
model = SalienceDETR(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    focus_criterion=foreground_criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    min_size=800,
    max_size=1333,
)
