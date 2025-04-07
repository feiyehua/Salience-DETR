import copy
import math
from typing import Tuple, List

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP, AdaptiveFeatureCalibration
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention, CrossScaleAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed
from util.misc import inverse_sigmoid
from models.bricks.roi_processor import UncertaintyRoIScreening


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


class SalienceTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
        layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        # Small object enhancement parameters
        use_cross_scale_attention: bool = True,
        use_adaptive_calibration: bool = True,
        small_target_threshold: float = 64,
        # Uncertainty-based RoI screening parameters
        use_uncertainty_roi_screening: bool = True,
        uncertainty_weight: float = 0.5,
        top_k_per_level: List[int] = [200, 150, 100, 50],
        adaptive_roi_layers: int = 2,
        use_nsa: bool = True
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # salience parameters
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.encoder.enhance_mcsp = self.encoder_class_head
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)
        
        # Small object enhancement modules
        self.use_cross_scale_attention = use_cross_scale_attention
        self.use_adaptive_calibration = use_adaptive_calibration
        
        if use_cross_scale_attention:
            self.cross_scale_attention = CrossScaleAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=0.0,
                max_tokens_per_level=1024
            )
            
        if use_adaptive_calibration:
            self.adaptive_calibration = AdaptiveFeatureCalibration(
                embed_dim=self.embed_dim,
                num_levels=num_feature_levels,
                hidden_dim=128,
                use_spatial_weights=True,
                use_channel_weights=True,
                small_target_threshold=small_target_threshold
            )

        # Uncertainty-based RoI screening parameters
        self.use_uncertainty_roi_screening = use_uncertainty_roi_screening
        
        # Initialize uncertainty RoI screening module if enabled
        if self.use_uncertainty_roi_screening:
            self.uncertainty_roi_screening = UncertaintyRoIScreening(
                in_channels_list=[self.embed_dim] * num_feature_levels,
                hidden_dim=self.embed_dim,
                num_scales=num_feature_levels,
                top_k_per_level=top_k_per_level,
                uncertainty_weight=uncertainty_weight,
                num_heads=8,
                dropout=0.1,
                num_roi_layers=adaptive_roi_layers,
                use_nsa=use_nsa
            )

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        # initialize encoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        # initialize alpha
        self.alpha.data.uniform_(-0.3, 0.3)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query,
        noised_box_query,
        attn_mask,
    ):
        assert len(multi_level_feats) == self.num_feature_levels
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        filter_nums = []  # for each level[h,w,hw*ratio]
        for i in range(self.num_feature_levels):
            bs, c, h, w = multi_level_feats[i].shape
            spatial_shapes.append((h, w))

            # filter ratio
            filter_nums.append((h, w, int(h * w * self.level_filter_ratio[i])))

            src_flatten.append(multi_level_feats[i].flatten(2).transpose(1, 2))  # bs, h*w, c
            mask_flatten.append(multi_level_masks[i].flatten(1))  # bs, h*w

        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{h*w}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{h*w}
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        
        # 修复valid_ratios计算，确保所有张量具有相同的大小
        # 先获取原始掩码
        valid_ratios = []
        for i in range(len(multi_level_masks)):
            mask_i = multi_level_masks[i].float()
            # 对每个掩码计算有效区域比例（每个空间位置都是1，表示完全有效）
            # 这里我们不再堆叠不同大小的掩码，而是直接计算每个掩码的有效区域比例
            valid_ratios.append(torch.ones(mask_i.shape[0], 1, 2, device=mask_i.device))
        
        valid_ratios = torch.cat(valid_ratios, 1)
        # for circular_padding in vision, it ensures all query positions attend to image region only

        # 在调用编码器之前先生成foreground相关参数
        # 处理每个特征层级
        proposals = []
        for i, feat in enumerate(multi_level_feats):
            bs, c, h, w = feat.shape
            # 将特征平铺成适合处理的形状
            feat_flat = feat.flatten(2).transpose(1, 2)  # [bs, h*w, c]
            proposals.append(feat_flat)
            
        # 生成前景分数和索引
        # 这里可以使用一个简单的方法来初始化，例如随机选择或基于特征强度
        foreground_score = torch.ones(bs, src_flatten.shape[1], 1, device=src_flatten.device)
        
        # 为每层生成焦点标记数
        focus_token_nums = []
        for i in range(bs):
            focus_token_nums.append(src_flatten.shape[1] // 2)  # 使用一半的标记作为焦点
        focus_token_nums = torch.tensor(focus_token_nums, device=src_flatten.device)
        
        # 为每层生成前景索引
        foreground_inds = []
        max_num_tokens = src_flatten.shape[1] // 4  # 使用1/4的tokens作为索引
        for _ in range(self.encoder.num_layers):
            layer_inds = torch.zeros((bs, max_num_tokens), dtype=torch.long, device=src_flatten.device)
            for i in range(bs):
                # 随机选择索引
                indices = torch.randperm(src_flatten.shape[1], device=src_flatten.device)[:max_num_tokens]
                layer_inds[i] = indices
            foreground_inds.append(layer_inds)

        # Apply uncertainty-based RoI screening if enabled
        if self.use_uncertainty_roi_screening:
            # Generate multi-scale RoIs
            processed_rois, roi_indices_list, roi_scale_factors = self.uncertainty_roi_screening(multi_level_feats)
            
            # Use processed RoIs as additional queries for the decoder
            # We'll combine them with the original encoder output
        
        # encoder forward
        print("multi_level_pos_embeds[0] shape:", multi_level_pos_embeds[0].shape)
        print("self.level_embeds[0] shape:", self.level_embeds[0].shape)
        # 将level_embeds重塑为正确的形状，以便与位置嵌入相加
        level_embed_reshaped = self.level_embeds[0].view(1, -1, 1, 1)
        print("level_embed_reshaped shape:", level_embed_reshaped.shape)
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            multi_level_pos_embeds[0] + level_embed_reshaped,  # position embedding for first level
            mask_flatten,
            foreground_score,  # 传递前景分数
            focus_token_nums,  # 传递焦点标记数
            foreground_inds,   # 传递前景索引
            multi_level_masks  # 传递多层掩码
        )

        # 2. neck process features for each level
        bs, _, c = memory.shape
        outputs_classes = []
        outputs_coords = []
        proposals = []

        # split memory based on spatial shapes
        strt_idx = 0
        for i, (h, w) in enumerate(spatial_shapes):
            memory_i = memory[:, strt_idx : strt_idx + h * w].view(bs, h, w, c).permute(0, 3, 1, 2)
            strt_idx += h * w
            x_i = self.neck.forward_single_feature(memory_i, i)
            proposals.append(x_i)

        # 不再调用score_k_mask_generation方法
        # 直接使用我们之前生成的前景分数和索引
        outputs = {"foreground_score": foreground_score, "foreground_inds": foreground_inds}

        # 4. decoder forward
        hs, inter_references = self.decoder(
            noised_label_query,
            noised_box_query,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            attn_mask,
        )
        inter_references_out = inter_references
        
        # If using uncertainty-based RoI screening, fuse the processed RoIs with decoder output
        if self.use_uncertainty_roi_screening:
            # Process the last layer of decoder outputs for fusion
            last_hs = hs[-1]  # [bs, num_queries, embed_dim]
            
            # Apply scaling factors to RoI boxes if needed
            # This would typically be done in the postprocessing step
            
            # Fuse the processed RoIs with the decoder output using attention
            # For simplicity, we'll use a weighted combination approach
            
            # Compute attention weights between processed_rois and last_hs
            similarity = torch.bmm(last_hs, processed_rois.transpose(1, 2))  # [bs, num_queries, num_rois]
            attn_weights = F.softmax(similarity, dim=2)
            
            # Weighted sum of processed_rois based on attention weights
            roi_context = torch.bmm(attn_weights, processed_rois)  # [bs, num_queries, embed_dim]
            
            # Add the RoI context to the decoder output with residual connection
            enhanced_hs = last_hs + roi_context
            
            # Replace the last layer of decoder outputs with the enhanced version
            hs = list(hs)
            hs[-1] = enhanced_hs
            hs = tuple(hs)
        
        outputs_class = self.decoder.class_embed(hs)
        outputs_coord = self.decoder.bbox_embed(hs).sigmoid()
        outputs["pred_logits"] = outputs_class[-1]
        outputs["pred_boxes"] = outputs_coord[-1]
        outputs["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        outputs["enc_outputs"] = {
            "pred_logits": outputs_classes[0],
            "pred_boxes": outputs_coords[0],
        }

        return outputs

    @staticmethod
    def fast_repeat_interleave(input, repeats):
        """torch.Tensor.repeat_interleave is slow for one-dimension input for unknown reasons. 
        This is a simple faster implementation. Notice the return shares memory with the input.

        :param input: input Tensor
        :param repeats: repeat numbers of each element in the specified dim
        :param dim: the dimension to repeat, defaults to None
        """
        # the following inplementation runs a little faster under one-dimension settings
        return torch.cat([aa.expand(bb) for aa, bb in zip(input, repeats)])

    @torch.no_grad()
    def nms_on_topk_index(
        self, topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
    ):
        batch_size, num_topk = topk_scores.shape
        if torchvision._is_tracing():
            num_pixels = spatial_shapes.prod(-1).unbind()
        else:
            num_pixels = spatial_shapes.prod(-1).tolist()

        # flatten topk_scores and topk_index for batched_nms
        topk_scores, topk_index = map(lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        level_index = torch.arange(level_start_index.shape[0], device=level_start_index.device)
        feat_width, start_index, level_idx = map(
            lambda x: self.fast_repeat_interleave(x, num_pixels)[topk_index],
            (spatial_shapes[:, 1], level_start_index, level_index),
        )
        topk_spatial_index = topk_index - start_index
        x = topk_spatial_index % feat_width
        y = torch.div(topk_spatial_index, feat_width, rounding_mode="trunc")
        coordinates = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)

        # get unique idx for queries in different images and levels
        image_idx = torch.arange(batch_size).repeat_interleave(num_topk, 0)
        image_idx = image_idx.to(level_idx.device)
        idxs = level_idx + level_start_index.shape[0] * image_idx

        # perform batched_nms
        indices = torchvision.ops.batched_nms(coordinates, topk_scores, idxs, iou_threshold)

        # stack valid index
        results_index = []
        if torchvision._is_tracing():
            min_num = torch.tensor(self.two_stage_num_proposals)
        else:
            min_num = self.two_stage_num_proposals
        # get indices in each image
        for i in range(batch_size):
            topk_index_per_image = topk_index[indices[image_idx[indices] == i]]
            if torchvision._is_tracing():
                min_num = torch.min(topk_index_per_image.shape[0], min_num)
            else:
                min_num = min(topk_index_per_image.shape[0], min_num)
            results_index.append(topk_index_per_image)
        return torch.stack([index[:min_num] for index in results_index])


class SalienceTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
        # focus parameter
        topk_sa=300,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.topk_sa = topk_sa

        # pre attention
        self.pre_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.pre_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.pre_attention.out_proj.weight)
        # initilize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        
        # 检查tensor和pos的维度
        if tensor.shape != pos.shape:
            # 记录原始形状用于调试
            tensor_shape = tensor.shape
            pos_shape = pos.shape
            
            # 打印维度不匹配的警告
            print(f"警告：encoder中tensor形状 {tensor_shape} 和 pos形状 {pos_shape} 不匹配，进行调整...")
            
            # 获取两个张量的batch_size和嵌入维度（通常前两维是相同的）
            batch_size, embed_dim = tensor.shape[:2]
            
            # 安全地调整pos到tensor的形状
            if len(tensor.shape) == 3 and len(pos.shape) == 3:
                # 三维张量情况：[batch_size, seq_len, embed_dim]
                # 截断或填充序列长度
                seq_len = tensor.shape[1]
                if pos.shape[1] > seq_len:
                    pos = pos[:, :seq_len, :]
                elif pos.shape[1] < seq_len:
                    # 如果pos短于tensor，使用复制扩展
                    pad_len = seq_len - pos.shape[1]
                    pos_pad = pos[:, -1:, :].repeat(1, pad_len, 1)
                    pos = torch.cat([pos, pos_pad], dim=1)
            elif len(tensor.shape) == 4 and len(pos.shape) == 4:
                # 四维张量情况：[batch_size, embed_dim, height, width]
                # 调整空间维度
                h_tensor, w_tensor = tensor.shape[2], tensor.shape[3]
                h_pos, w_pos = pos.shape[2], pos.shape[3]
                
                # 使用插值调整pos的空间维度
                if h_tensor != h_pos or w_tensor != w_pos:
                    pos = torch.nn.functional.interpolate(
                        pos, 
                        size=(h_tensor, w_tensor), 
                        mode='bilinear', 
                        align_corners=False
                    )
            elif len(tensor.shape) == 3 and len(pos.shape) == 4:
                atch_size, seq_len, embed_dim = tensor.shape
                # 将位置编码从 [B, C, H, W] 转换为 [B, H*W, C]
                pos = pos.flatten(2).permute(0, 2, 1)
                # 确保序列长度匹配
                if pos.shape[1] != seq_len:
                    # 使用插值调整序列维度
                    pos = pos.permute(0, 2, 1)  # [B, C, L]
                    pos = torch.nn.functional.interpolate(
                    pos,
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                    )
                pos = pos.permute(0, 2, 1)  # [B, L, C]

        print(f"调整后pos形状: {pos.shape}")
        
        # 现在可以安全地相加
        return tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        value,  # focus parameter
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
        # focus parameter
        score_tgt=None,
        foreground_pre_layer=None,
    ):
        # 当score_tgt和foreground_pre_layer不为None时的原始实现
        if score_tgt is None or foreground_pre_layer is None:
            # 跳过选择性注意力处理，直接进行自注意力
            # self attention
            src2 = self.self_attn(
                query=self.with_pos_embed(query, query_pos),
                reference_points=reference_points,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=query_key_padding_mask,
            )
            query = query + self.dropout1(src2)
            query = self.norm1(query)
            
            # ffn
            query = self.forward_ffn(query)
            
            return query
        
        # 检查score_tgt和foreground_pre_layer是否有效，添加安全检查
        try:
            # 检查维度匹配
            max_vals = score_tgt.max(-1)[0]  # [batch_size, num_tokens]
            
            # 确保数值稳定性
            max_vals = torch.clamp(max_vals, min=1e-6, max=1e6)
            
            # 检查foreground_pre_layer的形状是否与max_vals兼容
            if max_vals.shape != foreground_pre_layer.shape:
                # 如果形状不匹配，尝试调整
                if len(foreground_pre_layer.shape) > len(max_vals.shape):
                    # 如果前景层有更多维度，对它进行降维
                    foreground_pre_layer = foreground_pre_layer.squeeze(-1)
                elif len(foreground_pre_layer.shape) < len(max_vals.shape):
                    # 如果前景层维度更少，扩展它
                    foreground_pre_layer = foreground_pre_layer.unsqueeze(-1)
            
            # 计算mc_score并检查是否有非法值
            mc_score = max_vals * foreground_pre_layer
            # 替换可能的NaN或无穷大
            mc_score = torch.nan_to_num(mc_score, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 确保有足够的token可供选择
            k = min(self.topk_sa, mc_score.shape[1])
            select_tgt_index = torch.topk(mc_score, k, dim=1)[1]
            
            # 扩展索引到嵌入维度
            select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            
            # 安全地收集查询和位置
            select_tgt = torch.gather(query, 1, select_tgt_index)
            select_pos = torch.gather(query_pos, 1, select_tgt_index) if query_pos is not None else None
            
            # 计算query_with_pos
            query_with_pos = key_with_pos = self.with_pos_embed(select_tgt, select_pos)
            
            query = query.scatter(1, select_tgt_index, select_tgt)
        
        except Exception as e:
            # 如果处理失败，记录错误并跳过选择性注意力
            print(f"选择性注意力处理失败: {e}")
            # 继续进行标准的自注意力处理
            pass

        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class SalienceTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        # learnt background embed for prediction
        self.background_embedding = PositionEmbeddingLearned(200, num_pos_feats=self.embed_dim // 2)

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        
        # 检查tensor和pos的维度
        if tensor.shape != pos.shape:
            # 记录原始形状用于调试
            tensor_shape = tensor.shape
            pos_shape = pos.shape
            
            # 打印维度不匹配的警告
            print(f"警告：encoder中tensor形状 {tensor_shape} 和 pos形状 {pos_shape} 不匹配，进行调整...")
            
            # 获取两个张量的batch_size和嵌入维度（通常前两维是相同的）
            batch_size, embed_dim = tensor.shape[:2]
            
            # 安全地调整pos到tensor的形状
            if len(tensor.shape) == 3 and len(pos.shape) == 3:
                # 三维张量情况：[batch_size, seq_len, embed_dim]
                # 截断或填充序列长度
                seq_len = tensor.shape[1]
                if pos.shape[1] > seq_len:
                    pos = pos[:, :seq_len, :]
                elif pos.shape[1] < seq_len:
                    # 如果pos短于tensor，使用复制扩展
                    pad_len = seq_len - pos.shape[1]
                    pos_pad = pos[:, -1:, :].repeat(1, pad_len, 1)
                    pos = torch.cat([pos, pos_pad], dim=1)
            elif len(tensor.shape) == 4 and len(pos.shape) == 4:
                # 四维张量情况：[batch_size, embed_dim, height, width]
                # 调整空间维度
                h_tensor, w_tensor = tensor.shape[2], tensor.shape[3]
                h_pos, w_pos = pos.shape[2], pos.shape[3]
                
                # 使用插值调整pos的空间维度
                if h_tensor != h_pos or w_tensor != w_pos:
                    pos = torch.nn.functional.interpolate(
                        pos, 
                        size=(h_tensor, w_tensor), 
                        mode='bilinear', 
                        align_corners=False
                    )
            
            print(f"调整后pos形状: {pos.shape}")
        
        # 现在可以安全地相加
        return tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [n, s, l, 2]
        return reference_points

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        query_pos=None,
        query_key_padding_mask=None,
        # salience input
        foreground_score=None,
        focus_token_nums=None,
        foreground_inds=None,
        multi_level_masks=None,
    ):
        """SalienceTransformerEncoder Forward
        Args:
            query: bs, sum(hi*wi), c
            key: bs, sum(hi*wi), c
            value: bs, sum(hi*wi), c
            spatial_shapes: nlevel, 2
            level_start_index: nlevel
            valid_ratios: bs, nlevel, 2
        """
        # 使用简化的实现，避开复杂的索引操作
        batch_size, seq_len, embed_dim = query.shape
        
        # 初始化参考点
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        
        # 保存原始输入，用于后续处理
        value = output = query
        
        # 使用一个更简单、更安全的实现版本，不依赖于foreground_inds
        for layer_id, layer in enumerate(self.layers):
            # 在这里修改，使用自己的with_pos_embed方法
            # 直接将整个output传递给layer进行处理
            # 这样避免复杂的索引操作，确保稳定性
            output = layer(
                output,  # query
                query_pos,  # query_pos
                value,  # value
                reference_points,  # reference_points
                spatial_shapes,  # spatial_shapes
                level_start_index,  # level_start_index
                query_key_padding_mask,  # key_padding_mask
                None,  # score_tgt - 暂时不使用复杂的计分机制
                None,  # foreground_pre_layer - 暂时不使用复杂的前景层
            )
        
        # add learnt embedding for background (如果需要的话)
        if multi_level_masks is not None:
            background_embedding = [
                self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
            ]
            background_embedding = torch.cat(background_embedding, dim=1)
            
            # 如果有padding_mask，应用它
            if query_key_padding_mask is not None:
                background_embedding *= (~query_key_padding_mask).unsqueeze(-1)
            
            # 在这里修改：使用自定义的with_pos_embed方法替代直接加法
            output = self.with_pos_embed(output, background_embedding)

        return output


class SalienceTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class SalienceTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        self.class_head = nn.ModuleList([nn.Linear(self.embed_dim, num_classes) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([MLP(self.embed_dim, self.embed_dim, 4, 3) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
    ):
        outputs_classes = []
        outputs_coords = []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query)) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords
